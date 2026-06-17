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
from ..types import ScopedTask


# Strategy hints cycled across attempts to preserve diversity (decorrelates
# hallucinations — the whole reason best-of-N works).
DEFAULT_STRATEGIES = ("minimal", "comprehensive", "test_driven", "edge_case_hardening")

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
    if fs == "policy_violation" or "outside the root" in el or "permission denied" in el:
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
        repair_iters: int = 0,
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
        # HARD CEILING on test-driven repair depth (default 0 == OFF == flat best-of-N).
        # Clamps every solve_and_repair/make_repairing_attempt call, so repair is opt-in
        # regardless of what an authored orchestrator requests. run-4 showed repair-ON by
        # default blew the time budget; it is now off unless explicitly enabled.
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
        self.governor = RunGovernor(
            engine=engine, agent_ceiling=engine.max_total_agents,
            token_budget=engine.budget.total, agent_budget=self.max_agents, plateau_k_dry=2,
        )
        # Backbone 2.4 CUT-LOSSES detector state. Progress is tracked on the BEST-so-far
        # distance-to-solve (gold ids green = ``_best_gold_passed``; raw pass_rate as the
        # secondary tier), NEVER the last wave — so a refactor dip that recovers is not a
        # cut, and a high-but-flat pass_rate (no new gold) still counts as dry. Plus two
        # hard-cut streaks (all-nonresult waves; sterile/identical-diff waves).
        self._dry_rounds = 0                  # telemetry: consecutive dry WAVES (not the cut unit)
        self._best_pass_rate = 0.0
        self._best_gold_passed = 0
        self._nonresult_streak = 0
        self._sterile_streak = 0
        self._seen_shas: set = set()
        self._tokens_at_best = 0
        self._agents_at_best = 0              # agents_used when the BEST last improved (cut unit)
        self._halt_reason = ""
        self._halt_is_cut = False
        self._halted = False
        self._all_candidates: list = []
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
        # --- CUT-LOSSES accounting over the returned candidates (FRESH or CACHED, so the
        # detector state is faithfully reconstructed during a resume replay) ---
        round_gold = self._best_gold_passed
        round_pass = self._best_pass_rate
        n_attempts = len(out)
        n_nonresult = 0          # attempts that produced no usable work
        n_sterile = 0            # attempts with an empty diff OR a diff already seen this run
        any_new_useful = False   # at least one attempt produced a NEW non-empty diff
        for c in out:
            if c is None:
                n_nonresult += 1
                n_sterile += 1   # a None result is both no-work and no-useful-diff
                continue
            m = getattr(c, "meta", {}) or {}
            gp = int(m.get("gold_passed", 0) or 0)
            pr = getattr(c, "public_signal_score", None)
            pr = pr if (isinstance(pr, (int, float)) and not isinstance(pr, bool)) else 0.0
            if gp > round_gold:
                round_gold = gp
            if pr > round_pass:
                round_pass = pr
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
        improved = (round_gold > self._best_gold_passed) or (round_pass > self._best_pass_rate + 1e-9)
        if improved:
            self._best_gold_passed = round_gold
            self._best_pass_rate = round_pass
            self._dry_rounds = 0
            self._tokens_at_best = self._engine.budget.spent()
            self._agents_at_best = self._engine.agents_used()
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
        self._wave_verdict(self._wave_state())   # sets self._halted on a halt verdict
        return out

    def _wave_state(self) -> dict:
        """The cut-losses detector inputs at the current wave boundary. All cut signals are in
        ATTEMPTS (agents), so the rule is invariant to the wave schedule and the arm width."""
        agents = self._engine.agents_used()
        return {
            "attempts_since_improvement": max(0, agents - self._agents_at_best),
            "dry_rounds": self._dry_rounds,          # telemetry only (not a cut signal)
            "agents_used": agents,
            "nonresult_streak": self._nonresult_streak,
            "sterile_streak": self._sterile_streak,
            "tokens_since_improvement": max(0, self._engine.budget.spent() - self._tokens_at_best),
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
                 checkpoint: bool = True) -> Optional[Candidate]:
        """The SHARED body of every scored attempt (Backbone 2.2 extraction). Acquire a
        fresh worktree, run ONE journaled coding agent, materialize its diff, score it by
        execution, build a ranked Candidate, bank it, and (if accepted) checkpoint. This
        is the ONLY place an attempt is scored, so solve and repair are mechanically
        identical and journal-replay-safe — and neither can mark itself accepted.

        ``prefix`` is the candidate-id/worktree prefix ('a' solve, 'r' repair);
        ``node_prefix`` is the journal node-id prefix. Returns a Candidate, or None on infra
        failure. The attempt runs with internet OFF (iron-tight) and NO anti-fetch prompt:
        cheating is prevented STRUCTURALLY (worktree shadows site-packages + no network), not by
        limiting the model; a blocked escape is recorded as telemetry, never penalized."""
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
                      attempt_id: Optional[int] = None, checkpoint: bool = True) -> Optional[Candidate]:
        """Run ONE isolated coding-agent attempt and score it by execution.
        Returns a ranked Candidate (accepted iff the visible suite is green) or
        None on a hard infra failure.  Worktree lifecycle is managed by ``_attempt`` —
        the generated code never touches the filesystem."""
        aid = self._next_attempt_id() if attempt_id is None else _as_int_id(attempt_id)
        strat = strategy or self.strategies[aid % len(self.strategies)]
        task_prompt = prompt or self._prompt_builder(self, aid, strat)
        return self._attempt(aid=aid, prefix="a", node_prefix="attempt", prompt=task_prompt,
                             strategy=strat, vendor=vendor, model=model, checkpoint=checkpoint)

    # ---- read-only schema'd sub-question: a SIGNAL, never a Candidate ----
    def ask(self, prompt: str, *, schema: Optional[dict] = None, vendor: Optional[str] = None,
            model: Optional[str] = None, agent_id: Optional[int] = None):
        """Ask a coding agent a READ-ONLY sub-question over the source repo (Backbone 2.2).

        Generalizes the scout: runs in a fresh **forced read-only** session (no worktree,
        no diff, no score), is journaled/replayable, and returns the agent's
        ``structured_output`` dict (when ``schema`` is given) or its final text — a SIGNAL
        the orchestrator may use to STEER compute (which files, which approach, refute a
        candidate). It can NEVER produce a Candidate or touch acceptance, so a pattern
        built on it cannot promote an unverified solve. Returns None on infra failure.

        An explicit ``agent_id`` makes the question replayable; when omitted (review-fix)
        the id is DERIVED from a stable hash of (prompt, schema, vendor, model) — NOT the
        scheduling-dependent shared counter — so an unseeded ask is still replay-deterministic
        under concurrent fan-out (identical questions collide and replay; different ones don't)."""
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
            res = self._engine.agent(
                ScopedTask(prompt=str(prompt), schema=schema, sandbox="read-only",
                           model=model or spec.model, vendor=spec.vendor,
                           timeout_seconds=self.per_agent_timeout_seconds,
                           scoped_inputs={"ask": (f"{self._node_ns}{aid}" if self._node_ns else aid),
                                          "repo_snapshot_sha": self._provider.base_commit}),
                lambda t: session.run(t), node_id=f"{self._node_ns}ask{aid}",
                cli_version=getattr(session, "cli_version", ""), agent_type="ask",
            )
        except Exception as exc:
            self.log(f"ask {aid}: {type(exc).__name__}: {exc}")
            return None
        if not res.ok:
            return None
        if schema is not None:
            return res.structured_output if isinstance(res.structured_output, dict) else None
        return res.final_message or ""

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
        # Phase 3 (redact): by DEFAULT the raw pytest failure tail is DROPPED — it can carry
        # asserted-equal RHS answer values. The sanitized failing node ids below are the
        # Reflexion signal. With APEX_OMEGA_REPAIR_EXCERPTS=1 the tail is included but ONLY
        # after redact_excerpts() reduces it to sanitized node ids (review-fix #6: it used to
        # be injected raw, defeating the firewall the redactor exists to enforce).
        excerpt_block = ""
        if excerpts and os.environ.get("APEX_OMEGA_REPAIR_EXCERPTS") == "1":
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

    # ---- RALPH-WIGGUM baseline: naive persistence with feedback on one lineage ----
    def ralph_loop(self, *, id_base: int = 800000) -> Optional[Candidate]:
        """The ralph-wiggum baseline body: ONE sequential lineage, fed the failing tests
        each turn — naive iterate-until-done persistence. NO scout / author / patterns /
        parallel waves. Iteration 0 is a fresh solve; each later iteration REPAIRS the
        running best, seeded with the full accumulated diff + the failing test ids
        (Reflexion-style feedback). Each iteration is routed through ``ctx.parallel`` as a
        "wave of one", so the SAME cut-losses detector that governs omega governs ralph —
        an apples-to-apples persistence test. Stops on accept, on a governor cut (no
        progress / dead state / ceiling), and returns the best banked candidate (or
        abstains — never fakes a pass).

        Distinct from ``baseline`` (K independent THROWAWAY parallel rollouts, no feedback)
        and from omega (scout/author/parallel waves): ralph is sequential persistence on a
        growing single lineage. The full parent diff is carried (``_repair_diff_limit``
        raised) so iteration N+1 builds on N's work rather than a truncated summary."""
        # naive-persistence fidelity: carry the WHOLE accumulated diff forward, not a hint.
        self._repair_diff_limit = max(self._repair_diff_limit, 200_000)
        lineage: list = []
        cur: Optional[Candidate] = None
        k = 0
        while True:
            aid = id_base + k
            if cur is None:
                thunk = (lambda a=aid: self.solve_attempt(attempt_id=a))
            else:
                thunk = (lambda p=cur, a=aid: self.repair_attempt(p, attempt_id=a))
            try:
                out = self.parallel([thunk])   # raises CutLosses/PlateauStop once halted
            except PlateauStop:                 # CutLosses is a subclass -> also caught
                break
            k += 1
            cand = out[0] if out else None
            if cand is None:
                continue
            lineage.append(cand)
            # carry the STRONGEST state forward so the next turn repairs the best, not a dip.
            if (cur is None or cand.accepted
                    or (cand.public_signal_score or 0.0) >= (cur.public_signal_score or 0.0)):
                cur = cand
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
