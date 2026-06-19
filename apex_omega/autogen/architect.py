"""The architect: scout the repo, author a tailored orchestration script, FREEZE
it (content-hash + journal), execute it in the sandbox, and fail open to the
verified best-of-N floor (plan §7.3 / §2 / §22).

This is the generated-code orchestration approach: a planner agent emits real
Python (using only the ``ctx`` API) so the strategy is fully flexible and can
scale to thousands of agents — but it is frozen for deterministic replay, lint-
and capability-checked, and can never weaken the execution-grounded acceptance
gate. "Task completion above all" = escalate compute until a VERIFIED pass or a
hard ceiling; the floor guarantees we never do worse than v1's best-of-N.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ..ablation.arms import AblationConfig
from ..engine.runtime import Engine
from ..errors import FailLoud, PlateauStop
from ..journal.key import sha256_hex
from ..types import ScopedTask
from .context import OrchestrationContext
from .sandbox import extract_code, lint_source, run_orchestration
from .templates import (
    BEST_OF_N_ORCHESTRATION,
    CONVERGE_EXEMPLAR,
    DECOMPOSE_EXEMPLAR,
    DEFAULT_ORCHESTRATION,
    RALPH_ORCHESTRATION,
)


API_REFERENCE = """\
You are writing a Python function `orchestrate(ctx)` that orchestrates coding
agents to SOLVE a repository task. You may use ONLY the `ctx` object and plain
Python control flow (for/while/if/def/lambda/comprehensions, len/range/min/max/
sorted/sum/enumerate/zip). NO imports, NO os/sys/subprocess/open/time/random,
NO dunder access. The runtime injects everything you need via `ctx`:

  ctx.phase(title) / ctx.log(msg)            # narration (never affects results)
  ctx.repo_map                                # dict: discovered files/modules/tests/difficulty
  ctx.worker_specs                            # list of workers; each works as a tuple (w[0]=vendor, w[1]=model),
                                              #   a dict (w["vendor"]/w["model"]), an attr (w.vendor/w.model), or unpack (v, m = w)
  ctx.budget.can_start() / .remaining()       # token budget; can_start() False => stop starting work
  ctx.agents_used()                           # fresh agents dispatched so far
  ctx.max_agents                              # soft ceiling you SHOULD respect
  ctx.plan_waves((1,3,5,...))                 # agent-cap-bounded escalation wave sizes
  ctx.make_attempt(i)                         # -> thunk for a diversified attempt with id i
  ctx.solve_attempt(attempt_id=i, strategy=?, vendor=?, model=?, prompt=?) -> Candidate|None
  ctx.make_repairing_attempt(i, max_iters=2)  # -> thunk: a TEST-DRIVEN REPAIR lineage
  ctx.solve_and_repair(attempt_id=i, strategy=?, vendor=?, model=?, prompt=?, max_iters=2) -> Candidate|None
                                              #   base attempt then repair passes seeded by the
                                              #   failing tests (additional work beyond a flat
                                              #   rollout; stops on accept / plateau / non-genuine
                                              #   abort / cap). Same kwargs as solve_attempt + max_iters.
  ctx.parallel(thunks) -> list[Candidate|None]    # concurrent SOLVE fan-out (engine-bounded; counts toward plateau)
  ctx.pipeline(items, *stages) -> list            # per-item streaming (no inter-stage barrier)
  ctx.select(candidates) -> Candidate|None        # EXECUTION-AUTHORITATIVE winner, or None (abstain)
  ctx.any_accepted(candidates) -> bool
  ctx.should_continue_waves() -> bool             # resume-safe wave-loop condition (stops at plateau/ceiling)

  ctx.solve_attempt(..., phase=?, label=?)        # phase/label group+name this agent in the UI (per-agent opts)
  ctx.workflow(name_or_ref, args=?)               # compose another orchestration inline (one level deep);
                                              #   names: "default-best-of-n" | "converge" | "decompose" | "audit" | "ralph"

DECOMPOSE -> CONVERGE (the powerful default shape for medium/hard MODULAR repos — decompose the
work, solve per module in parallel, then ITERATE to convergence on the exact residual failures
instead of abstaining on a near-solve). Each seam is journaled and can NEVER set acceptance:
  ctx.decompose(vendor=?, model=?) -> {"modules":[{"module","gold_test_ids","depends_on"}],"order":[...]} | None
                                              #   ONE read-only scoping agent returns a module breakdown.
                                              #   Returns None on an undecomposable repo -> fall back to best-of-N.
  ctx.carry_best() -> str                         # the running BEST partial diff (highest valid gold-pass count)
  ctx.fanout_modules(modules, carry_diff=?) -> [Candidate]
                                              #   per-module solve fan-out (ctx.pipeline, no barrier); each agent
                                              #   is seeded with carry_diff and scoped to its module's gold ids.
  ctx.solve_module(module, carry_diff=?) -> Candidate|None   # ONE module-scoped solve (carry applied pre-agent)
  ctx.reduce_residuals(cands, carry_diff=?) -> {"merged_diff","residual_failing_ids","accepted","candidate","conflicts","indeterminate"}
                                              #   PLAIN-py merge of per-module diffs into one tree + ONE full-suite
                                              #   score (zero tokens). A conflicting module is recorded in
                                              #   conflicts[] and re-solved clean — progress is NEVER erased.
  ctx.repair_residual(residual_ids, carry_diff=?, round=?) -> Candidate|None
                                              #   ONE repair agent on the LIVE merged tree, scoped to the EXACT
                                              #   still-failing node-ids (carry applied pre-agent so it edits live).

READ-ONLY SIGNALS (steer compute; they can NEVER create a Candidate or accept anything):
  ctx.ask(prompt, schema=?, vendor=?, model=?, agent_id=?, max_nudges=2, strict=False, phase=?, label=?)
                                              #   a read-only sub-question; returns structured_output (dict OR list,
                                              #   if schema) or text. When schema is set, an invalid reply is RE-ASKED
                                              #   with a nudge up to max_nudges times, then -> None (or raises if
                                              #   strict=True). Pass a fixed agent_id to make it replayable.
  ctx.signals(thunks) -> list                     # read-only ask fan-out (NOT counted as a solve wave)
  ctx.quarantined_ask(question, untrusted_content, schema=?) -> dict|str|None
                                              #   analyze UNTRUSTED content with an anti-injection read-only agent

QUALITY PATTERNS (compose these; each DEGRADES to plain best-of-N at zero knobs and can
only re-rank/downgrade/extend — never promote an unverified solve):
  ctx.adversarial_verify(cand, n=3, refute_if="majority") -> Candidate
                                              #   independent skeptics try to REFUTE an accepted cand;
                                              #   downgrades it if they do (guards against cheated/incomplete passes)
  ctx.adversarial_filter(items, votes=3) -> items # ADMIT-gate plain-data findings: keep only survivors
  ctx.judge_panel(cands, lenses=[...]) -> cands   # attach a SOFT tiebreak score (sub-execution; never an accept)
  ctx.judge_select(cands, lenses=[...]) -> Candidate|None   # judge_panel then ctx.select (the winner)
  ctx.tournament(cands, lens=?) -> cands          # pairwise round-robin -> SOFT win-rate tiebreak (re-rank w/ select)
  ctx.classify_and_route(items, classify=fn, routes={cat: handler}) -> [result]
                                              #   classify each item, dispatch to its handler (e.g. cheap vs strong model)
  ctx.synthesize(cands, attempt_id=i, top_k=3) -> Candidate|None
                                              #   combine the best PARTIAL solutions into ONE new
                                              #   execution-scored attempt (the legit accept path)
  ctx.loop_until_dry(make_round, k_dry=2) -> [Candidate]
                                              #   make_round(i)->[thunk]; widen waves until accept / K dry rounds / plateau
  ctx.completeness_critic(cand) -> {complete, gaps, recommendation}   # read-only "what's missing?" signal

A Candidate has .accepted (bool, EXECUTION evidence — read-only to you), .combined_score,
.public_signal_score (pass_rate), .meta (incl .meta["failing_nodeids"]).
"""

INVARIANTS = """\
HARD RULES (the run is invalid if you break them):
  1. The ONLY way to declare success is `ctx.select(candidates)` returning a
     non-None, .accepted candidate. You cannot mark anything accepted yourself.
  2. TASK COMPLETION IS THE TOP PRIORITY: keep escalating (more attempts, more
     diversity, more vendors, decomposition) until ctx.select returns an accepted
     winner OR budget/agent ceiling is hit. NEVER return an unverified guess.
  3. FEWEST AGENTS FIRST: start with 1 attempt; widen only while unsolved. Respect
     ctx.budget.can_start() and ctx.max_agents.
  4. Assign attempt ids deterministically at creation (e.g. base+j), NOT at call
     time, so the run is replayable.
  5. Return the result of ctx.select(...) (a Candidate or None). Do not loop forever;
     prefer `while ctx.should_continue_waves():` for open-ended escalation (it is
     resume-deterministic and stops itself at plateau/ceiling).
  6. You may NOT set or fake acceptance. `c.accepted = True` is a hard lint error.
     Patterns (verify/judge/synthesize/critic) steer compute but cannot promote an
     unverified candidate — acceptance is earned ONLY by a green visible suite.
"""

# A compact, copy-adaptable composition: escalate waves until plateau, harden the
# winner with adversarial verification, and synthesize partials if none passed.
PATTERN_EXEMPLAR = """\
def orchestrate(ctx):
    ctx.phase("solve")
    cands = []
    # 1) escalate best-of-N waves until an accepted solve, plateau, or the ceiling.
    base = 0
    while ctx.should_continue_waves():
        wave = [ctx.make_attempt(base + j) for j in range(max(1, ctx.initial_agents))]
        base += len(wave)
        cands += ctx.parallel(wave)
        if ctx.any_accepted(cands):
            break
    # 2) no clean pass yet? synthesize the best partials into one new scored attempt.
    if not ctx.any_accepted(cands):
        syn = ctx.synthesize(cands, attempt_id=base + 1)
        base += 1
        if syn is not None:
            cands.append(syn)
    # 3) harden a candidate before shipping: skeptics can only DOWNGRADE a cheat/incomplete.
    winner = ctx.select(cands)
    if winner is not None:
        ctx.adversarial_verify(winner, n=3)
        winner = ctx.select(cands)   # re-rank after any downgrade
    return winner
"""


@dataclass
class FrozenWorkflow:
    source: str
    content_sha: str
    origin: str            # "authored" | "template" | "fallback"
    lint_ok: bool
    lint_violations: list

    def to_dict(self) -> dict:
        return {"content_sha": self.content_sha, "origin": self.origin,
                "lint_ok": self.lint_ok, "lint_violations": self.lint_violations}


def build_repo_map(source_repo: str, *, base_commit: Optional[str] = None,
                   extra: Optional[dict] = None, max_files: int = 4000) -> dict:
    """Lightweight scout: enumerate source modules + test files + a difficulty
    proxy.  ``extra`` (issue description, test command, expected ids) is merged
    in.  Reuses v1's contract slice opportunistically if available."""
    root = Path(source_repo)
    py_files, test_files, modules = [], [], set()
    for p in root.rglob("*.py"):
        rel = p.relative_to(root).as_posix()
        if any(seg in rel for seg in (".git/", ".venv/", "site-packages/", "__pycache__/")):
            continue
        if "test" in Path(rel).name:
            test_files.append(rel)
        else:
            py_files.append(rel)
            top = rel.split("/", 1)[0]
            modules.add(top if top.endswith(".py") is False else top[:-3])
        if len(py_files) + len(test_files) >= max_files:
            break
    n = len(py_files)
    difficulty = "easy" if n < 15 else ("medium" if n < 80 else "hard")
    repo_map = {
        "source_repo": str(root),
        "base_commit": base_commit,
        "n_source_files": n,
        "n_test_files": len(test_files),
        "modules": sorted(m for m in modules if m and not m.endswith(".py"))[:50],
        "sample_source_files": py_files[:40],
        "sample_test_files": test_files[:40],
        "difficulty": difficulty,
    }
    if extra:
        repo_map.update(extra)
    return repo_map


# Pipeline-vs-parallel decision rule (dynamic-workflows parity §4.1), taught to the author so
# generated orchestrations default to the latency-optimal primitive.
_PIPELINE_VS_PARALLEL_RULE = (
    "\nDECISION RULE — pipeline vs parallel: DEFAULT to ctx.pipeline(items, *stages) for "
    "independent per-item multi-stage work so fast items never wait for slow ones. Reach for "
    "ctx.parallel(thunks) ONLY when a downstream step needs ALL prior results at once "
    "(ctx.select, global dedup, cross-item ranking, early-exit on a total). A transform with no "
    "cross-item dependency belongs INSIDE a pipeline stage, not behind a parallel barrier. Use "
    "ctx.signals(thunks) for read-only verifier/judge fan-out (it does NOT advance the "
    "non-progress plateau, so a verify wave is never mistaken for a solve wave).\n"
)


def build_author_prompt(repo_map: dict) -> str:
    # The author prompt carries the API/invariants for writing orchestrate(ctx), the orchestrator's
    # OWN scouting (the repo map), and the BINDING TASK-FRAMING rules. NO harness-derived "design
    # contract" — figuring out the API/enum/parametrization shape from the gold tests is the model's
    # job (fairness). The framing is the eval RULES (not answer hints); the orchestrator may
    # restate/amplify them to subagents but can never remove them (workers get them independently
    # via build_issue_description). Defense-in-depth: never render a stray design_contract key.
    framing = str(repo_map.get("task_framing") or "").strip()
    framing_block = ("\n\nTASK FRAMING (binding eval rules — your workers ALSO receive these "
                     "independently; you may restate/amplify but never remove them):\n" + framing) if framing else ""
    rmap = {k: v for k, v in repo_map.items()
            if k not in ("design_contract", "task_framing", "brief_builders")}
    # The PRIMARY exemplar is difficulty-gated: medium/hard MODULAR repos get the decompose->
    # converge shape (decompose -> fan-out -> reduce -> loop-until-dry on residuals), which closes
    # the off-by-K near-solve class; easy repos get the cheap best-of-N (decomposition over-spawn
    # is the cost pathology on easy repos, so we do NOT push it there).
    difficulty = str(repo_map.get("difficulty") or "").lower()
    if difficulty in ("medium", "hard"):
        primary_label = ("PRIMARY EXEMPLAR (decompose -> fan-out per module -> reduce -> "
                         "loop-until-dry on the EXACT residual failing tests, carrying the best "
                         "partial forward — the convergence shape; easy/single-module repos should "
                         "compose ctx.workflow(\"default-best-of-n\") instead)")
        primary_src = CONVERGE_EXEMPLAR.strip()
    else:
        primary_label = ("REFERENCE EXEMPLAR (a safe completion-first best-of-N you can adapt or "
                         "improve on — e.g. decompose by module, pipeline stages, route hard work "
                         "to a stronger vendor)")
        primary_src = BEST_OF_N_ORCHESTRATION.strip()
    return (
        "Write a Python function `orchestrate(ctx)` tailored to THIS repository to "
        "solve its task with the fewest agents necessary, escalating until a verified "
        "pass.\n\n" + API_REFERENCE + "\n" + INVARIANTS + _PIPELINE_VS_PARALLEL_RULE + framing_block +
        "\nDISCOVERED REPOSITORY MAP:\n" + json.dumps(rmap, indent=2)[:6000] +
        "\n\n" + primary_label + ":\n```python\n" + primary_src +
        "\n```\n\nQUALITY-PATTERN EXEMPLAR (escalate -> synthesize -> adversarially verify; "
        "adapt the patterns to the repo's difficulty — they cost more agents, so reserve "
        "verification/synthesis for harder tasks):\n```python\n" + PATTERN_EXEMPLAR.strip() +
        "\n```\n\nReturn ONLY a python code block defining orchestrate(ctx)."
    )


def _freeze(engine: Engine, source: str, origin: str, lint_ok: bool, violations: list) -> FrozenWorkflow:
    sha = sha256_hex(source)
    fw = FrozenWorkflow(source=source, content_sha=sha, origin=origin,
                        lint_ok=lint_ok, lint_violations=list(violations))
    odir = Path(engine.run_dir) / "orchestrator"
    odir.mkdir(parents=True, exist_ok=True)
    (odir / f"{sha}.py").write_text(source, encoding="utf-8")
    (odir / "frozen.json").write_text(json.dumps({**fw.to_dict(), "source_path": f"{sha}.py"}, indent=2),
                                      encoding="utf-8")
    engine.log(f"froze orchestration script origin={origin} sha={sha[:12]} lint_ok={lint_ok}")
    return fw


def load_frozen(engine: Engine) -> Optional[FrozenWorkflow]:
    """Reuse a previously-frozen orchestration on resume (do NOT re-author)."""
    fp = Path(engine.run_dir) / "orchestrator" / "frozen.json"
    if not fp.exists():
        return None
    meta = json.loads(fp.read_text(encoding="utf-8"))
    src = (Path(engine.run_dir) / "orchestrator" / meta["source_path"]).read_text(encoding="utf-8")
    return FrozenWorkflow(source=src, content_sha=meta["content_sha"], origin=meta["origin"],
                          lint_ok=meta["lint_ok"], lint_violations=meta.get("lint_violations", []))


def author_orchestration(
    engine: Engine, *, executor: Any, worker_specs: Sequence[Any], repo_map: dict,
    author: bool = True, author_fn: Optional[Callable[[dict], str]] = None,
    author_vendor: Optional[str] = None,
) -> FrozenWorkflow:
    """Author (or fall back to template), lint, and FREEZE the orchestration.
    Resumes a prior frozen script if present."""
    prior = load_frozen(engine)
    if prior is not None:
        engine.log(f"resuming frozen orchestration sha={prior.content_sha[:12]} (no re-author)")
        return prior

    # RALPH-WIGGUM baseline: freeze the fixed ralph workflow DIRECTLY (no scout, no author),
    # so it resumes deterministically like every other arm. A first-class control arm.
    _orch_selector = os.environ.get("APEX_OMEGA_ORCHESTRATION")
    if _orch_selector == "ralph":
        lint = lint_source(RALPH_ORCHESTRATION)
        return _freeze(engine, RALPH_ORCHESTRATION, "ralph", lint.ok, lint.violations)

    # CONVERGE arm (Phase 3 A/B): freeze the convergence default DIRECTLY (no author), so the
    # rebuilt decompose->fan-out->reduce->loop-until-dry orchestration is the frozen plan even
    # when author=True. DEFAULT_ORCHESTRATION already IS the convergence shape; this selector
    # pins it explicitly so Arm B is reproducible and resumes deterministically.
    if _orch_selector in ("converge", "rebuild"):
        lint = lint_source(DEFAULT_ORCHESTRATION)
        return _freeze(engine, DEFAULT_ORCHESTRATION, "converge", lint.ok, lint.violations)

    # HYBRID arm: a Claude-Code-style host-side PHASE PLANNER runs AROUND the frozen converge body
    # (autosolve calls phase_planned_solve before run_orchestration). We freeze the converge default
    # as the FALL-THROUGH body so a degenerate / abstained / crashed phase plan degrades to the
    # proven decompose->fan-out->reduce->loop-until-dry path (then the best-of-N floor). The phase
    # loop lives host-side (depth 0) to sidestep the ctx.workflow() one-level nesting cap.
    if _orch_selector == "hybrid":
        lint = lint_source(DEFAULT_ORCHESTRATION)
        return _freeze(engine, DEFAULT_ORCHESTRATION, "hybrid", lint.ok, lint.violations)

    if not author:
        lint = lint_source(DEFAULT_ORCHESTRATION)
        return _freeze(engine, DEFAULT_ORCHESTRATION, "template", lint.ok, lint.violations)

    # author via a stub (tests) or a real architect LLM call
    source = None
    try:
        if author_fn is not None:
            source = author_fn(repo_map)
        else:
            source = _author_via_llm(executor, worker_specs, repo_map, author_vendor)
    except Exception as exc:
        engine.log(f"author failed ({type(exc).__name__}: {exc}); falling back to template")
        source = None

    if source:
        source = extract_code(source)
        lint = lint_source(source)
        if lint.ok:
            return _freeze(engine, source, "authored", True, [])
        engine.log("authored orchestration failed lint: " + "; ".join(lint.violations) + "; using template")

    lint = lint_source(DEFAULT_ORCHESTRATION)
    return _freeze(engine, DEFAULT_ORCHESTRATION, "fallback", lint.ok, lint.violations)


def _author_via_llm(executor: Any, worker_specs: Sequence[Any], repo_map: dict,
                    author_vendor: Optional[str]) -> str:
    spec = next((s for s in worker_specs if s.vendor == author_vendor), worker_specs[0])
    tmp = tempfile.mkdtemp(prefix="apexomega_author_")
    session = executor.spawn(tmp, spec.vendor, spec.model, spec=getattr(spec, "extra", {}))
    res = session.run(ScopedTask(prompt=build_author_prompt(repo_map), sandbox="read-only",
                                 model=spec.model, vendor=spec.vendor))
    return res.final_message or ""


# JSON schema the scout agents return.
SCOUT_SCHEMA = {
    "type": "object", "additionalProperties": True, "required": ["difficulty"],
    "properties": {
        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
        "approach": {"type": "string"},
        "key_files": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "string"},
    },
}

# difficulty -> (INITIAL wave size, soft agent cap). Most tasks a couple/tens of
# agents; only proven-hard tasks escalate toward the 1000 hard ceiling.
_DIFFICULTY_PROFILE = {"easy": (1, 8), "medium": (3, 24), "hard": (8, 64)}
_DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}
_DIFFICULTY_INV = {0: "easy", 1: "medium", 2: "hard"}


def difficulty_profile(difficulty: Optional[str], *, ceiling: int = 1000) -> tuple[int, int]:
    initial, cap = _DIFFICULTY_PROFILE.get(str(difficulty or "").lower(), _DIFFICULTY_PROFILE["medium"])
    return min(initial, ceiling), min(cap, ceiling)


def build_scout_prompt(repo_map: dict, scout_extra: Optional[dict], i: int) -> str:
    lens = ("the implementation surface (what's missing/stubbed)",
            "the test topology, fixtures, and what the visible suite demands",
            "edge cases, hidden coupling, and risks")[i % 3]
    body = json.dumps({k: repo_map.get(k) for k in (
        "modules", "sample_source_files", "sample_test_files",
        "n_source_files", "n_test_files")}, indent=1)[:3000]
    task = ("\nTASK:\n" + json.dumps(scout_extra, indent=1)[:2000]) if scout_extra else ""
    # Give the scout the binding rules so it plans an IN-REPOSITORY implementation and does NOT
    # propose "restore/download the official upstream" approaches (which only get budget-blocked).
    framing = str(repo_map.get("task_framing") or "").strip()
    framing_block = ("\n\nTASK FRAMING (binding rules — plan within these):\n" + framing) if framing else ""
    return (
        "You are a scout assessing a repository-COMPLETION task before orchestration. "
        f"Examine the repository (focus this pass on: {lens}) and decide how to "
        "complete it and how hard it is.\n\nREPO MAP:\n" + body + task + framing_block +
        "\n\nReturn a JSON assessment: difficulty (easy|medium|hard = how hard to make "
        "the visible test suite fully pass), approach (a concrete IN-REPOSITORY completion plan: "
        "what to implement, in what order), key_files (the files to implement/edit), risks."
    )


def agent_scout(
    engine: Engine, *, executor: Any, worker_specs: Sequence[Any], source_repo: str,
    base_commit: Optional[str], base_repo_map: dict, n_scouts: int = 3,
    scout_vendor: Optional[str] = None, scout_extra: Optional[dict] = None,
) -> dict:
    """Parallel READ-ONLY scout fan-out: each agent decides HOW to complete the
    task and HOW HARD it is; we aggregate (median difficulty, merged plan).
    Journaled + replayable; fails open to the static difficulty if scouts error."""
    specs = list(worker_specs)

    def make_scout(i: int) -> Callable[[], Any]:
        if scout_vendor:
            spec = next((s for s in specs if s.vendor == scout_vendor), specs[i % len(specs)])
        else:
            spec = specs[i % len(specs)]

        def _thunk():
            session = executor.spawn(source_repo, spec.vendor, spec.model, spec=getattr(spec, "extra", {}))
            res = engine.agent(
                ScopedTask(prompt=build_scout_prompt(base_repo_map, scout_extra, i),
                           schema=SCOUT_SCHEMA, sandbox="read-only", model=spec.model,
                           vendor=spec.vendor,
                           scoped_inputs={"scout": i, "repo_snapshot_sha": base_commit}),
                lambda t: session.run(t), node_id=f"scout{i}",
                cli_version=getattr(session, "cli_version", ""), agent_type="scout",
            )
            return res.structured_output if (res.ok and isinstance(res.structured_output, dict)) else None

        return _thunk

    raw = engine.parallel([make_scout(i) for i in range(max(1, n_scouts))])
    assessments = [a for a in raw if isinstance(a, dict) and a.get("difficulty")]
    if not assessments:
        return {"difficulty": base_repo_map.get("difficulty", "medium"), "approach": "",
                "key_files": [], "risks": "", "n_scouts": 0, "source": "static_fallback"}
    vals = sorted(_DIFFICULTY_ORDER.get(str(a["difficulty"]).lower(), 1) for a in assessments)
    difficulty = _DIFFICULTY_INV[vals[len(vals) // 2]]  # median; upper-median on ties (conservative)
    approach = "\n".join(f"- {str(a.get('approach', '')).strip()}"
                         for a in assessments if a.get("approach"))[:4000]
    # NOTE: we deliberately do NOT strip "fetch the upstream package" guidance from the scout
    # plan. Per the sandbox-not-prompt policy we never limit the model via prompts; the worktree
    # shadows site-packages so a fetched package can't produce a false solve regardless of plan.
    key_files = sorted({f for a in assessments for f in (a.get("key_files") or [])
                        if isinstance(f, str)})[:40]
    risks = "; ".join(str(a.get("risks", "")) for a in assessments if a.get("risks"))[:1500]
    return {"difficulty": difficulty, "approach": approach, "key_files": key_files,
            "risks": risks, "n_scouts": len(assessments), "source": "agent_scout"}


def _run_phase_codegen(engine: Engine, ctx, repo_map: dict, phase: dict, carry: str):
    """Per-phase GENERATED-ORCHESTRATION seam (the user's "generate orchestration code per phase";
    flag-gated APEX_OMEGA_PHASE_CODEGEN=1, DEFAULT OFF). Author a phase-scoped ``orchestrate(ctx)``
    via the EXISTING author/lint/freeze machinery, then run it depth-1 via ctx.workflow. Any
    failure -> None so the caller falls back to ctx.run_phase (scoped converge). Acceptance stays
    engine-owned; a lint/compile reject is caught."""
    try:
        sub_map = dict(repo_map)
        sub_map["phase"] = {k: phase.get(k) for k in
                            ("name", "objective", "acceptance_gold_ids", "files_owned", "modules")}
        sub_map["approach"] = (
            "FOCUS ONLY ON THIS PHASE: " + str(phase.get("objective") or "") + "\n"
            "Make EXACTLY these gold tests pass (the phase acceptance): "
            + ", ".join(map(str, (phase.get("acceptance_gold_ids") or [])[:40])))
        source = extract_code(_author_via_llm(ctx._executor, ctx.worker_specs, sub_map, None) or "")
        lint = lint_source(source)
        if not source or not lint.ok:
            engine.log("phase codegen lint-fail; falling back to scoped converge")
            return None
        sha = sha256_hex(source)
        odir = Path(engine.run_dir) / "orchestrator"
        odir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(ch for ch in str(phase.get("name") or "p") if ch.isalnum() or ch in "-_")[:24]
        path = odir / f"phase_{safe_name}_{sha[:12]}.py"
        path.write_text(source, encoding="utf-8")
        engine.log(f"phase codegen: authored + froze {path.name}; running depth-1")
        ctx.workflow({"scriptPath": str(path)})
        w = ctx.select(ctx.all_candidates())
        if w is not None and getattr(w, "accepted", False):
            return {"accepted_full": True, "candidate": w, "merged_diff": ctx.carry_best() or carry,
                    "residual": [], "phase_passed": True, "phase_pass_count": 0, "phase_total": 0,
                    "conflicts": []}
        red = ctx.reduce_residuals([], carry_diff=(ctx.carry_best() or carry),
                                   scope_ids=phase.get("acceptance_gold_ids"))
        return {"merged_diff": red.get("merged_diff") or carry,
                "residual": list(red.get("residual_failing_ids") or []),
                "phase_passed": bool(red.get("phase_passed")),
                "phase_pass_count": int(red.get("phase_pass_count", 0) or 0),
                "phase_total": int(red.get("phase_total", 0) or 0),
                "accepted_full": bool(red.get("accepted")), "candidate": red.get("candidate"),
                "conflicts": list(red.get("conflicts") or [])}
    except Exception as exc:
        engine.log(f"phase codegen raised ({type(exc).__name__}: {exc}); falling back to scoped converge")
        return None


def phase_planned_solve(engine: Engine, ctx, repo_map: dict):
    """Claude-Code-style PHASED solve (the HYBRID core). Plan ordered phases (objectives + per-phase
    acceptance ids), solve each with the PROVEN converge inner loop scoped to its gold subset
    (ctx.run_phase), bank partial frontier gains the instant they appear (survives an outer kill),
    and GATE progression with an adversarial goal-alignment review so a long run never veers off the
    goal. Acceptance stays engine-owned (ctx.select). Returns a verified Candidate or None — None
    (degenerate plan / clean abstain) means the caller falls through to the whole-repo converge body
    and then the best-of-N floor. Respects the converge skip-gate (easy / <2 modules / <2 phases) so
    the easy-repo over-spawn pathology (C3) never bites."""
    difficulty = str((repo_map or {}).get("difficulty") or "").lower()
    if difficulty == "easy":
        return None                                       # easy stays on the cheap path (C3)
    plan = ctx.decompose()
    modules = (plan or {}).get("modules") or []
    if not plan or len(modules) <= 1:
        return None                                       # undecomposable -> whole-repo converge
    max_phases = {"medium": 3, "hard": 4}.get(difficulty, 3)
    n_gate = {"medium": 1, "hard": 3}.get(difficulty, 1)
    phases = ctx.plan_phases(plan=plan, max_phases=max_phases)
    if not phases or len(phases) <= 1:
        return None                                       # degenerate plan -> whole-repo converge
    engine.log("phase plan: " + str(len(phases)) + " phases [" +
               ", ".join(str(p.get("name")) for p in phases) + "]")
    codegen = os.environ.get("APEX_OMEGA_PHASE_CODEGEN") == "1"
    carry = ctx.carry_best()
    for pidx, ph in enumerate(phases):
        if not ctx.should_continue_waves():
            break
        # PRE goal-alignment gate (grounded in the live residual): proceed / revise / abort.
        g = ctx.goal_align_gate(plan, ph, residual_ids=ctx.last_residual(), stage="pre", n=n_gate)
        if g.get("verdict") == "abort":
            ctx.defer("plan_abort", ph.get("name"), g.get("reason") or "")
            engine.log("goal-gate ABORT (pre) phase=" + str(ph.get("name")) + ": " + str(g.get("reason")))
            break
        if g.get("verdict") == "revise" and g.get("retarget_gold_ids"):
            ph = {**ph, "acceptance_gold_ids": list(g["retarget_gold_ids"])}
            engine.log("goal-gate REVISE phase=" + str(ph.get("name")) + ": re-scoped acceptance ids")
        red = None
        if codegen and ph.get("needs_custom_orchestration"):
            red = _run_phase_codegen(engine, ctx, repo_map, ph, carry)
        if red is None:
            red = ctx.run_phase(ph, carry_diff=carry, phase_index=pidx)
        if red.get("accepted_full"):
            engine.log("SOLVED (full suite) during phase " + str(ph.get("name")))
            return red.get("candidate")
        if red.get("phase_passed") and red.get("candidate") is not None:
            ctx._checkpoint_phase(red["candidate"], subset_passed=red.get("phase_pass_count", 0),
                                  subset_total=red.get("phase_total", 0),
                                  phase_id=str(ph.get("name") or ""))
            engine.log("phase " + str(ph.get("name")) + " DONE (" +
                       str(red.get("phase_pass_count")) + "/" + str(red.get("phase_total")) + " gold ids)")
        carry = red.get("merged_diff") or carry
        # POST goal-alignment gate: did the phase output drift off the goal?
        g2 = ctx.goal_align_gate(plan, ph, residual_ids=red.get("residual"), stage="post", n=n_gate)
        if g2.get("verdict") == "abort":
            ctx.defer("plan_abort", ph.get("name"), g2.get("reason") or "")
            engine.log("goal-gate ABORT (post) phase=" + str(ph.get("name")) + ": " + str(g2.get("reason")))
            break
    return ctx.select(ctx.all_candidates())               # engine-owned; may be None (abstain)


def autosolve(
    engine: Engine, *,
    source_repo: str,
    executor: Any,
    worker_specs: Sequence[Any],
    score_fn: Callable[[str], Any],
    prompt_builder: Callable[[OrchestrationContext, int, str], str],
    base_commit: Optional[str] = None,
    repo_map: Optional[dict] = None,
    abl: Optional[AblationConfig] = None,
    author: bool = True,
    author_fn: Optional[Callable[[dict], str]] = None,
    author_vendor: Optional[str] = None,
    max_agents: Optional[int] = None,
    run_scope: str = "autosolve",
    scout_extra: Optional[dict] = None,
    scout_agents: int = 0,
    scout_vendor: Optional[str] = None,
    agent_ceiling: int = 1000,
    timeout_seconds: Optional[int] = None,
    repair_iters: int = 0,
    expected_ids_sha: str = "",
    scoring_env_sha: str = "",
    args: Any = None,
) -> dict:
    """Scout -> author -> freeze -> sandboxed execute, with fail-open to the
    verified best-of-N floor.  When ``scout_agents > 0`` an agent fan-out decides
    the completion approach + difficulty, and difficulty sets the INITIAL agent
    count + the soft cap.  Returns a result dict with the winner (or abstain) and
    the frozen-orchestration metadata."""
    abl = abl or AblationConfig()
    if repo_map is None:
        engine.phase("scout")
        repo_map = build_repo_map(source_repo, base_commit=base_commit, extra=scout_extra)
        engine.log(f"scouted repo: {repo_map.get('n_source_files')} src / "
                   f"{repo_map.get('n_test_files')} test files, difficulty={repo_map.get('difficulty')}")

    # BUDGET-AWARE SCOUTING: scouts count against the SAME K-agent pool as solve attempts (the
    # engine charges every dispatch). At small K a 3-scout fan-out cannibalizes solve shots — the
    # verified root cause of autogen < template on jinja at K=8 (3 scouts + author left only 4 solve
    # attempts vs the template's 8, and the template's win lived at depth 7). Cap scouts to a small
    # fraction of the budget so the MAJORITY of K stays available for actual solve attempts.
    _budget = int(max_agents if max_agents is not None else agent_ceiling)
    if scout_agents > 0:
        scout_agents = max(1, min(int(scout_agents), _budget // 6))
    if scout_agents > 0:
        engine.phase("scout-fanout")
        scout = agent_scout(engine, executor=executor, worker_specs=worker_specs,
                            source_repo=source_repo, base_commit=base_commit,
                            base_repo_map=repo_map, n_scouts=scout_agents,
                            scout_vendor=scout_vendor, scout_extra=scout_extra)
        repo_map["scout"] = scout
        # ANTI-INFLATION: the scout INFORMS approach + key_files, but it must NOT escalate the budget/
        # strategy difficulty ABOVE the static proxy. Scouts rated voluptuous (static=easy) "hard"
        # (wasting agents) and jinja (static=medium) "hard" (steering the architect toward heavy
        # decompose/repair instead of the wide best-of-N that actually solved it). Take the LOWER of
        # the static proxy and the scout read so the scout can refine DOWN but never inflate UP.
        _so = _DIFFICULTY_ORDER.get(str(repo_map.get("difficulty") or "").lower(), 1)
        _sc = _DIFFICULTY_ORDER.get(str(scout.get("difficulty") or "").lower(), 1)
        repo_map["difficulty"] = _DIFFICULTY_INV[min(_so, _sc)]
        if scout.get("approach"):
            repo_map["approach"] = scout["approach"]
        if scout.get("key_files"):
            repo_map["key_files"] = scout["key_files"]
        engine.log(f"scout fan-out (n={scout['n_scouts']}, src={scout['source']}): "
                   f"scout-difficulty={scout['difficulty']} -> used={repo_map['difficulty']} "
                   f"(scouts capped to {scout_agents} of budget {_budget})")

    # difficulty -> INITIAL agents + soft cap (caller max_agents overrides the cap).
    init_agents, soft_cap = difficulty_profile(repo_map.get("difficulty"), ceiling=agent_ceiling)
    effective_max = min(max_agents if max_agents is not None else soft_cap, agent_ceiling)
    init_agents = min(init_agents, effective_max)
    engine.log(f"agent budget: initial={init_agents} soft_cap={effective_max} "
               f"(difficulty={repo_map.get('difficulty')}, ceiling={agent_ceiling})")

    engine.phase("author")
    frozen = author_orchestration(engine, executor=executor, worker_specs=worker_specs,
                                  repo_map=repo_map, author=author, author_fn=author_fn,
                                  author_vendor=author_vendor)

    ctx = OrchestrationContext(
        engine, executor=executor, worker_specs=worker_specs, source_repo=source_repo,
        base_commit=base_commit, score_fn=score_fn, prompt_builder=prompt_builder,
        repo_map=repo_map, abl=abl, run_scope=run_scope, max_agents=effective_max,
        initial_agents=init_agents, timeout_seconds=timeout_seconds, repair_iters=repair_iters,
        args=args,
    )
    # review-fix #13: make the journaled-score drift keys content-bearing (they default ""
    # and were never set, so the score key relied on diff_sha alone). Now a legitimate
    # scoring-env change (expected-id inventory / venv / eval cap) invalidates a stale score.
    ctx.expected_ids_sha = expected_ids_sha
    ctx.scoring_env_sha = scoring_env_sha

    winner = None
    error = None
    cut_losses = None

    # Backbone 0.5: host-side template FLOOR-PROBE. Always BANK a verified floor
    # candidate first (resilience; journaled best-of-N wave-0). The RESCUE (using it as
    # the winner when the authored plan abstains) is OPT-IN, DEFAULT OFF, so the autogen
    # arm "stands alone" for honest measurement; enable for production completion-first
    # via the `floor_rescue` ablation or APEX_OMEGA_FLOOR_RESCUE=1. checkpoint=rescue_enabled
    # keeps the floor's solve from being reported as the cell's solve in honest mode.
    rescue_enabled = bool(getattr(abl, "floor_rescue", False)) or os.environ.get("APEX_OMEGA_FLOOR_RESCUE") == "1"
    # review F2: the RALPH baseline is a VANILLA control — it must NOT run the template best-of-N
    # floor probe (a template shot it never uses would both contaminate the "vanilla" result and
    # inflate ralph's reported agents_used/cost, breaking the apples-to-apples cost comparison).
    _ralph_mode = os.environ.get("APEX_OMEGA_ORCHESTRATION") == "ralph"
    floor_cand = None
    if not _ralph_mode:
        try:
            engine.phase("floor-probe")
            floor_cand = ctx.solve_attempt(attempt_id=0, strategy="minimal", checkpoint=rescue_enabled)
        except Exception as exc:
            engine.log(f"floor-probe failed: {type(exc).__name__}: {exc}")

    def _floor():
        # The floor runs the CHEAP verified best-of-N template DIRECTLY — it must NOT go
        # through author_orchestration (which would resume the just-frozen, failing
        # authored script and re-crash), and it must NOT be the convergence default (which
        # may decompose/fan-out — over-spawn risk on a fall-open). The cheap escalating
        # best-of-N + repair path is the guaranteed completion-first floor.
        fw = FrozenWorkflow(BEST_OF_N_ORCHESTRATION, sha256_hex(BEST_OF_N_ORCHESTRATION),
                            "fallback", True, [])
        return run_orchestration(BEST_OF_N_ORCHESTRATION, ctx), fw

    # HYBRID: the host-side Claude-Code-style phase planner runs FIRST (depth 0), then falls through
    # to the frozen converge body when it abstains/degenerates. Gated by the selector + flag so the
    # converge/template/baseline arms are byte-for-byte unchanged.
    if (os.environ.get("APEX_OMEGA_PHASE_PLANNER") == "1"
            and os.environ.get("APEX_OMEGA_ORCHESTRATION") == "hybrid"):
        try:
            engine.phase("phase-plan")
            pw = phase_planned_solve(engine, ctx, repo_map)
            if pw is not None and getattr(pw, "accepted", False):
                winner = pw
        except PlateauStop as exc:
            engine.log(f"phase planner plateau-stop: {exc}; selecting best banked candidate")
            winner = ctx.select([c for c in ctx.all_candidates() if c is not floor_cand])
        except Exception as exc:
            engine.log(f"phase planner raised ({type(exc).__name__}: {exc}); falling through to converge")

    if winner is not None and getattr(winner, "accepted", False):
        pass   # the phase planner already produced a verified winner; skip the converge body
    else:
        try:
            winner = run_orchestration(frozen.source, ctx)
        except PlateauStop as exc:
            # Backbone 2.1: a CLEAN governor stop (plateau/ceiling), not a defect. Select the
            # best banked AUTHORED candidate (exclude the template floor -> autogen stands alone;
            # the gated rescue below still applies).
            engine.log(f"plateau-stop: {exc}; selecting best banked authored candidate")
            winner = ctx.select([c for c in ctx.all_candidates() if c is not floor_cand])
        except FailLoud as exc:
            # lint/compile guard tripped -> fall open to the verified best-of-N floor
            engine.log(f"frozen orchestration rejected ({exc}); failing open to best-of-N floor")
            error = str(exc)
            winner, frozen = _floor()
        except Exception as exc:  # generated strategy crashed -> floor
            engine.log(f"orchestration raised ({type(exc).__name__}: {exc}); failing open to floor")
            error = f"{type(exc).__name__}: {exc}"
            try:
                winner, frozen = _floor()
            except Exception as exc2:
                error = f"{error}; floor also failed: {exc2}"

    # NOTE: there is deliberately NO "no-winner -> fall open to template" rescue here.
    # If the AUTHORED orchestration runs cleanly but produces no accepted winner (it
    # abstained), that is autogen's REAL result and is reported as a failure — so the
    # autogen-vs-template/baseline comparison stays honest (autogen must stand on its
    # own, not inherit the template's solves). The only fall-opens above are for a
    # MALFORMED generated orchestrator (lint/compile reject, or runtime crash), which
    # is an authoring defect rather than a strategy outcome.

    # review-fix #8: never report worse than already-banked verified work. If no accepted
    # winner emerged (a clean abstain, or a post-accept crash that fell open to the floor)
    # but a verified-accepted candidate was banked into ctx, select it — excluding the wave-0
    # floor probe so the honest "autogen stands alone" measurement is preserved. Mirrors the
    # PlateauStop path and honors the acceptance checkpoint's intent (a banked accept is never
    # silently discarded by a normal-completion exit).
    if winner is None or not getattr(winner, "accepted", False):
        banked = ctx.select([c for c in ctx.all_candidates() if c is not floor_cand])
        if banked is not None and getattr(banked, "accepted", False):
            engine.log("reconciled winner from banked verified candidate (review-fix #8)")
            winner = banked

    # Backbone 0.5 (gated): rescue an abstained/unaccepted authored plan with the verified
    # floor ONLY when rescue is enabled (opt-in). Default OFF -> autogen stands alone.
    floor_rescued = False
    if (rescue_enabled and (winner is None or not getattr(winner, "accepted", False))
            and floor_cand is not None and getattr(floor_cand, "accepted", False)):
        engine.log("authored orchestration did not accept; rescued by the verified template floor")
        winner = floor_cand
        floor_rescued = True

    # CUT-LOSSES record: if the governor halted the run (a genuine non-progress cut, or an
    # honest ceiling/explored stop), surface WHY + the best distance-to-solve reached, so the
    # reclassifier/ledger books it as a diagnosable non-progress FAILURE — never as
    # infra/timeout. Populated whether the stop propagated (omega) or was caught inside the
    # workflow (ralph): ctx carries the halt reason either way.
    if getattr(ctx, "_halt_reason", "") and not bool(winner is not None and getattr(winner, "accepted", False)):
        # SPFG+ outcome taxonomy: map the halt reason to a distinct outcome so the
        # reclassifier/ledger books a genuine no-progress plateau (cut:no-progress) separately
        # from a harness/scorer wall (cut:harness-stall, excluded from solve-rate denominators)
        # and from an honest explored/ceiling stop. seconds_since_frontier_improved /
        # valid_measurements / indeterminate_total / frontier_history surface WHY.
        _reason = str(ctx._halt_reason or "")
        if _reason == "cut:no-progress":
            _outcome = "cut:no-progress"
        elif _reason == "cut:harness-stall":
            _outcome = "cut:harness-stall"
        elif _reason.startswith("cut:"):
            _outcome = _reason
        else:
            _outcome = "stopped-ceiling"
        cut_losses = {
            "reason": ctx._halt_reason,
            "is_cut": bool(getattr(ctx, "_halt_is_cut", False)),
            "outcome": _outcome,
            "best_gold_passed": int(getattr(ctx, "_best_gold_passed", 0)),
            "best_pass_rate": float(getattr(ctx, "_best_pass_rate", 0.0)),
            "agents_used": engine.agents_used(),
            "valid_measurements": int(getattr(ctx, "_valid_measurements", 0)),
            "seconds_since_frontier_improved": float(
                (getattr(ctx, "_valid_wall_accum", 0.0) - getattr(ctx, "_valid_wall_at_best", 0.0))
                if (getattr(ctx, "_wall_started", False)
                    and getattr(ctx, "_valid_wall_at_best", None) is not None) else 0.0),
            "indeterminate_total": int(getattr(ctx, "_indeterminate_total", 0)),
            "frontier_history": list(getattr(ctx, "_frontier_history", []) or []),
        }

    return {
        "floor_rescued": floor_rescued,
        "solved": bool(winner is not None and getattr(winner, "accepted", False)),
        "abstained": winner is None,
        "cut_losses": cut_losses,
        "agents_used": engine.agents_used(),
        "budget": engine.budget.to_dict(),
        "orchestration": frozen.to_dict(),
        "difficulty": repo_map.get("difficulty"),
        "scout": repo_map.get("scout"),
        "agent_budget": {"initial": init_agents, "soft_cap": effective_max, "ceiling": agent_ceiling},
        "error": error,
        "winner": (None if winner is None else {
            "candidate_id": winner.candidate_id, "accepted": winner.accepted,
            "score": winner.combined_score, "vendor": winner.meta.get("vendor"),
            "strategy": winner.meta.get("strategy"),
        }),
    }
