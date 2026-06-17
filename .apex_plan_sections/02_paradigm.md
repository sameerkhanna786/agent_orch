## 2. The Orchestration Substrate: Dynamic Workflows as APEX’s Foundation

APEX-Ω is built **on top of** the vendor-agnostic dynamic-workflow paradigm: a deterministic *orchestration-as-code* engine in which a program — not a conversation — holds the plan, fans scoped work out to isolated coding **workers** (Codex, Claude Code, or any agent CLI/API, even mixed in one run), keeps every intermediate result in **script variables and a durable journal rather than a chat window**, and converges via execution-grounded **verify-and-refute**. This section specifies that substrate: the orchestrator–worker model, the five primitives with exact semantics, the determinism/journaling discipline that makes restart-survivable resume possible, and the precise places where APEX v1 has already converged on the paradigm versus where it falls short and must be extended.

One framing claim, stated up front and load-bearing, governs the whole section. The substrate is **necessary plumbing for exceeding base-model capability, but it is not the mechanism.** The capability unlock is execution-grounded verify-and-refute plus evidence-authoritative selection (Section 13); the scaling unlock is context isolation. The orchestration engine is what lets those two properties be *expressed, scaled, and resumed* cleanly. The adversarial review of this exact claim returned **sound-with-caveats**: attributing capability to "a deterministic dynamic-workflow engine" rather than to "execution-grounded verification running on that substrate" is a category error that, if it drives prioritization, produces the classic failure mode — investing in fan-out and agent count instead of judge quality ([Du et al., ICML 2024](https://arxiv.org/abs/2305.14325); [Limits of Inference Scaling Through Resampling, arXiv:2411.17501](https://arxiv.org/abs/2411.17501); [CodeMonkeys, arXiv:2501.14723](https://arxiv.org/abs/2501.14723)). We therefore design the substrate to make verification *cheap to scale*, and we keep the Cardinal Safety Contract (Section 13) as an engine-level invariant, not an afterthought.

### 2.1 The orchestrator–worker model

The mental model is **orchestrator–worker**. A deterministic orchestration program holds all run state in local variables; **stateless-per-call workers** receive a scoped job and return validated structured data plus an observable filesystem diff. This is the same shape the Claude Code dynamic-workflow tool ships, but that tool is **one implementation, not the engine** — APEX owns a vendor-neutral engine in which Codex/Claude/Gemini/opencode/API are merely leaf workers (Section 3).

Two properties of this model do the real work:

1. **State lives in variables and a journal, never in a conversation window.** The orchestrator’s context holds only what it must to make the next decision; a 500-node run does not drift because no single context accumulates 500 nodes of history. This is independently corroborated, not vendor marketing: [Chroma’s context-rot study](https://www.trychroma.com/research/context-rot) found all 18 frontier models tested (including Opus-class) degrade as input grows *well before the window fills*, and [Liu et al., "Lost in the Middle"](https://arxiv.org/abs/2307.03172) measured a 15–20-point U-shaped accuracy drop driven by position alone. Keeping intermediate results in variables is an architectural fix for a measured failure, which is why context isolation is the **scaling** unlock.

2. **Workers are opaque and scoped.** APEX does **not** drive a worker’s internal tool loop. It spawns a worker with a scoped prompt and a restricted tool set, observes it (stdout stream + a progress watchdog), and reads back a structured result and the resulting git diff. The only mid-flight steering channel is stream observation (v1’s `CLITurnParser` `turn_observer`), and even that can only *course-correct or abort*, never rewrite a running subprocess prompt.

A critical caveat that the SOTA evidence forces into this section: **coding is the harder case for fan-out.** Anthropic’s own multi-agent research write-up reports a +90.2% gain over single-agent on a *research* eval, with token usage explaining ~80% of variance — but the same source warns "most coding tasks involve fewer truly parallelizable tasks than research" and agents "are not yet great at coordinating and delegating in real time" ([Anthropic, multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)). Those headline numbers are cross-domain and vendor-reported; they are **not** load-bearing justification for APEX and must not be cited as such. The defensible position is narrower: the substrate wins on *decomposable, verification-heavy, isolatable* coding work (large migrations, codebase-wide audits, repository-scale issue resolution with worktree isolation) and is potentially net-negative (≈15× token cost) on small, tightly-coupled changes — which is why APEX **auto-routes** rather than auto-orchestrates (Section 8).

### 2.2 The five primitives (exact semantics)

APEX re-implements five vendor-neutral primitives. v1 already implements the first two and analogs of three more; the engine consolidates them so the *orchestration program*, not bespoke Python in `solver.py`, composes them.

| Primitive | Semantics | v1 analog | Disposition |
|---|---|---|---|
| `agent(prompt, opts)` | Spawn one isolated worker; with a JSON schema, returns a **validated** structured object (validation at the tool layer, model retries on mismatch); else final text. `isolation:"worktree"` gives the worker its own git worktree. | `CLIModelClient.run_structured_prompt` (`cli_backend.py`) | **adopt** (lift to Executor interface, Section 3) |
| `parallel(thunks)` | **Barrier** fan-out: run concurrently, await ALL; a failed thunk resolves to `null` (**caller must filter**). Use only when all results are needed together (dedup/merge/early-exit). | `RolloutEngine.execute_rollout_requests` (`rollout/engine.py`) | **adopt** |
| `pipeline(items, ...stages)` | Per-item **staged streaming, no inter-stage barrier**: item A can be in stage 3 while item B is in stage 1. Wall-clock = slowest single chain, not sum-of-slowest-per-stage. The **default** for multi-stage work. | **absent** (v1 has barrier waves only) | **adopt** (the one genuinely net-new primitive) |
| `phase(title)` / `log(msg)` | Progress grouping + narration; emits durable artifacts so narration *is* the journal. | per-phase `atomic_write_json` + `controller_decisions.jsonl` | **adopt** (formalize) |
| `budget {total, spent(), remaining()}` | Shared token/cost ceiling; supports loop-until-budget. **Opt-in; defaults unbounded.** | `repo_token_cap`/`max_tokens_per_repo_followup` (default OFF) | **adopt** (with v1 invariant preserved, §2.6) |

#### 2.2.1 `agent()` — single isolated worker

`agent()` is the atom. Conceptually:

```text
agent(prompt: str, opts: {
  schema?:    JSONSchema,        # if present, return validated object; else final text
  model?:     str,              # canonical alias (e.g. "opus"); resolved at command-build time
  vendor?:    Vendor,          # claude_cli | codex_cli | gemini_cli | opencode_cli | openai_api | ...
  label?:     str,             # human-readable, for phase()/log() + journal keys
  phase?:     str,
  isolation?: "none" | "worktree" | "snapshot" | "synthetic",  # FS isolation tier
  agentType?: str,             # role: reproducer | localizer | patcher | test_writer | reviewer | ...
  allowedTools?: [str],        # restricted tool set (scope-down for safety + cost)
}) -> AgentResult
```

```text
AgentResult = {
  ok:        bool,             # transport/finalization success, NOT correctness
  text?:     str,
  parsed?:   object,           # schema-validated structured payload, if schema given
  fs_diff:   UnifiedDiff,      # the authoritative artifact (git diff in the worker's worktree)
  usage:     {input_tokens:int, output_tokens:int, cache_read_tokens:int, cache_write_tokens:int},
  finalization_status: enum,   # completed | timeout | policy_violation | output_limit
                               # | progress_abort | isolation_error | infra_nonresult
  vendor:    Vendor,
  resolved_model: str,         # pinned launcher id, recorded in RunManifest
}
```

Three semantics are non-negotiable and inherited verbatim from v1:

- **`agent()` never raises to the caller.** Every abnormal exit becomes a typed result with a `finalization_status`. v1’s `run_structured_prompt` already guarantees this; the engine preserves it so a single worker crash can never crash the orchestration program.
- **Schema validation happens at the tool layer with model retries.** Native where the vendor supports it (Claude `--json-schema`), degraded to schema-as-prompt-text with post-parse where it does not (gemini/codex; codex additionally normalizes `additionalProperties=false` + `required=all keys`). This is **graceful degradation, not uniform guarantee** — native > prompt-text fidelity, and the engine records which path was used. v1 retries up to `max_attempts=4` on infra non-results for claude/codex.
- **`ok` is transport success, never correctness.** Correctness is decided downstream by executing against the filesystem (Section 13). The `fs_diff` field — the git diff the worker produced regardless of vendor — is the **authoritative artifact** and the single property that makes heterogeneous fleets possible (Section 3).

#### 2.2.2 `parallel()` — barrier fan-out

`parallel(thunks)` runs thunks concurrently and awaits **all** of them; a failed thunk resolves to `null`. **The caller must filter `null` before use** — this is the contract, and it must be paired with fail-loud accounting so a silently-null result never masquerades as "no problem found." v1’s `execute_rollout_requests` is exactly this: a single-threaded scheduler over an abandonable thread pool, each thunk a rollout in its own worktree under an `fcntl` lock, failed rollouts classified into failed `RolloutResult`s. `stop_on_result` enables early-exit/preempt/drain.

Use `parallel()` **only when you genuinely need all results together** (judge panel, candidate dedup, merge, early-exit on first verified pass). When stages differ in duration, `parallel()` wastes wall-clock at the barrier — which is precisely why `pipeline()` exists.

#### 2.2.3 `pipeline()` — per-item staged streaming (the net-new primitive)

`pipeline(items, stage1, stage2, ...)` streams each item through stages with **no barrier between stages**. This is the **default for multi-stage work** and the one primitive with **no v1 analog** — v1 runs only barrier waves, and inside each rollout the stages `reproduce → localize → patch → test` run sequentially. The natural APEX mapping streams items (files, sub-tasks, or rollouts) through that cascade so a fast localizer result begins patching while a slow reproducer is still running on another item:

```text
pipeline(work_items,
  stage("reproduce",  reproducer_agent),
  stage("localize",   localizer_agent),
  stage("patch",      patcher_agent),
  stage("verify",     verify_on_diff))     # execution-grounded; see Section 13
```

```text
# Scheduler invariant (no inter-stage barrier):
for each item in items:
    place item at stage[0] in ready_queue
loop until all items terminal:
    dispatch up to concurrency_cap ready (item, stage) units      # one journaled agent() call each
    on a (item, stage) completion:
        if result is terminal-fail and not recoverable: mark item failed (fail-loud, do not fake)
        elif stage is last: mark item complete
        else: advance item to next stage, re-enqueue
    # KEY: an item in stage k+1 does NOT wait for siblings still in stage k
```

The win is concrete and measurable: **wall-clock = slowest single chain, not sum-of-slowest-per-stage.** The shape is proven by Inngest’s memoized `step.run`/`step.invoke` ([Inngest durable steps](https://www.inngest.com/blog/ai-agents-inngest-durable-steps)). The cost is more complex scheduling and resume bookkeeping: the journal must key cache entries per **`(item, stage)`** (§2.5), and inter-stage data contracts must be explicit typed artifacts (v1’s `ReproductionArtifact → LocalizationArtifact → PatchArtifact → TestSuiteArtifact` with `to_dict`/`from_dict` are the reusable substrate). This is net-new code and must be validated against the determinism/journaling invariants before it is trusted.

#### 2.2.4 `phase()` / `log()` — narration that is the journal

`phase(title)` groups progress; `log(msg)` narrates. In APEX these are **not** decorative: `phase()` boundaries coincide with cache/journal checkpoints, and both emit the durable artifacts and transition records v1 already writes (`repo_context.json`, `baseline_result.json`, `apex_result.json` via `atomic_write_json`; `controller_decisions.jsonl`, one JSON line per decision). The discipline is **fail-open**: narration is side-effect-free and may never block or crash a run. Coupling `phase()` to durable artifact + transition emission means **the narration is the journal** — a tidy integration that keeps observability and durability consistent.

#### 2.2.5 `budget {}` — shared ceiling, opt-in, loop-until-budget

`budget {total, spent(), remaining()}` is a shared token/cost ceiling supporting loop-until-budget patterns. The machinery exists in v1 (`repo_token_cap`, `max_tokens_per_repo_followup`, `_cap_followup_rollouts_for_token_budget`, `BudgetPlanner`/`TurnBudget`) but defaults **OFF** per the "never optimize for cost" directive. APEX exposes it as a **first-class primitive, defaulted unbounded** so that cross-vendor cost arbitrage (heavy orchestrator on one vendor, cheap executors on another — Section 12) is available when an operator opts in. The invariant in §2.6 is mandatory: **budget exhaustion must never abort an in-flight succeeding rollout.**

### 2.3 The two unlocks, kept in their lanes

The substrate exists to serve two properties, and the section must keep them attributed correctly:

- **Context isolation = the scaling unlock.** State in variables + per-worker scoped context + per-rollout worktrees. v1’s analogs: `RepoContext` scanned once and read-only, the relevance-ranked `EpisodicMemoryBus` blackboard (which *excludes the caller’s own rollout_id* and shares **negative/ruled-out** discoveries so siblings avoid dead ends), and per-worker air-gapped `HOME`. The engine generalizes this into a uniform "orchestration holds state, workers receive only scoped context" substrate, with scoping budgets that are **capability-aware** because cross-vendor context windows differ. Over-scoping reintroduces bloat; under-scoping starves a worker — so scope is a tuned, not fixed, parameter.

- **Verify-and-refute = the capability unlock — and it is the verifier, not the agent count, that pays.** The mechanism is: independent attempts are produced, *other agents try to refute them*, iteration continues until convergence — "the model stops saying done when it is half done." The quality (not throughput) gain has independent academic backing ([Du et al.](https://arxiv.org/abs/2305.14325); Tool-MAD evidence-grounded debate; A-HMAD reliability-weighted consensus). v1 already implements **three** forms — family-disjoint independent-CLI tool-call review, the self-play tournament (K patches × M independently-generated tests), and the `VerificationAmplifier` (discriminating tests applied only at `confidence ≥ 0.6`). The literature’s clearest weakness is the **judge/aggregator** ([DebateCV: LLMs struggle as moderators]); the directive that follows is decisive: **invest in verifier/judge quality, default-to-refute, weight verifiers by reliability, and ground every claim in executed evidence — do not buy capability by spawning more skeptics.** The scaling theory makes this concrete: against imperfect verifiers the compute-optimal sample count is finite and often **< 10**, and CodeMonkeys shows 69.8% coverage collapsing to 57.4% after selection on SWE-bench Verified ([arXiv:2411.17501](https://arxiv.org/abs/2411.17501); [arXiv:2501.14723](https://arxiv.org/abs/2501.14723)). The substrate buys coverage; the verifier buys realized capability. Full mechanics live in Section 13.

### 2.4 Where v1 has already converged — and where it falls short

v1 independently converged on the paradigm’s mental model but encoded it as a **bespoke single-purpose pipeline** rather than a general engine. The honest accounting:

| Paradigm element | v1 status | Gap / action |
|---|---|---|
| `agent()` | ✅ `run_structured_prompt` (spawn external CLI, observe stdout + S1–S7 watchdog, typed `CLIModelResult`, never raises) | Lift to a normalized vendor-neutral **Executor** (Section 3) |
| `parallel()` | ✅ `execute_rollout_requests` (barrier fan-out, worktree+lock isolation, blackboard) | Generalize; keep overlap-diversity capacity cap |
| `pipeline()` | ❌ **absent** — barrier waves only | **Net-new build** — the largest new primitive |
| `phase()`/`log()` | ◐ atomic artifacts + `controller_decisions.jsonl` exist, but not as a called API | Formalize; couple to journal checkpoints |
| `budget {}` | ◐ machinery exists, defaulted OFF | First-class, opt-in, unbounded default, §2.6 invariant |
| verify-and-refute | ✅ three scattered forms | Unify under one primitive (Section 13) |
| context isolation | ✅ `RepoContext`-once + blackboard + worktrees | Generalize into the engine |
| orchestration-as-code | ❌ **absent** — hard-coded in `solver.py` | Lift `_execute_with_dynamic_transitions` (escalation while-loop, `max_strategy_iterations=20`) + `_execute_progressive_rollout_plan` (wave loop, `max_progressive_rollout_waves=6`) into a re-implementable program |
| durable resume | ❌ **narrow** — session-scoped equivalent | **Promote** unused `ReplayRecorder` + narrow escrow WAL into a per-`agent()`-call journal (§2.5) |

Two gaps are the headline work. (1) **No orchestration-as-code layer:** the model never writes a script; the dependency graph, fan-out, and loops are hard-coded as `_execute_with_dynamic_transitions` + `_execute_progressive_rollout_plan` (~8.2k-line `solver.py`). (2) **Resume is the explicit "do better than the reference impl" mandate:** the reference Claude Code engine resumes only *within a session* — "if you exit Claude Code while a workflow is running, the next session starts the workflow fresh" ([Claude Code workflows docs](https://code.claude.com/docs/en/workflows)) — and v1 is at the same limitation. v1’s `ReplayRecorder`/`ReplayPlayer` record/verify CLI+tool I/O but have **NO production callsite**, and the escrow WAL (CCEDF) is an fsync-durable backstop that rescues only **one** confirmed-full-scope-pass candidate across restart. Durable restart-survivable resume is therefore **genuinely unbuilt in v1**, not merely under-documented — the clearest "APEX must do better" target.

### 2.5 Durable execution: the deterministic-workflow / non-deterministic-activity split

APEX adopts the **durable-execution template** that the entire industry has converged on (Temporal event-sourced replay; DBOS Postgres checkpoints; Inngest memoized steps; AWS Step Functions Standard / Lambda Durable Functions). The model splits cleanly:

- The **orchestration program is the deterministic "workflow."** It may contain **no nondeterminism** — no wall-clock reads, no RNG, no unguarded I/O. Temporal’s constraint is blunt and correct: *non-determinism is fatal to replay* ([Temporal workflows](https://docs.temporal.io/workflows)). This is exactly why the Claude Code engine bans `Date.now`/`Math.random`. v1 already honors the *spirit*: `temperature=0.0`, deterministic 5-tuple failover ranking (`_candidate_failover_rank`), pure `assign_strategy`/`get_temperature` functions of `rollout_id`, bit-identical snapshot SHAs (fixed author/date), and atomic JSON writes. APEX has no JS runtime, so the rule is enforced at the engine API: the orchestration layer is given a deterministic clock and a deterministic seed source, both journaled.

- **`agent()`/tool/shell/LLM calls are non-deterministic "activities."** They run once, are journaled, and their results are **replayed** on resume. Because workers call external services, the real semantics are **at-least-once with idempotency keys**, not exactly-once. Every activity carries `idempotency_key = run_id + node_id + attempt` so a re-run after a crash does not double-apply an external side effect (a duplicate repo edit). This is the universal rule across Temporal activities, DBOS steps that call external services, and Step Functions Express ([learn.temporal.io](https://learn.temporal.io/tutorials/go/background-check/durable-execution/); [DBOS](https://docs.dbos.dev/why-dbos)).

**Journal design.** Each `agent()` call is keyed by a content hash and persisted to a WAL:

```text
journal_key = sha256(
    canonical_json({
        prompt, schema, model, vendor, agentType,
        scoped_inputs,        # the exact scoped context/files the worker saw
        repo_snapshot_sha,    # bit-identical snapshot SHA of the worker's input tree
        item_id, stage,       # per-(item,stage) for pipeline() nodes
    })
)
JournalEntry = {
    key:        str,                # journal_key
    run_id:     str,
    node_id:    str,                # stable position in the orchestration program
    attempt:    int,
    result:     AgentResult,        # the recorded structured output + fs_diff (replayed, not re-derived)
    status:     "committed" | "failed" | "in_flight",
    vendor_pin: {vendor, resolved_model, version},   # from RunManifest
    ts_logical: int,                # monotonic logical clock (NOT wall-clock)
}
```

```text
on agent(prompt, opts):
    key = journal_key(prompt, opts, scoped_inputs, repo_snapshot_sha, item_id, stage)
    entry = wal.lookup(key)
    if entry and entry.status == "committed":
        return entry.result            # UNCHANGED call -> replay cached result
    # edited/new call OR previously in_flight (crash): re-run
    wal.append({key, status:"in_flight", attempt})
    result = executor.spawn(opts, prompt)   # the only non-determinism, idempotency-keyed
    wal.commit({key, status: result.ok ? "committed":"failed", result})
    return result
```

The cache-validity semantic is the subtle part and must be designed deliberately: a "cached result" means **replaying the recorded output**, *not* re-deriving it. The input hash includes the `repo_snapshot_sha`, so if the underlying code changed, the hash changes and the call re-runs — preventing a stale answer from being replayed against changed code. This is the documented hazard the adversarial review flagged, and the snapshot-SHA term in the hash is the mitigation.

**Storage.** The default journal is **Postgres-as-WAL à la DBOS** — a library, not a cluster, the most self-hostable and vendor-neutral choice, with SQL observability over the checkpoint table. The known scaling trap is Postgres contention under high fan-out (lock contention on a single status row, WAL/autovacuum pressure); the mitigation is to **avoid hammering one row** — partition/shard the journal by `run_id` and batch commits. For deployments that cannot run Postgres, a local fsync-durable append-only WAL (generalizing v1’s CCEDF escrow) is the fallback.

**Cross-vendor replay** requires the **RunManifest** to be authoritative: it pins `apex_git_sha`, python/platform, `model_versions`, digest-pinned `docker_images`, and harness versions — directly satisfying the mandate’s "pin vendor+model+version for replay." Replay reproduces **artifacts** (diffs + re-run verification), **not** token streams: bit-reproducible agent *output* is impossible across hosted APIs (temperature-0 batch non-invariance), so it is explicitly **rejected** — we reproduce what the run *produced and verified*, which is what matters.

### 2.6 Conflicts with v1 invariants, and how the substrate respects them

Three places where the paradigm’s defaults would, taken naively, violate a load-bearing v1 invariant. Each is resolved here, not deferred.

1. **Model-authored control flow vs. determinism — freeze-then-journal.** The paradigm’s defining move is "the model writes the orchestration script." But v1 keeps the orchestration layer *pure* precisely so an infra/model artifact never masquerades as a result. The resolution: **either a deterministic planner emits the workflow, or a model-authored script is snapshotted, hashed into the RunManifest, and journaled as a deterministic activity, so replay runs over a FROZEN script.** Live model output must **never** be un-journaled control flow. This keeps Temporal-style replay soundness and the `apex replay-deterministic --verify` guarantee intact while still admitting model-authored plans.

2. **"Substrate exceeds the base model" vs. the Cardinal Safety Contract.** The engine must **not** gain the power to promote an unverified candidate. In the new engine, selection/acceptance primitives keep **soft signals downgrade-only** and every soft/LLM signal strictly below every execution signal (v1’s `_apply_evidence_bound_review` flips `accepted` only `True → False`). This is the rule that converts best-of-N into trustworthy gains and counters the false-positive inversion that reward-hacking makes worse as capability scales ([ImpossibleBench: GPT-5 exploits tests 76%, arXiv:2510.20270](https://arxiv.org/html/2510.20270v1)). It is an **engine-level invariant**, carried verbatim into Section 13.

3. **`budget{}` loop-until-budget vs. "a cap must never abort succeeding work."** v1 fires cumulative caps **only when no successful patch exists**, so a cap can never kill a winning run. APEX preserves this exactly: `budget{}` is first-class and may shape allocation (fewer/cheaper attempts), but **budget exhaustion never aborts an in-flight succeeding rollout** and never suppresses a candidate that has execution evidence. Loop-until-budget governs *whether to start more work*, not *whether to stop work that is winning*.

**Non-conflicts** worth noting explicitly: durable resume, `pipeline()`, context isolation, git-worktree isolation, fail-loud-never-fake, progress-based liveness, and manifest pinning are all **extensions of** existing v1 invariants, not tensions with them.

### 2.7 Concurrency caps, fail-loud, and what is NOT a law

Two final clarifications keep the substrate honest:

- **Concurrency caps are reference-impl constants, not laws.** The Claude Code engine’s `min(16, cores-2)` concurrency and 1000-agent lifetime are *that vendor’s* numbers. APEX uses its **own derivation** — v1 already does (`min(parallel_workers, requests)`, further capped by overlap-diversity and `global_parallel_worker_budget // outer_task_parallelism`). The proof points the paradigm cites (the Bun 750k-line port; "85 agents in 16 min") are vendor marketing — and the Bun port shipped 13,000+ `unsafe` blocks with "no human having fully read the codebase," a pointed reminder that **passing tests ≠ correctness** ([The Register](https://www.theregister.com/devops/2026/05/14/anthropics-bun-rust-rewrite-merged-at-speed-of-ai/)). None of these drive APEX’s design.

- **Fail-loud-never-fake is an engine rule.** No swallowing errors or substituting placeholder/mock data behind `try/catch`; the human is the final gate (read the git diff, run tests). v1 enforces this via the strict acceptance gate (the legacy `overall_score ≥ 0.9` shortcut was *removed*), `salvage != success` (ABSTAINED is a first-class `Status` peer), `HeuristicRepairAgent`’s apply-test-revert (a mutation is kept only if `test_command` returns 0, else byte-identical revert), and **fail-open instrumentation** — a sampling/watchdog bug can only *delay* a kill, never accelerate or fake one. Liveness is **progress-based, not wall-clock**: the S1–S7 watchdog and K1 stall measure stdout/stderr/worktree-edit/CPU progress, so a long legitimate thinking turn is never false-killed (the motivating bug — a confident 1.0 rollout discarded by a wall-clock cancel — is documented), with emergency-silence/no-edit backstops to reap a truly-wedged worker.

The net design: a deterministic, journaled, restart-survivable orchestration engine that expresses `agent`/`parallel`/`pipeline`/`phase`/`budget` over **vendor-neutral workers**, holds state in variables and a WAL rather than a context window, and treats the filesystem/git diff as the authoritative artifact — the substrate on which the verify-and-refute and evidence-grounded selection of Section 13 (the actual capability mechanism) run cleanly across full restarts and heterogeneous fleets.
