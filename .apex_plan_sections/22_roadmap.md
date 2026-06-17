## 22. Implementation Roadmap

This section sequences the build of APEX-Ω from APEX v1's hardened substrate to the full evidence-grounded, vendor-neutral dynamic-workflow orchestrator described in Sections 8–17. It is decisive about ordering, names the v1 code reused at each stage, makes every research-bearing amplifier a default-off ablation flag, and treats the held-out-vendor harness as a continuous gate rather than a closing chore. The governing fact is that **the engine, the normalized Executor, and the journal are the critical path**: nothing downstream — not speculative search (Section 9), not the model economy (Section 12), not the learned controller (Section 14) — is meaningful or even reproducible until those three exist and survive a kill-mid-run. After Phase 0, the remaining phases are largely parallelizable, because they ride on an instrumented substrate that records every `agent()` decision (see Section 15 for the journal, Section 14 for why the controller needs it).

This roadmap respects the foundational invariants verbatim: filesystem-as-source-of-truth, execution-evidence-authoritative selection (the Cardinal Safety Contract, Section 13), fail-loud-never-fake, durable resumable journaling, and vendor neutrality. None of the four phases may weaken any of these; amplifiers may only re-rank within an execution-verified tier or set search priors a controller can override — never promote an unverified candidate.

### 22.1 Sequencing rule and the critical path

The single ordering principle is **workflow-engine-first**. Concretely:

- Phase 0 is a hard prerequisite for everything: it builds the re-implementable engine (`agent / parallel / pipeline / phase / budget`), the normalized Executor with capability negotiation, and durable input-hash journaled resume. Its exit criterion is the foundational mandate made literal: the reference best-of-N workflow runs end-to-end on Codex, on Claude Code, and on a mixed fleet, produces results of v1 quality, and survives a kill-mid-run.
- Phases 1, 2, and 3 each depend on Phase 0, but only weakly on each other, so they can proceed by parallel sub-teams once Phase 0 ships. The one cross-phase dependency that must be honored: **Phase 3's learned controller requires the journaled decisions emitted from Phase 0 onward** (reproducible off-policy credit assignment is impossible without them), and benefits from Phases 1–2 being instrumented so its action space (allocation, gating, branching, economy) is real.

The deliberate inversion of the v3 redesign's instinct (start with the exciting amplifiers) is the central risk mitigation: do not start amplifiers before the engine and journal exist, because an amplifier without journaling cannot be ablated, cannot fail open reproducibly, and cannot move the headline number honestly.

#### Reuse split (target proportions)

| Bucket | Target share | What it is | Representative v1 assets |
|---|---|---|---|
| Generalize v1 | ~60% | Lift existing v1 machinery into engine primitives and library calls | `cli_backend.py`, `llm_routing.py`, `backend_portfolio.py`, `cli_turn_parser.py`, `FrontierSearchController`, `EpisodicMemoryBus`, `contract_slice.py`, verification cascade, worktree isolation, RunManifest, anti-cheat |
| Substrate grafts | ~25% | New plumbing that wraps/extends v1 to satisfy the paradigm | `pipeline()` primitive, per-`agent()`-call WAL (promote unused `ReplayRecorder` + escrow WAL), capability-negotiation layer, CTDG test edge, provider-cache adapter |
| New controller + eval science | ~15% | The defensible research contribution | active controller (bandit→GEPA→RL), learned capability/cost profiles, held-out-vendor harness, full eval suite |

The conservative ~85% (generalize + graft) is exactly what makes the high-variance ~15% research bet affordable; if the controller science fails to beat its fail-open heuristic, APEX-Ω still ships as a hardened, vendor-neutral best-of-N engine that beats v1 on cost (Phase 1) and matches it on quality.

### 22.2 Phase 0 — Vendor-neutral engine + reference workflow (critical path)

**Goal.** A re-implementable, deterministic, journaled orchestration engine exposing the five paradigm primitives, with a normalized Executor that runs Codex, Claude Code, or a mixed fleet, and durable restart-survivable resume. This is the substrate; it must be boring, correct, and vendor-blind.

**What is lifted, not rebuilt.** v1 already encodes the orchestration program as bespoke Python in `solver.py` (`_execute_with_dynamic_transitions`, `_execute_progressive_rollout_plan`). Phase 0 extracts that hard-coded pipeline into a library the program calls. Mapping (from the paradigm ingest, verified against v1):

| Engine primitive | v1 origin to generalize | Net-new work |
|---|---|---|
| `agent(prompt, opts)` | `CLIModelClient.run_structured_prompt` (spawns external CLI, observes stdout via S1–S7 watchdog, returns typed `CLIModelResult`, never raises) | Normalize return to `{final_message, structured_output?, usage, session_id, raw_events}` + observe git diff |
| `parallel(thunks)` | `RolloutEngine.execute_rollout_requests` (barrier fan-out, per-rollout worktree + fcntl lock, failures → classified failed results) | Conform to "failed thunk → null, filter before use" contract |
| `pipeline(items, ...stages)` | **No v1 analog** — all v1 fan-out is barrier waves | Build from scratch (the one genuinely net-new primitive) |
| `phase(title)` / `log(msg)` | per-phase `atomic_write_json` artifacts + `controller_decisions.jsonl` | Expose as API; couple `phase()` boundaries to journal checkpoints |
| `budget {total, spent(), remaining()}` | `repo_token_cap`, `max_tokens_per_repo_followup` (machinery exists, defaulted OFF) | Expose as first-class primitive, defaulted unbounded |

#### 22.2.1 The normalized Executor interface

The Executor is the load-bearing vendor-neutral abstraction. It must run on either Codex (`codex exec`) or Claude Code (`claude -p`) — a coding agent reading this plan must be able to implement against it on either CLI. Sketch (Python-flavored pseudocode; field types annotated):

```python
@dataclass
class ScopedTask:
    prompt: str                      # the scoped job (inspect/rewrite/test/review)
    schema: dict | None              # JSON Schema for structured return, if any
    allowed_tools: list[str]         # restricted tool set per worker
    sandbox: str                     # "read-only" | "workspace-write" | "apex-worktree"
    model: str                       # vendor-resolved at command-build time
    effort: str | None               # low|medium|high|xhigh|max (degrade if unsupported)
    internet: bool                   # capability-negotiated
    mcp_servers: list[McpRef]        # uniform tool plane injected into every vendor

@dataclass
class ExecResult:
    final_message: str
    structured_output: dict | None   # validated; None if schema unmet after retries
    usage: TokenUsage                # input/cached_input/output/reasoning tokens (normalized)
    session_id: str | None           # vendor session handle, for resume
    raw_events: list[dict]           # best-effort telemetry, NOT the contract
    fs_diff: str                     # git diff of worktree == the authoritative artifact

class Executor(Protocol):
    def negotiate(self, vendor: str, model: str, version: str) -> CapabilityProfile: ...
    def spawn(self, worktree_cwd: str, vendor: str, model: str, version: str) -> "Session": ...

class Session(Protocol):
    def run(self, task: ScopedTask) -> ExecResult: ...
    def observe_diff(self) -> str: ...    # git diff is ground truth (filesystem-as-truth)
```

Vendor adapters map the common surface to native flags. Per the vendor ingest these CLIs have converged on a near-identical headless contract:

| Capability | Codex (`codex exec`) | Claude Code (`claude -p`) | Degradation if absent |
|---|---|---|---|
| Single-shot headless | `codex exec` (alias `e`) | `-p / --print` | n/a (both have it) |
| Structured output | `--output-schema <file>` | `--json-schema` → `structured_output` | embed schema in prompt + post-parse |
| NDJSON event stream | `--json` (thread/turn/item events) | `--output-format stream-json` | parse final message only |
| Sandbox | `--sandbox {read-only\|workspace-write\|danger-full-access}` | `--permission-mode` + `--allowedTools` | wrap in APEX worktree + fcntl lock |
| Reproducible CI | `--skip-git-repo-check`, `--ignore-user-config` | `--bare` (skips hooks/skills/MCP/CLAUDE.md) | document & pin |
| Resume | `codex exec resume --last\|<SESSION_ID>` | `--input-format stream-json` continuation | rely on APEX journal (see below) |
| MCP tool plane | required-MCP config | `--mcp-config` | inject same MCP set into both |

**Capability negotiation, ACP-style.** Borrow the Agent Client Protocol `initialize` handshake pattern (each side advertises a typed capability set; optional methods unlock only when the matching capability is present). `negotiate()` returns a `CapabilityProfile{internet: bool, native_schema: bool, sandbox_levels: list[str], thinking: bool, bidirectional_stream: bool, mcp: bool}`. The hard rule is **degrade, do not crash**: no native schema → embed schema in prompt text + post-parse and retry on mismatch (this is exactly what v1 already does for gemini/codex via `_augment_prompt_for_backend`); no read-only sandbox → fall back to APEX's own worktree-plus-fcntl isolation, which v1 already owns. This consolidates v1's scattered per-vendor fragments (`_internet_launcher_args`, `CLIToolHookSupport`, `_CLIBackendSandboxSpec`, effort flags) into one layer instead of per-call special-casing.

Reuse `llm_routing.py`'s two-tier failure memory (call-failover for transient 429/stall vs backend-level global reroute for auth/missing-binary/SDK breakage) and `backend_portfolio.py`'s self-evicting ledger so a transient blip on one vendor cannot poison a heterogeneous fleet — essential the moment Codex and Claude run together.

#### 22.2.2 The `pipeline()` primitive (the one net-new build)

v1 has only barrier waves; the paradigm's per-item staged streaming (item A in stage 3 while item B in stage 1; wall-clock = slowest single chain, not sum-of-slowest-per-stage) has no v1 analog. The natural fit is the reproduce → localize → patch → verify cascade. Control flow:

```python
def pipeline(items, *stages, budget=None):
    # No barrier between stages. Each item streams through stages independently.
    # Scheduler holds many items at different stages simultaneously.
    in_flight = {stage_idx: queue() for stage_idx in range(len(stages))}
    enqueue(in_flight[0], items)
    results = {}
    while any(in_flight.values()):
        for stage_idx in reverse(range(len(stages))):   # drain later stages first (no head-of-line block)
            item = poll(in_flight[stage_idx])
            if item is None: continue
            key = (item.id, stage_idx)
            out = journal.get_or_run(key, lambda: stages[stage_idx](item))   # per-(item,stage) cache
            if stage_idx + 1 < len(stages):
                enqueue(in_flight[stage_idx + 1], out)
            else:
                results[item.id] = out
            if budget and budget.remaining() <= 0: break
    return results
```

The new bookkeeping is per-`(item, stage)` journaling (next subsection) and explicit inter-stage data contracts. This must be validated against the determinism invariant: stage scheduling order must be a pure function of item ids and stage indices, never wall-clock.

#### 22.2.3 Durable input-hash journaled resume (the "do better" mandate)

The reference impl's resume is session-scoped and does not survive a full restart; v1 is at the same limitation (`ReplayRecorder` has **no production callsite**; the escrow WAL/CCEDF is a narrow durability backstop for one confirmed-pass loss). Phase 0 promotes these into a per-`agent()`-call write-ahead log:

```python
def agent(prompt, opts):
    h = input_hash(prompt, opts.model, opts.vendor, opts.cli_version, scoped_inputs(opts))
    cached = journal.lookup(h)
    if cached is not None and cached.inputs_match(h):
        return cached.result            # unchanged call → cached result
    result = executor_run(prompt, opts) # edited/new call → re-run
    journal.append_fsync(h, result, manifest_pin)   # WAL: fsync-durable, seq-ordered, idempotent
    return result
```

Cache validity is **input-hash match**, not output reproduction — agent outputs are stochastic and (per Thinking Machines, batch non-invariance) temperature-0 is not bitwise reproducible across hosted APIs, so resume replays the *recorded* artifact (the diff + verification result), never attempts to reproduce token streams. This reuses v1's determinism substrate (temperature 0.0, pure failover ranking, bit-identical snapshot SHAs, `atomic_write_json`) which is exactly what input-hash journaling needs. The journal doubles as the off-policy credit substrate Phase 3 consumes.

**Exit criterion (must all hold).** (1) The reference best-of-N workflow runs end-to-end on Codex; (2) on Claude Code; (3) on a mixed Codex+Claude fleet; (4) produces results of v1 quality on a smoke benchmark slice; (5) survives `kill -9` mid-run and resumes from the journal, re-running only edited/new `agent()` calls. Test resume by killing mid-run and confirming the resumed run reproduces the artifact set (durable-execution best practice).

### 22.3 Phase 1 — Cheap, safe, evidence-positive efficiency wins (parallelizable)

**Goal.** Cut cost without risking the headline number, mostly by flipping v1 seams default-on with a quality SLA and adding portable cost machinery. Every item here has measured, stacking, evidence-positive support and respects the Cardinal Contract (none may suppress a candidate without execution evidence).

| Lever | v1 reuse | Disposition / source | Safety property |
|---|---|---|---|
| Difficulty-adaptive low-K | `enable_adaptive_allocation` (exists, OFF), `estimate_difficulty` → `compute_rollout_count` → `_clamp_rollout_bucket` | adopt, default ON with quality SLA | Optimal K often <10; biggest single cost lever |
| Early localization-futility gate | amortized localization + `EpisodicMemoryBus` | adopt | Routes budget to surviving hypotheses; never suppresses a candidate without execution evidence |
| Prefix-stable prompt assembly + provider-cache adapter | prompt templates | adopt | Cached reads ~0.10x input; portable across opaque vendor CLIs |
| Cascade escalation (cheap → cheap-verify → frontier) | verification cascade (Section 13) | adopt | Cascade-over-route avoids xRouter brittleness; escalate on verify-on-diff failure |
| `(vendor, model)` diversity axis | `LLMBackend` portfolio | adopt | Cross-vendor decorrelates hallucinations (Devlo/TRAE) |
| Test-impact pruning (CTDG) | `RepoGraph`, `CoverageReport`, `prune_by_regression` | adopt-modified (Section 10) | Reorder + dynamic-coverage prune + full-suite backstop; static-as-gate rejected |

**Adaptive low-K (default ON).** Flip `enable_adaptive_allocation` on, gated by a quality SLA that reverts to the higher K if measured resolve-rate drops below baseline minus a tolerance on a continuous canary slice. This is the cost lever with the clearest ROI; v1 already has the difficulty estimator and bucket clamps.

**Localization-futility gate.** After amortized localization, compute a per-hypothesis survivability signal; route remaining budget away from hypotheses with no surviving evidence. Critically this *informs allocation* — it never excludes a candidate pre-execution, which would invert the Cardinal Contract.

**Prefix-stability contract + cache adapter.** Make every agent template emit `[stable: tooling + system + policies]` then `[volatile: task + live context]`; lint/forbid timestamps, UUIDs, session IDs, and dynamic tool sets inside the stable region. A pure API consumer cannot literally share KV across forks, so the portable win is maximizing prefix hit-rate plus dispatch ordering (longest-shared-prefix-first). Compile the stable spans into Anthropic `cache_control` breakpoints or rely on OpenAI/Google auto-cache + a pinned `prompt_cache_key`. Caution: caching below the per-model min token threshold causes a 10–18% TTFT regression — selectivity is mandatory.

**Test-impact pruning.** Add a code→test edge to v1's `RepoGraph` (which today stops at source-to-source edges), ideally seeded by a one-time dynamic coverage run rather than pure static reachability. Use it only for reordering (zero false-negative risk) and dynamic-coverage pruning (near-safe), with a full-suite backstop. Static-AST CTDG as a gate is **rejected** (PyCG ~70% recall; reflection/monkeypatch/fixtures invisible; gating silently drops fault-revealing tests, violating execution-authority).

**Dependencies.** Phase 0 engine + journal only. All six levers are mutually independent and can be built by parallel sub-teams.

### 22.4 Phase 2 — Bounded amplifiers (behind default-off ablation flags)

**Goal.** Add the redesign's bolder mechanisms as *amplifiers* of the three capability properties (Section 7), each behind a default-off flag that **fails open to the heuristic baseline** so enabling an experiment cannot move the headline number. Each item carries an adversarial disposition that must be honored.

| Amplifier | v1 reuse | Disposition | Bound / mitigation |
|---|---|---|---|
| Bounded adaptive-branching + `speculate()` | extend `FrontierSearchController` (PUCT, virtual-loss, `max_depth`, `max_frontier_branching`, `min_branch_reward`, value backup, early-stop) | adopt-modified (Section 9) | AB-MCTS wider/deeper/diversify Thompson sampling (content-hash-seeded); runs inside FrontierSearch budget caps; **mandatory collapse to verified best-of-N below a feedback-confidence floor** |
| Agent-initiated `speculate()` fork | `FrontierSearch` ranking; worktree fork | adopt-modified | Admit only at turn/checkpoint boundaries feeding ranking/budget; constant-factor cheaper via prefix reuse, not exponential; bounded by virtual-loss/`min_branch_reward` |
| Hybrid verifier (execution + generative critic) | verification cascade | adopt (Section 13) | Critic is discrimination-only: breaks ties **only within** the execution-verified tier; R2E-Gym ~43%→51% |
| Blackboard 2.0 (push-at-turn-boundary, abstracted negatives) | evolve `EpisodicMemoryBus` *delivery* | adopt-modified (Section 11) | Keep relevance/confidence/dedup/own-rollout-exclusion; abstracted negatives preserve diversity; verifier must not see producer context |
| Model-economy contract cascade by sub-role | reuse `contract_slice.py` | adopt-modified (Section 12) | Cheapen run/verify + narrow edits only; keep frontier on navigation/multi-file edits (HyperAgent); escalate on first verify-on-diff failure with a rewrite-cycle cap |

**Critical constraints carried from the adversarial verdicts and the in/out list.** Distributed/classical MCTS as the *core loop* is **rejected** (re-describes FrontierSearch; brittle against non-serializable container state) — the adopted form is bounded adaptive-branching *inside* FrontierSearch. Raw share-all / mid-subprocess injection blackboard is **rejected** (share-all lowers accuracy ~-3.7pp; mid-subprocess mutation is infeasible against opaque CLIs and breaks determinism/replay) — hence push only at turn boundaries. Heavy-orchestrator + thin-executor as the *default* shape on hard repo SWE is **rejected** (HyperAgent: cheapening navigation/multi-file editing causes the worst resolve-rate drops; the almost-right trap can cost more than one frontier pass) — hence the economy cheapens only run/verify/narrow-edits.

Each flag must have an explicit fail-open: `applied=False ⇒ value==baseline`, mirroring v1's `evaluate_policy_model` discipline. The branch primitive's collapse-below-confidence-floor rule means that when execution feedback is thin (the dangerous regime), the system reverts to the best-of-N floor it can never do worse than.

**Dependencies.** Phase 0 (engine + journal for reproducible flag ablation); Phase 1 is helpful but not strictly required. Sub-teams can build the four amplifiers concurrently because each is independently flagged.

### 22.5 Phase 3 — Active controller + the paper (rides on the instrumented substrate)

**Goal.** Promote the controller from v1's passive blend-into-heuristics layer to an active, vendor-agnostic policy that steers backend choice, allocation, branching, gating, and economy — staged bandit → GEPA → RL — plus the learned capability/cost profiles and the full evaluation science. This is the defensible NeurIPS-grade contribution (Section 14, Section 19).

**Build the held-out-vendor harness FIRST.** Do not defer it. It is a continuous gate from day one of Phase 3: the controller and profiles are trained/calibrated on a set of vendors+models, then evaluated on a *held-out* vendor (e.g., train on Codex+Claude, test on Gemini/opencode) to prove the policy generalizes rather than overfitting one vendor's quirks. Wiring it as a gate is the antidote to the pitfall of a learned controller that silently encodes a single vendor's behavior. Reproduce **artifacts** (diffs + re-run verification), not token streams, since temperature-0 is not reproducible across hosted APIs.

Staging (the disposition list is explicit here):

| Stage | Mechanism | Disposition | Why this order |
|---|---|---|---|
| Stage-0 | Preference-conditioned contextual bandit / REINFORCE router; finally wire/remove `library_enabled` | adopt (active control day one) | Learns online from the verifier signal v1 already emits; **blend-not-switch**, `applied=False ⇒ value==baseline`; fail-open to heuristic |
| Stage-1 | GEPA-style reflective prompt evolution of the controller's instructions | defer to here | 35x cheaper than RL; prompts-not-weights = inherently vendor-agnostic |
| Stage-2 | Cost-penalized REINFORCE/GRPO over orchestrator decisions, emitting plans as parsable structures | defer (volume-gated) | Only when volume justifies; cost-penalized terminal reward keyed on deterministic verifiers; short episodes to limit credit diffusion |

**Learned capability/cost profiles.** Build alongside the harness: per-`(vendor, model)` profiles of capability (resolve rate by task class) and normalized cost (token *yield*, not invoice — the xRouter lesson). These feed both the economy cascade routing and the controller's action space.

**The full eval suite (Section 20).** Run the cost-matched Pareto comparison, the re-trained-baseline control, the held-out-vendor generalization test, the emergent-structure analysis, and the "when orchestration hurts" study. The controller's defensibility rests on showing it blends-not-switches, fails open, and generalizes across vendors.

**Dependencies.** Phases 0–2. The hard one: **Phase 3 needs the journaled decisions from Phase 0 onward** — reproducible off-policy credit assignment over a deterministic journal is what makes a learned orchestration controller trainable and auditable at all. The controller's action space (allocation from Phase 1, branching/economy from Phase 2) must exist and be instrumented before it can learn to drive them; the bandit fails open to those phases' heuristics until it earns trust.

### 22.6 Cross-phase invariants and pitfalls

Three pitfalls govern the whole roadmap and are restated as hard rules:

- **Do not start amplifiers before the engine and journal exist.** Any Phase 2/3 work begun before the Phase 0 exit criterion passes is unablatable and cannot fail open reproducibly.
- **Do not rebuild what v1 already provides — lift and generalize.** The ~60% reuse target is a directive: `cli_backend.py` / `llm_routing.py` / `backend_portfolio.py` / `cli_turn_parser.py` are the bought-and-paid-for vendor-agnostic foundation; `FrontierSearchController`, `EpisodicMemoryBus`, `contract_slice.py`, the verification cascade, worktree isolation, RunManifest, and the anti-cheat kernel are library primitives the engine calls, not things to reinvent.
- **Do not defer the held-out-vendor harness.** Build it as a continuous gate at the start of Phase 3, not as a closing validation step.

Every phase preserves the foundational frame: filesystem-as-source-of-truth (verification on the resulting git diff regardless of which vendor produced it), execution-evidence-authoritative selection (Section 13's Cardinal Contract — soft/learned/critic signals re-rank within a tier or downgrade an accepted candidate, never promote an unverified one), fail-loud-never-fake (strict acceptance gate, salvage≠success, first-class abstention), durable resumable journaling (Section 15), and vendor neutrality (the normalized Executor + capability negotiation). The conservative kernel is precisely what makes the high-variance controller research affordable: if Stage-1/2 fail, APEX-Ω still ships as a hardened, cheaper, vendor-neutral best-of-N engine.

### 22.7 Milestone summary

| Phase | Headline deliverable | Primary v1 reuse | Exit / gate |
|---|---|---|---|
| 0 (critical path) | Vendor-neutral engine (`agent/parallel/pipeline/phase/budget`) + normalized Executor + capability negotiation + durable journaled resume | `cli_backend`, `llm_routing`, `backend_portfolio`, `cli_turn_parser`; kernel as library | Reference workflow runs on Codex / Claude / mixed at v1 quality; survives kill-mid-run |
| 1 (parallelizable) | Cost wins: adaptive low-K, futility gate, prefix-cache, cascade, `(vendor,model)` axis, test-impact pruning | `enable_adaptive_allocation`, difficulty estimator, `RepoGraph`, verification cascade | Lower cost at ≥ baseline quality (SLA) |
| 2 (default-off flags) | Bounded branching + `speculate()`, hybrid verifier, Blackboard 2.0, model economy | `FrontierSearchController`, `EpisodicMemoryBus`, `contract_slice` | Each flag fails open; ablations show net-positive or neutral |
| 3 (rides substrate) | Active controller (bandit→GEPA→RL) + capability/cost profiles + held-out-vendor harness + full eval suite | `controller_policy` blend discipline; journaled decisions from Phase 0+ | Held-out-vendor generalization proven; blend-not-switch; fail-open verified |
