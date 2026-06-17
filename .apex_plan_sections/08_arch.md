## 8. Target Architecture Overview

APEX-Ω is organized as **four concentric layers** wrapped by **two cross-cutting planes**. The inner layers are the parts we cannot get wrong; the outer layers are the parts that, if they go wrong, must fail *toward* the inner layers rather than around them. This is the structural expression of the thesis in Section 7: the engine and the executor plane are necessary plumbing; the execution-authoritative kernel is the mechanism that converts diverse candidates into trustworthy resolutions; and the amplifiers are bounded extensions that may only *steer*, never *promote*.

The four layers, from inside out:

| Layer | Name | What lives here | Disposition source |
|---|---|---|---|
| **L0** | Vendor-neutral workflow engine | `agent` / `parallel` / `pipeline` / `phase` / `budget`; state in script variables + a journal | adopt (generalize v1) |
| **L1** | Executor plane | Normalized `Executor` over `codex_cli` / `claude_cli` / `gemini_cli` / `opencode` / `openai_api`; capability negotiation; observe-the-diff | adopt (consolidate v1) |
| **L2** | Execution-authoritative kernel | Cardinal Safety Contract; cheap-first verification cascade; worktree isolation + `fcntl` locks; RunManifest + Docker digest pinning; anti-cheat / fairness / failure taxonomy / abstention | adopt (v1 verbatim) |
| **L3** | Amplifier layers | Bounded adaptive-branching search; CTDG hint-prioritizer; epistemic blackboard 2.0; model economy; hybrid verifier; **active controller** | adopt-modified (judged extensions) |

The two cross-cutting planes — **the durable journal + replay** and **vendor neutrality** — touch every layer rather than sitting at one altitude, and are drawn alongside the stack rather than inside it.

The single most important architectural rule, repeated throughout this plan and enforced structurally below, is the **composition rule**:

> **L3 amplifiers may feed priors, ordering, and budget into L0–L1, but every candidate diff flows through L2, and L2's Cardinal Safety Contract is inviolate. No amplifier — including the active controller — may promote an unverified candidate. The controller sits *above* selection (it shapes what is generated and in what order) but *below* the Cardinal Contract (it cannot override an acceptance decision).**

This is not stylistic. The adversarial verdict on the substrate claim is explicit: capability comes from execution-grounded selection, not from orchestration topology, and "attributing the capability gain to [the] deterministic dynamic-workflow engine rather than to execution-grounded verification/selection running on that substrate is a category error that, if it drives prioritization, risks the classic failure: invest in fan-out/agent-count instead of judge quality." The layering exists to make that category error structurally impossible: an amplifier that could promote an unverified diff would be a layer-violation the type system rejects, not a tuning mistake.

### 8.1 The diff-as-truth acceptance boundary (component/flow sketch)

The diagram below is the canonical sketch for the whole plan. Read it top-to-bottom as data flow and note the two cross-cutting planes on the right. The load-bearing line is the one marked `DIFF = source of truth`: that horizontal cut is the **acceptance boundary**, and it is *vendor-blind*. Above it, anything may happen (any vendor, any model, any amplifier-chosen order). Below it, only execution evidence decides.

```
                              ┌───────────────────────────────────────────────┐        ┌──────────────────────────┐
 user(repo, issue, test) ───▶ │  L0  WORKFLOW ENGINE                          │ ─────▶ │ CROSS-CUTTING:           │
                              │  agent / parallel / pipeline / phase / budget │ journal│  DURABLE JOURNAL + REPLAY │
                              │  (all run-state in script variables)          │ every  │  per-agent()-call WAL    │
                              └───────────────┬───────────────────────────────┘ call   │  input-hash cache        │
                                              │ authors + executes program            │  reproduce ARTIFACTS     │
                              ┌───────────────▼───────────────┐                        │  (diffs), not tokens     │
                              │ L3  ACTIVE CONTROLLER          │◀───── priors/profiles  └──────────────────────────┘
                              │ per node: {wider, deeper,      │
                              │   speculate, stop} +           │        (controller is ABOVE selection,
                              │   worker-PROFILE target        │         BELOW the Cardinal Contract)
                              │ blend-not-switch, fail-open    │
                              └───────────────┬───────────────┘
                                              │ dispatch ScopedTask
        ┌─────────────────────────────────────▼─────────────────────────────────────┐    ┌──────────────────────┐
        │ L1  EXECUTOR PLANE  (capability negotiation, graceful degradation)         │    │ CROSS-CUTTING:       │
        │   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │    │ VENDOR NEUTRALITY    │
        │   │ codex_cli│ │claude_cli│ │gemini_cli│ │ opencode │ │openai_api│ ...     │ ◀──│ filesystem-as-truth  │
        │   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘         │    │ RunManifest pins     │
        └────────┼────────────┼────────────┼────────────┼────────────┼──────────────┘    │ {vendor,model,ver}   │
                 │  each worker in its OWN git WORKTREE (fcntl lock)  │  ◀── MIXED in one run │ two-tier failover    │
        ┌────────▼────────────▼────────────▼────────────▼────────────▼──────────────┐    └──────────────────────┘
        │ L3  AMPLIFIERS (HINTS ONLY)                                                │
        │   bounded adaptive branch │ CTDG test-order prior │ blackboard 2.0 (neg)   │
        │   model economy (heavy-plan + cheap-exec cascade)                          │
        └─────────────────────────────────┬─────────────────────────────────────────┘
                                           │ produces CANDIDATE DIFFS
        ┌──────────────────────────────────▼─────────────────────────────────────────┐
        │ L2  EXECUTION-AUTHORITATIVE KERNEL                                           │
        │   cheap-first cascade (syntax→lint→reproduce→regression-prune→Nxn xval)      │
        │     — NEVER synthesizes a pass —                                             │
        │   → hybrid verifier (execution score + generative critic, discrimination)    │
        │   → CARDINAL CONTRACT selection (soft signals re-rank-within / downgrade)    │
        │   → Status ∈ {SOLVED, ABSTAINED, FAILED, ENV_SKIPPED}                        │
        └──────────────────────────────────┬─────────────────────────────────────────┘
                                           │
            ═══════════════════════════════╪═══════════════════════════════════════════  ◀── ACCEPTANCE BOUNDARY
                                           │   DIFF = source of truth (vendor-blind accept)
                                           ▼
                       unified diff + Status (SOLVED / ABSTAINED / FAILED / ENV_SKIPPED)
```

Three things to read off this sketch:

1. **Amplifiers feed *into* the top of generation, not into selection.** The CTDG, blackboard, model economy, and adaptive-branch arrows all point *down* into L1 dispatch or *across* into the controller's priors. None of them touches the L2 box except by producing candidate diffs that L2 then judges. This is the composition rule drawn literally.
2. **The controller is one box, drawn above L1 and below L2.** It can change *which* worker runs, *how many* run, and in *what order*, and it can `stop`. It cannot reach below the acceptance boundary. (Contrast: a naive "active orchestrator" would be drawn straddling L2 — that is the pitfall this layout exists to prevent, and is exactly the "Do not draw the controller as bypassing the execution-authoritative kernel" guardrail.)
3. **Vendor neutrality is realized *at* the acceptance boundary.** Because acceptance is computed on the git diff, the executor plane can be a heterogeneous fleet — a Claude branch, a Codex branch, and a cheap Codex-Haiku contract-executor leaf in the *same run* — and the kernel never needs to know or trust which vendor produced any diff. The filesystem is the contract; vendor JSON streams are telemetry. This is v1's existing property (`_selected_result_is_accepted` runs on verifier evidence over the diff, not on any vendor self-report) generalized into the engine.

### 8.2 How the v1 substrate composes with the amplifier layers

The redesign is **not a rewrite**. The adversarial verdict is blunt that v1 "already converged on the orchestrator-worker model… so the redesign is a generalization/lift of working code; the only net-new builds are `pipeline()` and true journaled resume — bounding execution risk." The architecture therefore maps each amplifier onto an *existing* v1 seam and constrains it to feed L2, never to replace it.

| Concern | v1 mechanism (kept) | L3 amplifier (added) | Composition: amplifier → L2 relationship |
|---|---|---|---|
| Where to spend samples | `RolloutEngine.execute_rollout_requests` barrier fan-out (= `parallel`); `FrontierSearchController` (PUCT, virtual loss, `max_depth=6`, `max_frontier_branching=3`, `min_branch_reward`) | Bounded adaptive-branching (wider/deeper/diversify, budget-aware) + agent-initiated `speculate()` at turn boundaries (see §9) | Amplifier sets branch *priors* and *budget share*; **every branch's diff still flows through the L2 cascade**; collapses to verified best-of-N below a feedback-confidence floor |
| Which tests to run first | `prune_by_regression` (baseline-passers, chunks of 50); `RepoGraph` (no test edges) | Hybrid CTDG = static import/call graph for breadth + one-time dynamic coverage; **reorders** test execution and seeds priors (see §10) | CTDG is a **hint only**; full regression-prune remains the authoritative gate + full-suite backstop; static-as-gate is rejected (PyCG ~70% recall) |
| Cross-rollout knowledge | `EpisodicMemoryBus` (append-only, shares *negative*/ruled-out discoveries, relevance-ranked, own-rollout-excluded, `positive_limit=5`/`negative_limit=3`) | Blackboard 2.0: push abstracted negative constraints at turn boundaries (see §11) | Shares *constraints*, never raw trajectories or the verifier's producer-context; delivery schedule evolves, storage/guards reused |
| Cost | Disabled-by-default token caps (`repo_token_cap=None`); `BackendPortfolio`; two-tier failure memory | Model economy: heavy planner + cheap cross-vendor executor *cascade* by sub-role (see §12) | Cheapen run/verify + narrow edits only; **escalate to frontier on first verify-on-diff failure**; never gate a candidate pre-execution |
| Selection | Cardinal Safety Contract; `_apply_evidence_bound_review` (flips `accepted` True→False only); deterministic lexicographic ranking tuple | Hybrid verifier: execution score + swappable generative critic for discrimination among equally-passing patches (see §13) | Critic re-ranks **within** the execution-verified tier only; execution score is authoritative; abstention is first-class |
| Steering | Passive controller (`evaluate_policy_model`, `applied=False ⇒ value==baseline`); `library_enabled` is *unwired* | Active controller over learned capability/cost profiles; staged bandit → GEPA → RL (see §14) | Blend-not-switch; fail-open to heuristic; wire-or-remove `library_enabled`; structurally cannot promote unverified |

**Where reused v1 components live in the new architecture** (the explicit mapping the chief guidance requires):

- **`FrontierSearchController`** → **L3 search.** It is the substrate for the adaptive-branching amplifier (§9). The redesign's "novel MCTS" is, per the panel and the verdicts, largely a re-description of this existing PUCT/best-first tree; we present `speculate()` as an *extension* (a second target source feeding the same ranking/budget/virtual-loss machinery), not a from-scratch tree. This both avoids the combinatorial blow-up the redesign omits and avoids a prior-art rejection.
- **`EpisodicMemoryBus`** → **L3 blackboard.** It already does the hard part the SOTA digest validates (abstracted *negative* sharing, relevance/confidence/dedup, own-rollout exclusion). Blackboard 2.0 evolves only the *delivery schedule* (pull-at-boundary → push-at-turn-boundary), keeping every guard. Crucially, the verifier must never see producer context (the "collective delusion" warning from MAST).
- **`cli_backend.py` / `llm_routing.py` / `backend_portfolio.py` / `cli_turn_parser.py`** → **L1 executor plane.** `CLIModelClient.run_structured_prompt` *is* `agent()` already (multi-vendor, returns a normalized `CLIModelResult`, never raises). The new work is consolidating the scattered per-vendor fragments (`_internet_launcher_args`, schema delivery, `CLIToolHookSupport`, `_CLIBackendSandboxSpec`, effort levels) behind one capability-negotiation handshake.
- **`contract_slice.py`** → **L3 model economy.** The contract-authoring scaffold already exists; the economy adds the heavy/cheap *cascade* on top.
- **Cardinal Safety Contract, cheap-first cascade, worktree isolation, RunManifest + Docker pinning, anti-cheat/fairness, failure taxonomy, abstention** → **L2, verbatim.** These are the parts the verdicts call "non-negotiable for a mixed-vendor fleet," and they are copied, not changed.
- **`ReplayRecorder` (no production callsite) + escrow WAL/CCEDF (narrow)** → **cross-cutting journal/replay,** promoted from a narrow backstop into a per-`agent()`-call WAL (§15).

The reuse ratio is deliberately high (the verdict's framing: roughly 60% reuse/generalize, 25% substrate grafts, 15% genuinely-new controller/evaluation). The two genuinely net-new builds are `pipeline()` (the one primitive with no v1 analog) and durable input-hash journaled resume; everything else is a lift behind the composition rule.

### 8.3 L0 — the vendor-neutral workflow engine

L0 is a **deterministic orchestration program** that holds all run-state in script variables and a journal — never in a conversation window. This is the "context isolation as scaling unlock" property, independently corroborated by context-rot results across 18 frontier models ([Chroma](https://www.trychroma.com/research/context-rot)) and lost-in-the-middle ([Liu et al.](https://arxiv.org/abs/2307.03172)). It exposes five primitives:

```text
agent(prompt, opts) -> AgentResult
  opts: { schema?, model?, vendor?, label?, phase?, isolation?, effort? }
  AgentResult: { final_message: str, structured_output?: object,
                 usage: Usage, session_id: str, raw_events: list,
                 fs_diff: UnifiedDiff }            # fs_diff observed from the worktree
  # generalizes v1 CLIModelClient.run_structured_prompt; never raises (typed result)

parallel(thunks) -> list[AgentResult | null]      # BARRIER fan-out; failed thunk -> null; MUST filter
  # generalizes v1 RolloutEngine.execute_rollout_requests

pipeline(items, *stages) -> list[StageResult]     # PER-ITEM streaming, NO inter-stage barrier
  # NET-NEW. wall-clock = slowest single chain (not sum-of-slowest-per-stage)
  # default for multi-stage work: reproduce -> localize -> patch -> verify

phase(title) / log(msg)                            # narration aligned to journal checkpoints
budget { total, spent(), remaining() }            # shared ceiling; supports loop-until-budget
```

Determinism is load-bearing because it is the precondition for the journal/replay plane. `Date.now`/`Math.random` equivalents are unavailable in the orchestration program (they would break replay), mirroring Temporal's "non-determinism is fatal" constraint ([Temporal durable execution](https://learn.temporal.io/tutorials/go/background-check/durable-execution/)). v1 already satisfies the spirit (temperature 0.0, pure `assign_strategy`/failover ranking, bit-identical snapshot SHAs, atomic writes).

**Conflict to respect (from the verdict on the substrate claim):** if a *model* authors the orchestration script, that injects non-determinism into the exact layer v1 keeps pure. The resolution is **freeze-then-journal**: a deterministic planner (or a model-authored-then-frozen-and-hashed script) emits the workflow; the emitted script is snapshotted into the RunManifest and journaled as a deterministic artifact, so replay always runs over a *frozen* script. Live model output is never un-journaled control flow.

**`budget{}` caveat (conflict with v1's "never optimize for cost"):** `budget{}` is first-class but defaults **unbounded**, and a budget exhaustion **must never abort an in-flight succeeding rollout** — v1's invariant (`_cumulative_token_cap_exceeded` fires only when no successful patch exists). A naive `loop-until-budget` that killed a winning run would violate this and is forbidden.

### 8.4 L1 — the executor plane (Codex / Claude / mixed, with capability negotiation)

L1 turns any agent CLI/API into a leaf worker behind one interface. The interface is the same whether the build runs on Codex or Claude Code, which is what lets a coding agent implement this section on either backend.

```text
Executor.spawn(worktree_cwd, vendor, model, version) -> Session
Session.run(task: ScopedTask) -> ExecResult
  ScopedTask: { prompt, schema?, allowed_tools, sandbox_mode, effort?, mcp_servers }
  ExecResult: { final_message, structured_output?, usage, session_id, raw_events }
Session.observe_diff() -> UnifiedDiff               # the contract; JSON streams are telemetry
```

One adapter per backend maps native flags to this contract (these flag mappings are drawn from the vendor research and are how the build runs on each):

| Vendor (v1 `LLMBackend`) | Headless invocation (illustrative) | Schema delivery | Sandbox |
|---|---|---|---|
| `codex_cli` | `codex exec --json --sandbox workspace-write --skip-git-repo-check --output-schema F` | native `--output-schema` | 3 levels (read-only default) |
| `claude_cli` | `claude -p --output-format stream-json --json-schema --allowedTools … --permission-mode acceptEdits --mcp-config …` | native `--json-schema` → `structured_output` | permission-mode + allowedTools |
| `gemini_cli` | `gemini -p --output-format stream-json --yolo` | **none native** → embed in prompt + post-parse | `--yolo` all-or-nothing |
| `opencode` | `opencode run --attach …` / `opencode serve acp` | via OpenAPI / ACP | server perms |
| `openai_api` | in-process `chat.completions` (v1 fallback path) | prompt-embedded | n/a (APEX worktree) |

A single **capability-negotiation handshake** (modeled on [ACP `initialize`](https://agentclientprotocol.com/get-started/introduction)) consolidates v1's scattered fragments into one `CapabilityProfile` per `(vendor, model)`:

```text
CapabilityProfile {
  internet: bool, native_schema: bool, sandbox_levels: enum,
  thinking_effort: bool, bidirectional_stream: bool, mcp: bool
}
# graceful degradation (degrade, never crash):
#   no native_schema      -> embed schema in prompt + post-parse + retry-on-mismatch
#   no read-only sandbox  -> wrap in APEX's own worktree + fcntl isolation (the L2 floor)
#   no internet flag      -> route internet-needing tasks to a capable vendor or abstain on that capability
```

**Honesty caveat (from the vendor verdict, `sound_with_caveats`):** "without losing the paradigm benefits" is an overclaim; the truth is **bounded, declared degradation**. Per-vendor capability *does* differ (native vs prompt-embedded schema; sandbox granularity; tool-interception gaps where the family-disjoint reviewer fails open). The profile makes the degradation explicit and routes around it; it does not erase it. And because **harness dominates model** (Terminal-Bench 2.0 shows 30–50pt same-model swings across harnesses), faithful flag/sandbox/schema mapping is a *load-bearing* engineering requirement, not a nicety — a normalization-leaky adapter can erase the entire cross-vendor diversity gain. Pin and record the resolved CLI version (e.g. `@openai/codex@X.Y.Z`) in the RunManifest; Codex's `0.134.0` profile breaking change is the cautionary tale.

Two-tier failure memory and the self-evicting `BackendPortfolio` are reused verbatim so a transient 429 on one vendor (call-failover, current-stage reroute) never poisons a healthy backend (global reroute, reserved for auth/missing-binary/SDK breakage).

### 8.5 L2 — the execution-authoritative kernel (v1 verbatim, the trust anchor)

L2 is copied from v1 with no semantic change, because it is the layer the entire credibility argument rests on and the layer the SOTA evidence says is the actual capability lever:

- **Cardinal Safety Contract.** Execution evidence is authoritative; soft signals may re-rank *within* an execution-verified tier or *downgrade* an accepted candidate, **never promote** an unverified one. Enforced structurally: `_apply_evidence_bound_review` flips `accepted` only True→False, and the deterministic ranking tuple places every soft/learned/LLM key strictly below every execution key. This is what converts repeated sampling into trustworthy resolutions rather than the false-positive inversion that the resampling-limits work ([2411.17501](https://arxiv.org/abs/2411.17501)) and reward-hacking-scales-with-capability ([ImpossibleBench](https://arxiv.org/html/2510.20270v1): GPT-5 hacks 76%) prove is fatal. CodeMonkeys quantifies the stakes: 69.8% coverage collapses to 57.4% after selection ([2501.14723](https://arxiv.org/abs/2501.14723)) — the substrate buys coverage, the verifier buys realized capability.
- **Cheap-first verification cascade that never synthesizes a pass:** syntax → lint → reproduction → regression-prune → NxN cross-validation → score → accept. Encoded countermeasures kept verbatim: `rc==0` silent no-op → `errors=1` (never `passed=1`); timeout `rc==124` → separate `regression_inconclusive` axis (`+0.15` partial), not a failure; singletons abstain (empty cross-validation list), never a synthetic `0.5` prior.
- **Per-rollout git-worktree isolation + `fcntl` locks + deterministic snapshot SHAs.** Three-tier degradation (seed_clone → worktree → snapshot → synthetic); the per-rollout advisory lock is taken *before* touching the workspace. This is the primitive that makes any L3 branching safe (CAID ablation: worktree 63.3 > single 57.2 > soft 55.5).
- **RunManifest + Docker digest pinning; anti-cheat / fairness; 8-bucket failure taxonomy; first-class abstention.** ABSTAINED is a peer of SOLVED; salvage is not success unless `rollout.allow_salvage=True`; the upstream Docker harness is the only publishable number.

Because L2 is the only layer that may write the acceptance decision, the composition rule reduces to a single invariant a reviewer (or a type-checker) can audit: **the only producer of `accepted=True` is the L2 cascade.** Every L3 component is read-only with respect to that field.

### 8.6 L3 — the amplifier layers (judged extensions, all default-gated)

L3 holds the redesign mechanisms, each `adopt-modified` per the in/out list and each behind a default-off ablation flag so enabling an experiment cannot silently move the headline number (v1's triple-gate discipline). The detailed designs are deferred to their sections; here we fix only their *position* and their *contract with L2*:

- **Bounded adaptive-branching search (§9):** wider/deeper/diversify by Thompson sampling, conditioned on remaining budget (the AB-MCTS finding that adaptive branching beats both best-of-N and plain MCTS *once budget is non-trivial*, while plain MCTS does not — [AB-MCTS](https://arxiv.org/abs/2503.04412), [SWE-Search](https://arxiv.org/abs/2410.20285)). Runs inside `FrontierSearchController` budget caps; **mandatory collapse to verified best-of-N below a feedback-confidence floor** — the guaranteed floor means we can never do worse than v1.
- **CTDG (§10):** test-impact map used to *prioritize* test order and seed branch priors; **never a gate.** Hybrid static + one-time-dynamic coverage with a full-suite backstop. Static-AST-as-gate is rejected.
- **Epistemic blackboard 2.0 (§11):** push abstracted negative constraints at turn boundaries; verifier never sees producer context.
- **Model economy (§12):** heavy planner authors test-anchored contracts; cheap cross-vendor executors satisfy them for run/verify and narrow edits only; **cascade (escalate on first verify-on-diff failure), not blind up-front routing** ([xRouter](https://arxiv.org/html/2510.08439v1) shows static routing trees are brittle; [HyperAgent](https://arxiv.org/html/2409.16299v1) shows cheapening navigation/multi-file editing hurts most).
- **Hybrid verifier (§13):** execution + swappable generative critic, discrimination-only within the verified tier ([R2E-Gym](https://arxiv.org/html/2409.16299v1)-style ~43%→51% gains come from tie-breaking, not promotion).
- **Active controller (§14):** the defensible NeurIPS-grade contribution. Sits above selection, below the Cardinal Contract; blend-not-switch; fail-open to heuristic; staged bandit → GEPA → RL; wire-or-remove `library_enabled`.

### 8.7 The cross-cutting planes

**Durable journal + replay** is drawn beside the stack because it observes every layer. Every `agent()` call is journaled to a WAL keyed by `hash(prompt + model + vendor + scoped_inputs)`; on restart, unchanged calls replay cached results and only edited/new calls re-run (the explicit "do better than the reference impl" mandate — the reference dynamic-workflow resume is session-scoped and "starts the workflow fresh" on restart). The journal model follows durable-execution practice: deterministic orchestration script = the "workflow"; non-deterministic `agent()`/tool/shell calls = "activities" with **at-least-once + idempotency keys** (`run_id + node_id + attempt`) for external side effects ([DBOS Postgres-as-journal](https://docs.dbos.dev/why-dbos) is the most self-hostable, vendor-neutral choice). Critically, **replay reproduces artifacts (diffs + re-run verification), not token streams** — bit-reproducible agent output is impossible across hosted APIs (batch non-invariance: temp-0 gave 80 distinct completions per 1000 runs, [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)), and is therefore explicitly rejected as a goal. Full design in §15. The same WAL doubles as the off-policy credit substrate the active controller learns from (§14).

**Vendor neutrality** is the other cross-cutting plane: filesystem-as-truth at the acceptance boundary, the normalized `Executor`, the `CapabilityProfile` handshake, and per-rollout RunManifest pinning of `{vendor, model, cli_version, capability_profile, prompt_hash}`. It is structural rather than bolted on, and — per the heterogeneous-fleet evidence (Devlo 70.2%, TRAE 70.4% on SWE-bench Verified, both using three distinct cross-vendor models plus a selector, [arXiv 2506.17208](https://arxiv.org/html/2506.17208v2)) — it is a *solve-rate asset* when paired with the L2 execution-grounded selector, not merely a portability claim. Without that selector, diverse-but-wrong candidates add noise (HeuriGym: higher diversity lowers yield); with it, `(vendor, model)` becomes a first-class diversity/search axis (§9, §14).

### 8.8 Why this layering, and what it deliberately refuses

The layering is the minimal structure that satisfies three constraints simultaneously: (1) it preserves every v1 invariant the verdicts call non-negotiable (L2 verbatim, cross-cutting determinism); (2) it admits each redesign idea only in the bounded form its adversarial verdict permits (L3 hints, never gates; controller below the Cardinal Contract); and (3) it makes the central failure mode — promoting an unverified candidate — a structural impossibility rather than a discipline that can erode.

It refuses, by construction, four tempting designs the evidence rejects: an amplifier with promotion power (violates the Cardinal Contract); a controller drawn straddling L2 (the "bypass" pitfall); static-CTDG-as-gate (PyCG ~70% recall silently drops fault-revealing tests); and bit-reproducible token-stream replay (impossible across hosted APIs). Each refusal is visible in the sketch: there is no arrow from any L3 box into the L2 `accepted` decision, and no path from the controller around the acceptance boundary. The honest framing — **amplifiers as bounded steering, execution evidence as both the steering signal and the brake, best-of-N as the floor we can never fall below** — is the architecture, not just its description.
