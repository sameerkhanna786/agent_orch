## 7. Design Thesis & Principles

This section states the one idea that organizes the whole plan, defends *why* that idea expands capability beyond the base model, names the three composed properties that do the work (in priority order), explains why the Cardinal Contract makes the system trustworthy where pure LLM-judge pipelines are not, and restates the eight design principles as binding commitments — each tied to a concrete v1 mechanism or an adversarial verdict so a builder can implement against it on **either Codex or Claude Code**. It is the contract the rest of the plan (Sections 8–22) is held to.

### 7.1 The Central Unifying Idea

**Search and economy are bounded amplifiers; execution evidence is both the steering signal and the brake; verified best-of-N is the floor we can never do worse than.**

That sentence is the load-bearing claim of APEX-Ω. Unpack it into three commitments:

1. **Bounded amplifiers.** Every "bolder" mechanism the redesign proposed — speculative branching, CTDG pruning, the epistemic blackboard, the model economy, the active controller — is admitted only as an *amplifier* of an already-trustworthy base, run *inside* explicit budget caps, and required to *collapse to the base* when its own confidence signal is thin. None of them is a new source of truth. This is the direct consequence of the adversarial record: distributed/classical MCTS is `unsound` for repo-SWE at fixed budget, static-AST CTDG-as-gate is `unsound`, and cheap pre-execution plan scoring as a hard prune is `reject`ed. What survives (`adopt-modified`) is the *allocation* core of those ideas, never their *gating* power.

2. **Execution evidence as steering AND brake.** The same execution signal that *guides* search (which branch to widen, which to deepen, which to stop) is the *only* signal allowed to *gate* acceptance. A soft signal may reorder or downweight; it may never promote an unverified candidate, and it may never suppress a candidate before execution evidence exists. Steering is permissive (hints, priors, ordering); braking is conservative (execution-authoritative, downgrade-only). Conflating the two is the failure mode this plan exists to prevent.

3. **Best-of-N as the floor.** Below an explicit feedback-confidence threshold, every amplifier disables itself and the engine runs v1's proven verified best-of-N (the wave loop in `_execute_progressive_rollout_plan`). The floor is not a fallback we hope never to hit; it is the reason the downside is *bounded to v1's already-strong behavior*. Search only ever adds upside.

This framing is deliberately the opposite of "more agents = more capability." Both the source article and the SOTA digest refute agent-count as the unlock. The article's own words: "the genuine unlock is the verify-and-refute loop, NOT agent count." The digest is sharper: against an imperfect verifier the compute-optimal sample count is *finite and often `< 10`* ([Limits of Inference Scaling Through Resampling](https://arxiv.org/abs/2411.17501)), and "no amount of inference scaling of weaker models can match the single-sample accuracy of a sufficiently strong model." CodeMonkeys quantifies it for SWE exactly: `69.8%` coverage collapses to `57.4%` after selection ([arXiv:2501.14723](https://arxiv.org/abs/2501.14723)) — the fan-out buys coverage; the *selector* buys realized capability. APEX-Ω therefore invests disproportionately in the verifier/selector (Section 13) and keeps default sample counts low (Section 14), and it treats every "we ran N more workers" instinct as a cost pathology to be justified, not a capability lever to be assumed.

### 7.2 Why Orchestration-as-Code + Verify-and-Refute Exceeds the Base Model

A single base-model call is bounded three ways. The engine removes each bound with a *named* mechanism, and the removal is *necessary plumbing* — not the capability itself. The capability comes from the verify-and-refute/selection loop *running on* that plumbing. Stating this split precisely matters: the adversarial panel judged the unqualified claim "the dynamic-workflow engine is what lets APEX exceed the base model" `sound_with_caveats` precisely because attributing the gain to the *architecture* rather than to *execution-grounded selection running on it* is a category error.

| Base-model bound | What causes it (evidence) | APEX-Ω mechanism that removes it | Why removal is sound, not magic |
|---|---|---|---|
| **Context bound** | Context rot across 18 frontier models *below* the window limit ([Chroma](https://research.trychroma.com/context-rot)); lost-in-the-middle 15–20pt drop ([Liu et al.](https://arxiv.org/abs/2307.03172)) | **Context isolation**: state in script variables + a durable journal, never a conversation window; workers get scoped context only (v1: `RepoContext`-once read-only, `EpisodicMemoryBus`, per-rollout worktrees) | Independently corroborated phenomenon; v1 already converged on it. A 500-node run does not drift because the orchestrator never accumulates a transcript. |
| **Single-trajectory bound** | One sample explores one approach; RLHF mode-collapse makes naive resampling redundant ([Kirk et al.](https://arxiv.org/abs/2310.06452)) | **Diversity-preserving branching + cross-vendor fleets**: `(vendor, model)` is a first-class search axis; bounded adaptive branching (Section 9) | Cross-vendor diversity decorrelates hallucinations (Devlo/TRAE; Multi-LLM AB-MCTS, [Sakana](https://sakana.ai/ab-mcts/)); but kept *bounded* — AB-MCTS only beats sampling above ~64 calls ([arXiv:2503.04412](https://arxiv.org/abs/2503.04412)), so we collapse to best-of-N at small budgets. |
| **Self-verification bound** | Models cannot honestly grade themselves; reward-hacking *scales* with capability (GPT-5 exploits tests 76% on impossible-SWEbench, [ImpossibleBench](https://arxiv.org/html/2510.20270v1)) | **Execution-authoritative selection + adversarial refutation** (Section 13): family-disjoint tool-call review, self-play tournament, VerificationAmplifier, all under the Cardinal Contract | This is the *actual* capability multiplier. Du et al. ([arXiv:2305.14325](https://arxiv.org/abs/2305.14325)) show refutation reduces hallucination; the model "stops saying done when it is half done." |

The honest scope, carried forward from the adversarial verdicts: the multi-agent +90.2% / "token usage explains 80% of variance" numbers are **research-domain** results, and Anthropic itself warns coding is *less* parallelizable than research. Those numbers are **not** load-bearing for APEX-Ω's design and are not cited as justification; we rely on coding-specific evidence (CodeMonkeys, S*, AlphaCodium, ImpossibleBench, the resampling-limits paper). Likewise the Bun port and "85 agents in 16 min" are marketing/partly-unverified and appear nowhere in the justification chain.

### 7.3 The Three Composed Properties, In Priority Order

When two of these conflict, the higher-priority one wins. This ordering is a design rule, not a ranking of importance-for-marketing.

#### 7.3.1 Property 1 (highest priority): Execution-grounded verify-and-refute

The unlock is the loop where independent attempts are produced and *others try to refute them until convergence*. v1 already ships three forms — family-disjoint independent-CLI tool-call review, the self-play tournament (`K` patches × `M` independently-generated tests), and the `VerificationAmplifier` (discriminating-test generation to break ties) — and gates all of them on the Cardinal Safety Contract. This is what converts log-linear *coverage* (repeated sampling: SWE-bench Lite `15.9% → 56%`, [Large Language Monkeys](https://arxiv.org/abs/2407.21787)) into *trustworthy resolved issues*, where pure best-of-N with weak selection plateaus far below the coverage ceiling. **If forced to keep only one property, keep this one** — it is the difference between APEX and an inference-scaling demo.

#### 7.3.2 Property 2: Context isolation

Intermediate results live in **script variables and a durable journal, never a conversation window**. This is the *scaling* unlock: a large run does not drift because the orchestrator holds no transcript. v1 converged on this via worktree-per-rollout isolation, `RepoContext`-once, and results-to-disk; APEX-Ω generalizes it into the engine (Section 8) so it is a property of *every* `agent()` call, not just the rollout pipeline. It is second because isolation without verify-and-refute scales an untrustworthy answer; verify-and-refute without isolation is trustworthy but does not scale.

#### 7.3.3 Property 3: Orchestration-as-code with durable, deterministic replay

Lifting v1's hard-coded `solver.py` pipeline (`_execute_with_dynamic_transitions`, `_execute_progressive_rollout_plan`) into a re-implementable program exposing `agent / parallel / pipeline / phase / budget` makes the orchestration auditable, resumable across full restart (the explicit "do better than the reference impl" mandate), and — critically — a tractable substrate for a *learned* controller (reproducible off-policy credit assignment over journaled decisions). It is third because it is *enabling plumbing*: necessary for the other two to scale and for the controller to be learnable, but not itself the source of capability. The adversarial verdict is explicit that crediting the engine (rather than the selection running on it) with exceeding the base model is overclaiming.

A guard that rides on Property 3 and protects Property 1: if a model emits orchestration control flow, **freeze-then-journal it** — snapshot + hash the emitted script into the `RunManifest` so replay runs over a *frozen* script. Live model output must never be un-journaled control flow, because non-determinism in the orchestration layer is fatal to replay ([Temporal: "non-determinism is fatal"](https://docs.temporal.io/workflows)).

### 7.4 Why the Cardinal Contract Makes the Whole Thing Trustworthy

A pure LLM-judge pipeline is untrustworthy for a measurable reason: pointwise LLM judges capture only ~21% of best-of-N gain, and against an imperfect verifier resampling gains *saturate or invert* because wrong code that passes the verifier gets shipped — the "Inference Scaling FLaws" failure mode. Reward-hacking makes this strictly worse as models get stronger ([ImpossibleBench](https://arxiv.org/html/2510.20270v1): GPT-5 hacks 76%). For a *mixed-vendor* fleet the adversary surface is wider still.

The **Cardinal Safety Contract** is the single rule that closes this hole, and APEX-Ω keeps it *verbatim*:

> Execution evidence is authoritative. Soft signals may re-rank *within* an execution-verified tier, or downgrade an already-accepted candidate — they may **never** promote an unverified candidate.

It is enforced *structurally*, not by convention, and the structure is what makes it auditable:

```
# Acceptance gate (v1 _apply_evidence_bound_review): monotone in one direction only.
def apply_evidence_bound_review(candidate, soft_reviews):
    # `accepted` starts from EXECUTION evidence (verifier decision), never a score threshold.
    # Soft reviews (deterministic critic, EG-critic, LLM voters, perspective/final-acceptance
    # reviewers, process-quality, evidence-ledger) can ONLY flip True -> False.
    for review in soft_reviews:
        if candidate.accepted and review.verdict == "refute":
            candidate.accepted = False          # downgrade allowed
        # there is NO branch that sets candidate.accepted = True from a soft signal.
    return candidate

# Deterministic ranking key (v1): lexicographic tuple; every soft/learned/LLM key sits
# STRICTLY BELOW every execution+critic key, terminating in a content hash, never insertion order.
def ranking_key(c):
    return (
        c.combined_score,          # execution-derived
        c.accepted,                # execution gate
        c.public_signal_score,     # execution-derived
        c.critic_score,            # deterministic critic
        c.size, c.verification_score,
        c.eg_critic_tiebreak,      # learned  -- below ALL execution keys
        c.perspective_score,       # LLM      -- below learned
        len(c.changed_files),
        -c.cluster_id,             # deterministic final tiebreak (NOT slot/insertion order)
    )
```

Two corollaries the contract forces, both kept:

- **The verifier cascade never synthesizes a pass** (Section 13): silent `rc == 0` no-op → `errors = 1` (never `passed = 1`); timeout `rc == 124` → a separate `regression_inconclusive` axis (`+0.15` partial), never a failure; singletons abstain (empty cross-validation list) rather than getting a synthetic `0.5` prior.
- **Abstention is a first-class outcome** (a `Status` peer of `SOLVED`): with no positive execution evidence the engine returns `None`/abstains rather than shipping its least-bad guess — the direct mitigation of the false-positive cost the inference-scaling literature proves is fatal.

The contract is what lets every *amplifier* be admitted safely: branching, pruning, plan-scoring, the blackboard, and the model economy are all expressed as operations that **steer** (reorder/prioritize/downweight) but are *structurally incapable* of **braking incorrectly** (promoting unverified, or suppressing pre-execution). That is why this plan can be aggressive about search and economy while remaining honest about correctness.

### 7.5 The Eight Design Principles, Restated as Commitments

Each principle is a non-negotiable, tied to the v1 mechanism that implements it or the verdict that bounds it. A coding agent building APEX-Ω on Codex or Claude Code must treat these as invariants; removing any one collapses a specific guarantee.

| # | Principle | Concrete v1 mechanism / verdict it rests on | What breaks if removed |
|---|---|---|---|
| 1 | **Carry every v1 invariant verbatim** | Cardinal Contract (`_apply_evidence_bound_review`, ranking tuple); cheap-first cascade that never synthesizes a pass; per-rollout `fcntl`-locked worktree isolation; anti-cheat/fairness (upstream harness is the only published number, scrub-at-load, true history flatten, NDFF); failure taxonomy; first-class abstention | The credibility argument; each invariant guards a specific false-SOLVED or false-MISS path |
| 2 | **Orchestration-as-code with state-in-variables** | Lift `_execute_with_dynamic_transitions` / `_execute_progressive_rollout_plan` into `agent/parallel/pipeline/phase/budget`; **freeze-then-journal** any model-emitted control flow into the `RunManifest` | Replay soundness; a stochastic model authoring un-journaled control flow makes runs non-reproducible (Temporal: non-determinism is fatal) |
| 3 | **Vendor neutrality via filesystem-as-source-of-truth + capability negotiation** | Acceptance decided on the git diff (vendor-blind); one normalized Executor + ACP-style `initialize` handshake; degrade-not-crash (no native schema → embed + post-parse; no read-only sandbox → APEX worktree+lock) | Heterogeneous fleets; cross-vendor comparison becomes apples-to-oranges and verify-and-refute loses its ground truth |
| 4 | **Evidence-grounded, bounded speculation** | Branching/pruning/plan-scoring may only reorder/prioritize/downweight; all search inside FrontierSearch caps (`max_depth`, `max_frontier_branching`, virtual-loss, `min_branch_reward`); **collapse to verified best-of-N below a feedback-confidence floor** | The floor guarantee; without it speculation can do *worse* than v1 (verdict: distributed MCTS `unsound`; static CTDG-gate `unsound`; pre-exec plan-gate `reject`) |
| 5 | **Diversity-preserving knowledge sharing** | Evolve `EpisodicMemoryBus` delivery: share only abstracted *negative/avoidance* constraints + execution facts at turn boundaries; keep relevance ranking, confidence floors, dedup-by-signature, own-rollout exclusion; first exploratory burst stays independent; verifier never sees producer context | Diversity; share-all measurably lowers accuracy (`-3.7pp`) and homogenizes attempts; mid-subprocess injection is infeasible against opaque CLIs and breaks replay (verdict: share-all/instant-push `reject`) |
| 6 | **Determinism + DURABLE resumable journaling** | Best-effort determinism (temp `0.0`, content-hash ordering, deterministic snapshot SHAs, atomic writes) + per-`agent()`-call WAL keyed by input hash: unchanged calls replay cached results, only edited/new re-run, **surviving full restart**; reproduce *artifacts* (diffs + verification), not token streams | The explicit "do better than the reference impl" mandate; v1's `ReplayRecorder` has no production callsite and the escrow WAL is narrow, so this is genuinely unbuilt (verdict: bit-reproducible OUTPUT replay `reject` — impossible across hosted APIs) |
| 7 | **Fail-loud, never-fake** | Strict acceptance gate (verifier decision, legacy `overall_score >= 0.9` shortcut removed); salvage ≠ success (`ABSTAINED`); `HeuristicRepairAgent` apply-test-revert never submits an unverified mutation; progress-based liveness (S1–S7 watchdog), never a flat wall-clock kill of a working agent; fail-open instrumentation can only *delay* a kill, never accelerate or fake one | The human/upstream-harness as final gate; placeholder/mock data behind try/catch or salvage-as-success would leak into published outcomes |
| 8 | **Cost-aware allocation as first-class control** | `budget{}` is a first-class primitive (defaulted unbounded to honor v1's "never optimize for cost" stance, opt-in to bound) but exhaustion **never aborts an in-flight succeeding rollout**; default difficulty-adaptive low-K; cascade (cheap → verify-on-diff → escalate to frontier); measure **token yield** (cost per verified-resolved task), not invoice | The single biggest cost lever (verdict: default full-cap 16 redundant trajectories `reject`); v1's `enable_adaptive_allocation` exists but is OFF by default |

Two cross-cutting notes on the principles, because they are where well-intentioned changes go wrong:

- **Principles 4 and 8 are the same discipline viewed twice.** Bounded speculation (P4) is what makes cost-aware allocation (P8) safe: the controller may shift budget toward surviving hypotheses (the early localization-futility gate of Section 14), but that is *allocation*, never *suppression*. A candidate is never excluded pre-execution; the budget simply flows to where the value gradient is steepest. This keeps the inverse-Cardinal-Contract violation (false-negative pruning) off the table.
- **Principle 5's hardest constraint is the easiest to forget:** the verifier/selector must *never* see the producers' epistemic context. If a refuter inherits the producer's blackboard, refutation degenerates into collective delusion (the multi-agent false-consensus cascade: [Spark to Fire](https://arxiv.org/abs/2603.04474), LangGraph hub injection = 100% failure). Independence is enforced by family-disjointness *and* context isolation, not just by spawning more skeptics.

### 7.6 Vendor Neutrality and Cost-Awareness as First-Class, Not Afterthoughts

These two are listed last because they are constraints that other designs treat as taxes; APEX-Ω converts them into wins, and that conversion is part of the thesis.

**Vendor neutrality is structural.** The mechanism is v1's: correctness is decided on the **git diff** via the execution cascade, so the orchestrator is vendor-blind at the verification boundary and any leaf worker is swappable. APEX-Ω formalizes this into a single normalized Executor interface (detailed in Section 3; restated here only to make the thesis self-contained):

```
# Vendor-neutral leaf-worker contract. Same shape on Codex, Claude Code, or mixed.
interface Executor:
    def initialize() -> CapabilityProfile          # ACP-style handshake: schema?, sandbox?, internet?, effort?
    def spawn(worktree_cwd, vendor, model, version) -> Session
    Session.run(scoped_task) -> StructuredResult {  # validated object if schema negotiated, else final text
        final_message, structured_output?, usage, session_id, raw_events
    }
    Session.observe_diff() -> GitDiff               # THE contract; vendor JSON streams are telemetry only
```

Heterogeneity is a **strength, not a compromise** on two independent axes:
- **Quality:** `(vendor, model)` is a first-class *diversity* axis; cross-vendor pools decorrelate hallucinations and a heterogeneous fleet + execution-grounded selector beats single-vendor best-of-N (Devlo/TRAE). One search tree can hold a Claude branch, a Codex branch, and a cheap Codex/opencode contract-executor leaf simultaneously.
- **Cost:** cross-vendor price spreads (15–60×) power the model economy (Section 12) — heavy orchestrator one vendor, cheap executors another — run as a *cascade* (cheapest capable first, escalate to frontier only on verify-on-diff failure), never brittle up-front routing.

Capability differences **degrade, never crash**: no native schema → embed schema + post-parse; no read-only sandbox → wrap in APEX's own worktree + `fcntl` isolation. The `RunManifest` pins `{vendor, model, cli_version, capability_profile, prompt_hash}` per branch; replay reproduces *artifacts* because true token-stream determinism is impossible across hosted APIs. v1's two-tier failure memory (transient call-failover vs. structural backend reroute) plus the self-evicting `BackendPortfolio` keep a `429` on one vendor from poisoning a healthy fleet.

**Cost-awareness is a control input, not a reporting afterthought.** `budget{}` is first-class and supports loop-until-budget, but it honors two v1 invariants absolutely: (a) defaulted unbounded (so an operator must *opt in* to optimize for cost, preserving v1's "never optimize for cost" default for headline runs), and (b) budget exhaustion **never aborts an in-flight succeeding rollout** — caps fire only when no successful patch exists, exactly as v1 does. The metric that matters is **token yield = cost per verified-resolved task**, reported on a cost-matched Pareto frontier — which is also the bar reviewers now enforce for any cost claim (Section 20). The default regime is difficulty-adaptive low-K (often `< 10`), not the full-cap-16 pathology.

### 7.7 What This Thesis Commits the Rest of the Plan To

So that later sections can be checked against this one, the thesis makes these falsifiable commitments:

1. Every amplifier in Sections 9–14 ships with (a) an explicit budget cap, (b) a defined collapse-to-best-of-N condition, and (c) a structural guarantee it cannot promote unverified or suppress pre-execution candidates. (See Section 18, the Fusion Ledger, for the kept/modified/dropped accounting.)
2. The selector/verifier (Section 13) receives disproportionate engineering relative to the search tree, because the digest repeatedly shows the selector — not the topology — caps every method.
3. Durable, restart-survivable, per-`agent()`-call journaled resume (Section 15) is the headline *systems* contribution and is treated as genuinely unbuilt in v1, validated by the "kill mid-run → identical resumed winner" test.
4. The defensible *scientific* contribution (Section 19) is the open-pool, vendor-agnostic active controller (Section 14) evaluated cost-matched on contamination-resistant benchmarks with a held-out-vendor split — **not** "we added MCTS" (published prior art) and **not** the individual amplifiers (most have v1 antecedents).
5. No design decision is justified by vendor marketing (Bun, "85 agents/16 min") or by cross-domain research numbers presented as if they were coding results.

In one line: **APEX-Ω is the hardened, execution-authoritative v1 kernel, re-platformed onto a vendor-neutral orchestration-as-code engine, with the redesign's bolder ideas admitted only as bounded amplifiers that steer with execution evidence and can never do worse than verified best-of-N.**
