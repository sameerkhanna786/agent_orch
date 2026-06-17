## 23. Comparison Matrices, Glossary & Bibliography

This section is the plan's quick-reference appendix. It is deliberately dense and self-contained: a coding agent (on Codex or Claude Code) or a reviewer should be able to answer "where did mechanism X come from, what did we decide, and which evidence backs it" without re-reading Sections 1-22. Nothing here introduces a new disposition; every entry is traceable to the Fusion Ledger (see Section 18). Where a claim rests on evidence judged *unproven* in the adversarial review, it is flagged as guarded.

### 23.1 The Master Comparison Matrix (v1 vs Redesign vs SOTA-best vs APEX-Ω)

This is the canonical cross-axis comparison. APEX-Ω = "this plan." SOTA-best = the strongest published behavior on each axis as of mid-2026. The redesign column is the v3 *as proposed*, before the ledger pruned it.

| Axis | APEX v1 | Redesign (v3, as proposed) | SOTA-best | THIS PLAN (APEX-Ω) |
|---|---|---|---|---|
| Substrate / orchestration | Bespoke hard-coded `solver.py` pipeline | Prose blueprint; speculative-MCTS spine | Durable execution ([Temporal](https://temporal.io)/[DBOS](https://www.dbos.dev)); orchestration-as-code; [Conductor](https://opensource.microsoft.com/blog/2026/05/14/conductor-deterministic-orchestration-for-multi-agent-ai-workflows/) | Re-implementable deterministic engine (`agent/parallel/pipeline/phase/budget`) lifting v1 |
| Vendor-neutrality | Multi-backend portfolio (6 backends), diff-as-truth | Implicit; opaque-CLI tension unaddressed | Filesystem-as-truth; [ACP](https://agentclientprotocol.com/get-started/introduction)/[A2A](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) interop | Normalized Executor + ACP-style negotiation; **Codex/Claude/mixed in one run** |
| Parallelism | Barrier waves only (`parallel_workers=3`) | Speculative forks (uncapped) | parallel + pipeline streaming | Barrier `parallel()` + net-new `pipeline()` streaming |
| Search | FrontierSearch (PUCT) present but planner-driven | "Distributed MCTS" (re-describes FrontierSearch) | Adaptive-branching ([AB-MCTS](https://arxiv.org/abs/2503.04412)) > MCTS only at large budget | Bounded adaptive-branching over FrontierSearch; **best-of-N floor**; MCTS rejected |
| Pruning | Regression-prune (baseline-passers, chunks of 50) | Static CTDG as gate (unsafe) | Dynamic coverage ([testmon](https://testmon.org)) near-safe; full-suite backstop | Static prioritize + dynamic-coverage prune + backstop; **hint never gate** |
| Knowledge sharing | EpisodicMemoryBus (pull, negative sharing) | "Instant push" Blackboard 2.0 (raw, risky) | Abstracted negative + selective admission ([MEMOIR](https://arxiv.org/abs/2503.07826)/LTS, [bMAS](https://arxiv.org/abs/2507.01701)) | Phased, abstracted negative-constraint push at turn boundaries |
| Model economy | Single strong backend at `--effort max` | Heavy planner + thin executor (default) | Heavy/cheap split with competent editor + calibrated cascade ([Aider](https://aider.chat)) | Sub-role, verification-gated cascade; thin-executor-default rejected |
| Verification | Execution-authoritative cascade + NxN cross-val | Largely silent on verification cost | Hybrid execution + generative critic ([R2E-Gym](https://r2e-gym.github.io) ~43%->51%) | v1 cascade + hybrid critic (discrimination-only, under Cardinal Contract) |
| Determinism / resume | Best-effort + RunManifest; resume narrow (escrow WAL) | Bit-for-bit claim (overstated) | Journaled deterministic replay (durable execution) | Durable input-hash journal; replay **artifacts**; restart-survivable |
| Cost posture | "Never optimize for cost"; caps OFF; full-cap-16 | Targets generation cost; verification cost unaddressed | Difficulty-adaptive ([Snell](https://arxiv.org/abs/2408.03314) 2-4x); cache reuse (~90% off) | Adaptive low-K default + cascade + cache/test-impact; token-yield metric |
| Novelty | Hardened engineering substrate | Mechanisms mostly have v1 antecedents | Learned orchestrators on FIXED pool ([Puppeteer](https://arxiv.org/abs/2505.19591)) | **Open-pool cross-vendor controller via capability profiles** (held-out-vendor test) |

Reading guidance: the APEX-Ω column is never "best on every axis" by construction. On *cost posture* and *novelty* it intends to lead; on *search* and *pruning* it deliberately ships a more conservative mechanism than SOTA-best because the aggressive forms (classical MCTS-as-core, static-CTDG-as-gate) were judged unsound and rejected (see Section 18). The honest framing — search/economy as bounded amplifiers, execution evidence as both steering signal and brake, verified best-of-N as the floor — is what the matrix encodes.

### 23.2 The Five Paradigm Primitives (Quick Reference)

These are the engine verbs. Everything else in the plan is composed from them. They are vendor-neutral: each spawns or coordinates *workers* (Codex, Claude Code, or other), keeps state in script variables and the durable journal, and never relies on a conversation window.

| Primitive | One-line semantics | v1 antecedent | Ledger disposition |
|---|---|---|---|
| `agent(task, vendor, model)` | Spawn one isolated worker on a scoped task; return `{final_message, structured_output?, usage, diff}`. | `run_structured_prompt` | adopt |
| `parallel(tasks[])` | Fan out N workers, barrier-join all; results to disk. | `execute_rollout_requests` | adopt |
| `pipeline(items[], stages[])` | Per-item staged streaming: each item flows reproduce->localize->patch->verify without a wave barrier. | none (net-new) | adopt |
| `phase(name, body)` | Named, journaled checkpoint boundary; the only place where blackboard sharing, speculation admission, and resume points are allowed. | implicit in `_execute_with_dynamic_transitions` | adopt (formalized) |
| `budget(tokens, time, K)` | Scoped resource envelope; FrontierSearch and adaptive-K allocation run *inside* a budget. | `enable_adaptive_allocation` (OFF by default in v1) | adopt (default ON) |

Why `pipeline()` is the one genuinely net-new primitive: a barrier `parallel()` over the four-stage repair chain pays sum-of-slowest-per-stage in wall-clock; `pipeline()` pays slowest-single-chain, because a fast candidate can reach `verify` while a slow sibling is still in `localize`. This is the AB-MCTS/Agentless decomposition (localize -> reproduce -> patch -> verify -> rank) expressed as streaming rather than waves.

### 23.3 Glossary of Key Terms

| Term | Definition | Origin / where used |
|---|---|---|
| **agent / parallel / pipeline** | The three execution-shape primitives (single worker / barrier fan-out / per-item streaming). | Section 2, 23.2 |
| **Cardinal Safety Contract** | The invariant that execution evidence is authoritative for *selection*: soft signals (LLM critics, plan scores, blackboard hints) may re-rank within an execution-verified tier or downgrade a candidate, but may **never promote** an unverified candidate above a verified one, and may never prune a candidate before execution evidence exists. Counters inference-scaling false positives. | Section 13; ledger "adopt verbatim" |
| **CTDG (Code-Test Dependency Graph)** | A mapping from code symbols to the tests that exercise them, used to *prioritize/prune* test execution. APEX uses it as a hint layer only: static reordering (zero false-negative risk) + dynamic coverage prune (near-safe) + full-suite backstop. Static-as-gate is rejected. | Section 10 |
| **FrontierSearch** | APEX's bounded adaptive-branching controller (PUCT-flavored, budget-capped) that allocates effort across candidate branches and collapses to verified best-of-N below a feedback-confidence floor. *Not* classical distributed MCTS. | Section 9, 14 |
| **capability profile** | A learned vector describing a (vendor, model) worker's {skills, cost, latency, sandbox levels, schema support, internet, thinking, reliability}, used by the controller to route work to a pool it may not have trained on. Enables open-pool / held-out-vendor generalization. | Section 12, 14; from [MoMA](https://arxiv.org/html/2509.07571v1)/[DAAO](https://arxiv.org/html/2509.11079v1) |
| **token yield** | Useful work (resolved tasks) per token spent, *not* invoice. The metric that exposes the "almost-right trap": a cheap worker needing 3-4 retries can cost more than one frontier pass even though its per-call invoice is lower. | Section 16; from [xRouter](https://arxiv.org/html/2510.08439v1) |
| **verify-and-refute** | The convergence loop: independent attempts are produced, then *others try to refute them* (family-disjoint review, self-play tournament, VerificationAmplifier) until the model "stops saying done when it is half done." The load-bearing source of capability gain. | Section 1, 13 |
| **speculate()** | An agent-initiated fork admitted only at turn/checkpoint boundaries, feeding FrontierSearch ranking/budget; constant-factor cheaper via prefix reuse, bounded by virtual-loss / `min_branch_reward`. | Section 9 |
| **Epistemic Blackboard 2.0** | Phased, abstracted negative-constraint sharing at turn boundaries (keep relevance/confidence/dedup/own-rollout-exclusion). Raw "share-all / instant push" is rejected. The verifier must never see producer context. | Section 11 |
| **localization-futility gate** | An early gate that routes budget away from doomed hypotheses (the "15/16 doomed" waste) before the patch loop; informs allocation, never suppresses a candidate without execution evidence. | Section 13 |
| **token snowball / expensive failure** | SWE-Effi pathologies: off-track runs balloon 4x+ in tokens, and failures can cost more than successes. Root cause is missing futility/stop detection. | Section 16; from [SWE-Effi](https://arxiv.org/pdf/2509.09853) |
| **Normalized Executor** | The single worker interface (`spawn -> run(scoped_task) -> {result, usage, session_id, diff}`) with one adapter per vendor and ACP-style capability negotiation + graceful degradation. | Section 3 |

### 23.4 Cross-Reference Table: Mechanism -> Origin -> Disposition

Each mechanism's disposition is copied verbatim from the Fusion Ledger; no new dispositions are introduced. "Origin" names the SOTA system or v1 component the mechanism mirrors. Dispositions: **adopt** (keep, build), **adopt-modified** (keep the winning part, bound the rest), **defer** (staged for later), **reject** (do not build).

| Mechanism | Origin / antecedent | Disposition | One-line rationale |
|---|---|---|---|
| Vendor-neutral engine (agent/parallel/pipeline/phase/budget) | v1 `run_structured_prompt` + `execute_rollout_requests` | adopt | Re-implementable, not rebuilt. |
| `pipeline()` per-item staged streaming | net-new (Agentless decomposition) | adopt | Slowest-chain not sum-of-stages. |
| Cardinal Safety Contract | v1 selection rule | adopt | Best-of-N -> trustworthy gains. |
| Cheap-first verification cascade (never synthesizes a pass) | v1 cascade | adopt | Safe per-candidate prune. |
| Per-rollout git-worktree isolation + fcntl locks | v1 | adopt | Makes any parallelism safe (CAID 63.3 vs 57.2). |
| Determinism + RunManifest + Docker digest pinning | v1 + durable-execution practice | adopt | Sound replay; reproduce artifacts. |
| Anti-cheat / fairness / failure taxonomy / abstention | v1 | adopt | Reward-hacking scales with capability ([ImpossibleBench](https://arxiv.org/abs/2510.20270)). |
| Two-tier failure memory + self-evicting BackendPortfolio | v1 | adopt | One vendor's 429 must not poison the fleet. |
| Durable input-hash journaled resume | v1 ReplayRecorder + escrow WAL (unused) | adopt | "Do better than reference"; off-policy credit substrate. |
| Normalized Executor + ACP-style negotiation | v1 per-vendor fragments; [ACP](https://agentclientprotocol.com/get-started/introduction) | adopt | Degrade-not-crash. |
| (vendor, model) as a diversity/search axis | [Devlo/TRAE](https://arxiv.org/html/2506.17208v2) | adopt | Cross-vendor decorrelates hallucinations. |
| Difficulty-adaptive low-K allocation (default ON) | v1 `enable_adaptive_allocation` (OFF); [Snell](https://arxiv.org/abs/2408.03314) | adopt | Optimal K often <10; biggest cost lever. |
| Early localization-futility gate | v1 + [SWE-Effi](https://arxiv.org/pdf/2509.09853) | adopt | Kills "15/16 doomed" before the patch loop. |
| Hybrid verifier (execution + swappable generative critic) | [R2E-Gym](https://r2e-gym.github.io) ~43%->51% | adopt | Critic breaks ties only within verified tier. |
| Prefix-stable prompt assembly + provider-cache adapter | provider caching | adopt | ~90% off cached reads; portable cost contract. |
| Bounded adaptive-branching over FrontierSearch | [AB-MCTS](https://arxiv.org/abs/2503.04412) | adopt-modified | Keep adaptive allocation; cap it; best-of-N floor. |
| Agent-initiated `speculate()` fork | redesign | adopt-modified | Turn-boundary only; prefix-reuse cheap; bounded. |
| CTDG as prioritizer + dynamic-coverage prune + backstop | [testmon](https://testmon.org)/[PyCG](https://github.com/vitsalis/PyCG) | adopt-modified | Reorder is safe; static-as-gate rejected. |
| Cheap pre-execution plan scoring (downgrade-only) | redesign | adopt-modified | Sets priors/budget; never excludes pre-execution. |
| Blackboard 2.0 (phased abstracted negatives) | [MEMOIR](https://arxiv.org/abs/2503.07826)/LTS; v1 EpisodicMemoryBus | adopt-modified | Abstracted negatives preserve diversity. |
| Model economy as verification-gated cascade | [Aider](https://aider.chat) architect/editor; [HyperAgent](https://arxiv.org/abs/2409.16299) | adopt-modified | Cheapen run/verify/narrow edits; frontier on navigation. |
| Open-pool active controller via capability profiles | [Puppeteer](https://arxiv.org/abs/2505.19591)/[MoMA](https://arxiv.org/html/2509.07571v1) | adopt-modified | The defensible contribution; staged; fail-open. |
| GEPA-style reflective prompt evolution of controller | [GEPA](https://arxiv.org/abs/2507.19457) | defer | Stage 1; 35x cheaper than RL; vendor-agnostic. |
| Full RL (Puppeteer/Conductor GRPO) over orchestrator | [Puppeteer](https://arxiv.org/abs/2505.19591) | defer | Stage 2; only when volume justifies. |
| Distributed/classical MCTS as the core loop | redesign | reject | Re-describes FrontierSearch; brittle vs container state. |
| Static-AST CTDG as a test-pruning gate | [PyCG](https://github.com/vitsalis/PyCG) ~70% recall | reject | Silently drops fault-revealing tests. |
| Cheap pre-execution plan scoring as a hard gate | redesign | reject | False-negative pruning violates Cardinal Contract. |
| Raw share-all / mid-subprocess injection blackboard | redesign | reject | -3.7pp accuracy; homogenizes; breaks replay. |
| Heavy-orchestrator + thin-executor as the default | [HyperAgent](https://arxiv.org/abs/2409.16299) ablation | reject | Cheapening navigation/multi-file editing hurts most. |
| Default full-cap 16 redundant trajectories (caps OFF) | v1 default | reject | The headline cost pathology. |
| Bit-reproducible agent OUTPUT replay | redesign | reject | Impossible across hosted APIs ([Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)). |

### 23.5 Per-Vendor Capability / Flag Cheat-Sheet

The Normalized Executor (Section 3) maps each native CLI onto the common contract `spawn(cwd) -> run(scoped_task) -> {final_message, structured_output?, usage, session_id, raw_events}` plus `observe_diff()`. Treat JSON event streams as telemetry; correctness is verified on the git diff. Pin and record the resolved CLI version per rollout (npm CLIs move fast; e.g., Codex profile semantics broke at 0.134.0).

| Capability | Codex CLI | Claude Code | Gemini CLI | opencode |
|---|---|---|---|---|
| Headless entrypoint | `codex exec` (alias `e`) | `claude -p / --print` | `gemini -p / --prompt` | `opencode run "…"` |
| JSON result mode | `--json` (JSONL events) | `--output-format json` | `--output-format json` | OpenAPI `serve` / `serve acp` |
| NDJSON streaming | `--json` event stream | `--output-format stream-json` | `--output-format stream-json` | `serve acp` (stdio NDJSON) |
| Structured output | `--output-schema <file>` | `--json-schema` -> `structured_output` | none native (embed+post-parse) | via server schema |
| Sandbox control | `--sandbox {read-only\|workspace-write\|danger-full-access}` (read-only default) | `--permission-mode {acceptEdits\|dontAsk\|…}` + `--allowedTools` | `--yolo` (all-or-nothing) | server perms |
| Model select | `--model/-m`, config.toml/profiles | `--model` | `--model` | per-config |
| MCP tool plane | required-MCP config | `--mcp-config` | built-in/config | server / ACP passthrough |
| Resume | `codex exec resume --last\|<ID>` | `session_id` reuse | `--session-summary <file>` | server session |
| Reproducible/CI | `--skip-git-repo-check`, `--ephemeral`, `--ignore-user-config` | `--bare` (skips auto-discovery; future `-p` default) | non-TTY trigger | `run --attach` (no MCP cold-start) |
| Auth | CLI login or inline `CODEX_API_KEY` (exec-only) | `ANTHROPIC_API_KEY`/apiKeyHelper (from 2026-06-15 `-p` draws separate Agent-SDK credit) | CLI auth | `OPENCODE_SERVER_PASSWORD` |
| Retry signal | exit codes / stderr | `system/api_retry` (rate_limit/overloaded/server_error) | exit codes / stderr | server errors |

Graceful-degradation rules the Executor applies: no native schema -> embed schema in prompt + post-parse; no read-only sandbox -> wrap in APEX worktree + fcntl lock (the isolation floor); no bidirectional streaming -> fall back to single-shot. One NDJSON parser with per-vendor event maps covers Codex/Claude/Gemini; opencode normalizes via its OpenAPI server. Sources: [Codex noninteractive](https://developers.openai.com/codex/noninteractive), [Claude headless](https://code.claude.com/docs/en/headless), [Gemini headless](https://geminicli.com/docs/cli/headless/), [opencode CLI/server/ACP](https://opencode.ai/docs/cli/).

### 23.6 Bibliography of Load-Bearing Citations

Each entry is tied to the specific claim it supports in this plan. This is not an exhaustive reading list; it is the set of sources whose removal would weaken a specific design decision.

| # | Source | Supports (specific claim) |
|---|---|---|
| 1 | [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787) | Coverage scales log-linearly with samples (SWE-bench Lite 15.9%->56%) but only with a *verifier*; pure best-of-N with weak selection plateaus. Motivates verify-and-refute + the best-of-N floor (Section 1, 13). |
| 2 | [AB-MCTS: Adaptive Branching Tree Search](https://arxiv.org/abs/2503.04412) | Adaptive width-vs-depth allocation beats fixed MCTS *only at larger budgets*. Basis for bounded adaptive-branching; justifies rejecting MCTS-as-core (Section 9). |
| 3 | [R2E-Gym](https://r2e-gym.github.io) | Hybrid execution + generative critic lifts ~43%->51%; execution anchors, critic breaks ties. Basis for the discrimination-only hybrid verifier (Section 13). |
| 4 | [testmon (test-impact analysis)](https://testmon.org) / [PyCG (static Python call graph)](https://github.com/vitsalis/PyCG) | Dynamic coverage is near-safe for pruning; static call graphs ~70% recall and miss reflection/monkeypatch/fixtures. Basis for CTDG-as-hint, static-as-gate rejected (Section 10). |
| 5 | [Aider architect/editor + leaderboard](https://aider.chat) | A heavy "architect" + cheap competent "editor" is 5-14x cheaper at maintained quality. Basis for verification-gated model-economy cascade (Section 12). |
| 6 | [Puppeteer / Multi-Agent Collaboration via Evolving Orchestration (NeurIPS 2025)](https://arxiv.org/abs/2505.19591) | Learned RL controller that prunes/sequences a *fixed* pool is published prior art -> a baseline we must beat, not our headline; cost-aware reward shape borrowed. Open-pool generalization is the unclaimed gap (Section 14, 19). |
| 7 | [GEPA: Reflective Prompt Evolution](https://arxiv.org/abs/2507.19457) | Reflective prompt evolution ~35x cheaper than RL and prompts-not-weights = vendor-agnostic. Basis for deferring controller evolution to Stage 1 (Section 14). |
| 8 | [MEMOIR / Lifelong/Long-Term-Sharing memory](https://arxiv.org/abs/2503.07826) | Abstracted negatives preserve diversity; share-all loses accuracy. Basis for Blackboard 2.0's abstracted-negative push (Section 11). |
| 9 | [SWE-Effi](https://arxiv.org/pdf/2509.09853) | Token/time-budget AUC metrics; token-snowball + expensive-failure pathologies from missing futility detection. Basis for token-yield metric + futility gate (Section 16). |
| 10 | [ImpossibleBench](https://arxiv.org/abs/2510.20270) | Reward-hacking scales with capability; agents exploit broken/satisfiable-by-cheating tests. Basis for anti-cheat + execution-authoritative selection (Section 13). |
| 11 | [Thinking Machines: Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | Temp-0 is not bitwise reproducible across hosted APIs (batch non-invariance). Basis for rejecting output-replay; reproduce artifacts (Section 15). |
| 12 | [Dissecting the SWE-Bench Leaderboards (arXiv 2506.17208)](https://arxiv.org/html/2506.17208v2) | No single architecture wins; localization + verification + ranking dominate; top systems ensemble multiple vendors (Devlo 70.2%, TRAE 70.4%). Basis for (vendor,model) diversity axis (Section 3, 13). |
| 13 | [Why SWE-bench Verified no longer measures frontier capabilities (OpenAI, Feb 2026)](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) + [SWE-bench Pro](https://arxiv.org/abs/2509.16941) | Verified is contaminated (59.4% flawed hard tests); use Pro/Live under standardized scaffold. Basis for the evaluation plan's benchmark choice (Section 20). |
| 14 | [Agent Client Protocol (ACP)](https://agentclientprotocol.com/get-started/introduction) / [A2A (Linux Foundation)](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) | `initialize` capability negotiation + Agent Cards are the portable templates for the Executor handshake / capability profile (Section 3). |
| 15 | [Live-SWE-agent (arXiv 2511.13646)](https://arxiv.org/abs/2511.13646) / [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) | Runtime tool synthesis from a minimal bash agent (~$0.73/issue, +22.6% for strong models). Basis for the minimal-worker-plus-orchestration shape (Section 8). |
| 16 | [xRouter (arXiv 2510.08439)](https://arxiv.org/html/2510.08439v1) / [RouteLLM](https://www.morphllm.com/llm-router) | Static routing trees are brittle and don't transfer; measure token yield not invoice; prefer cascade-with-verification. Basis for cascade-over-route (Section 12, 16). |
| 17 | [MoMA](https://arxiv.org/html/2509.07571v1) / [DAAO](https://arxiv.org/html/2509.11079v1) | Capability/cost profiles (not one-hot identity) enable routing to a growing heterogeneous pool. Basis for open-pool capability profiles (Section 14). |

Guarded claims (honor the adversarial posture): the open-pool cross-vendor controller "beats the single best model in the pool at matched cost on SWE-bench Pro" is **plausible but unproven** — it is framed as the experiment to run (Section 20), not a result. Classical MCTS, static-CTDG-as-gate, share-all blackboard, thin-executor-default, full-cap-16, and output-replay are **rejected** and must not be reintroduced. Where a mechanism is `defer`, it ships behind a flag with a fail-open path to the prior stage.
