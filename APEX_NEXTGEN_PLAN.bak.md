# APEX-Ω: A Vendor-Neutral Dynamic-Workflow Orchestrator for Evidence-Grounded Agentic Coding (Master Plan & Design Spine)

*A synthesis plan fusing APEX v1, the state of the art, and a proposed redesign — built on the vendor-agnostic dynamic-workflow (ultracode) paradigm so it runs on Codex, Claude Code, or both. Every adopted mechanism was checked against adversarial verdicts and published evidence.*

## Table of Contents

- [1. Executive Summary & Thesis](#1-executive-summary-thesis)
- [2. The Orchestration Substrate: Dynamic Workflows as APEX’s Foundation](#2-the-orchestration-substrate-dynamic-workflows-as-apexs-foundation)
- [3. Vendor-Agnostic Execution: Codex, Claude Code, or Both](#3-vendor-agnostic-execution-codex-claude-code-or-both)
- [4. APEX v1: Foundation, Strengths & Ceiling](#4-apex-v1-foundation-strengths-ceiling)
- [5. The Proposed Redesign (v3), Critically Assessed](#5-the-proposed-redesign-v3-critically-assessed)
- [6. State of the Art: Synthesis & Exploitable Gaps](#6-state-of-the-art-synthesis-exploitable-gaps)
- [7. Design Thesis & Principles](#7-design-thesis-principles)
- [8. Target Architecture Overview](#8-target-architecture-overview)
- [9. The Speculative Tree-Search Layer (as Workflow Patterns)](#9-the-speculative-tree-search-layer-as-workflow-patterns)
- [10. CTDG + Bidirectional Pruning](#10-ctdg-bidirectional-pruning)
- [11. The Epistemic Blackboard 2.0](#11-the-epistemic-blackboard-20)
- [12. The Vendor-Agnostic Model Economy](#12-the-vendor-agnostic-model-economy)
- [13. Verification & Evidence-Grounded Selection (Verify-and-Refute)](#13-verification-evidence-grounded-selection-verify-and-refute)
- [14. The Active Adaptive Controller & Learned Search Policy](#14-the-active-adaptive-controller-learned-search-policy)
- [15. Isolation, Determinism & Durable Resumable Runs](#15-isolation-determinism-durable-resumable-runs)
- [16. Speed & Cost Engineering](#16-speed-cost-engineering)
- [17. Self-Improvement & Memory](#17-self-improvement-memory)
- [18. Fusion Ledger: Kept / Modified / Dropped](#18-fusion-ledger-kept-modified-dropped)
- [19. Novelty & Scientific Contributions](#19-novelty-scientific-contributions)
- [20. Evaluation Plan & Experiment Matrix](#20-evaluation-plan-experiment-matrix)
- [21. Risk Register & Mitigations](#21-risk-register-mitigations)
- [22. Implementation Roadmap](#22-implementation-roadmap)
- [23. Comparison Matrices, Glossary & Bibliography](#23-comparison-matrices-glossary-bibliography)

## Executive Summary

**APEX-Ω is a vendor-neutral dynamic-workflow ENGINE.** It re-implements the orchestration-as-code paradigm (the "ultracode" model: a deterministic script spawns isolated subagent *workers*, keeps state in script variables, fans work out, and converges via verify-and-refute) as a re-implementable engine whose leaf workers are *any* coding CLI/API — Codex (`codex_cli`), Claude Code (`claude_cli`), Gemini, opencode, or **mixed in one run**. The hardened **APEX v1 substrate** is preserved verbatim as the inner kernel; the proposed-redesign ("v3") ideas are admitted only as **workflow patterns over that substrate**, each judged against the adversarial verdicts and accepted, modified, deferred, or rejected — never rubber-stamped.

**The fused thesis.** The binding constraint in repo-level SWE is not generation coverage but *where you spend coverage and how you select among it against an imperfect verifier*. v1 already solved selection (the **Cardinal Safety Contract**: execution evidence is authoritative; soft/LLM signals may only re-rank within a verified tier or downgrade, never promote). The genuine capability unlock is the **verify-and-refute loop running on a durable, vendor-neutral substrate**, not agent count and not a fancier search algorithm. So APEX-Ω's spine is: (1) a deterministic workflow engine exposing `agent/parallel/pipeline/phase/budget` over a normalized **Executor** interface; (2) v1's execution-authoritative verification kernel; (3) **bounded, evidence-grounded adaptive-branching** allocation (wider-vs-deeper conditioned on remaining budget) that *degrades to verified best-of-N as a guaranteed floor*; (4) a **vendor-agnostic active controller** that routes each decision node to a worker described by a learned *capability/cost profile* — the defensible scientific contribution.

**Honest disposition of the redesign (traceable to verdicts).** *Distributed MCTS beats best-of-N* — **rejected** as a headline (verdict: unsound; it largely re-describes v1's existing `FrontierSearchController` and plain MCTS does not reliably beat verified sampling at repo scale). *Static CTDG safe pruning* — **rejected as a gate** (unsound; PyCG ~70% recall, pytest collection not statically enumerable), **adopted as a test-prioritizer** over a one-time dynamic-coverage base with a full-suite backstop. *Cheap pre-execution plan scoring* — **adopted-modified** as a downgrade-only branch prioritizer, never a kill switch. *Cross-branch sharing* — **adopted-modified** as phased, abstracted, *negative-constraint* sharing (raw share-all measurably loses). *Heavy-orchestrator/thin-executor split* — **adopted-modified** as a sub-role, verification-gated cascade (cheapen run/verify and narrow edits; keep frontier on navigation/multi-file edits per the HyperAgent ablation). *Speculative branching* — **adopted-modified** (a large *constant* factor via prefix reuse, not "exponential"; forking only at turn/checkpoint boundaries into FrontierSearch's budget caps).

**The speed story.** Wall-clock and cost both fall versus v1's non-adaptive default (`num_rollouts=5`, no down-scaling, escalating to the `max_rollouts=16` cap) without lowering the solve ceiling: difficulty-adaptive low-K allocation (compute-optimal K is often <10), a net-new `pipeline()` per-item streaming primitive (wall-clock = slowest single chain, not sum-of-slowest-per-stage), an early **localization-futility gate** that kills the "15/16 doomed" waste before the patch loop, **prefix-stable prompt assembly** so provider KV caches fire across forks (~90% off cached reads), test-impact-pruned verification that bounds the N² cross-validation matrix, warm CoW worktree pools, and futility/token-snowball detection — all while keeping v1's progress-based liveness (never a flat wall-clock kill of a working agent).

**The novelty story.** Not "we added MCTS" (Puppeteer/AB-MCTS/AFlow are published) and not the individual mechanisms (most have v1 antecedents). The unclaimed, NeurIPS-grade gap is **open-pool cross-vendor search-policy generalization**: a learned controller that, via learned capability/cost *profile vectors* (not one-hot vendor IDs), routes to a vendor/model held out from training with no retraining, while every accept stays execution-gated. The central falsifiable hypothesis (H1): on contamination-resistant SWE (SWE-bench Pro / Live), under cost-matched budgets, the open-pool controller dominates the strongest single model, cost-equal best-of-N, and a re-trained published orchestrator — *and retains dominance on a held-out-vendor split*. Secondary (H2): execution-authoritative grounding is a *learnability requirement*, not a safety tax (relaxing the Cardinal Contract should degrade the learned policy via reward-hacked signal).

**The vendor story (Codex / Claude Code / both).** Vendor neutrality is structural because **the git diff is the source of truth**: acceptance runs on the resulting diff regardless of which vendor produced it, so JSON event streams are telemetry, not the contract. One normalized Executor (`spawn → run(ScopedTask) → {final_message, structured_output?, usage, session_id} + observe(git diff)`) has an adapter per backend, plus an **ACP-style capability-negotiation handshake** (probe internet/schema/sandbox/effort; degrade gracefully — no native schema → embed-in-prompt + post-parse; no read-only sandbox → wrap in APEX's worktree+`fcntl` isolation). Because the controller routes per decision node, a single solve can have a Claude branch, a Codex branch, and a cheap Codex/opencode contract-executor leaf simultaneously — the heterogeneous-fleet pattern (Devlo/TRAE) that *beats* single-vendor best-of-N. The RunManifest pins `{vendor, model, resolved cli_version, capability_profile, prompt_hash}` per node; replay reproduces **artifacts** (re-applied diffs + re-run verification), since temp-0 is not bitwise reproducible across hosted APIs.

**Net.** APEX-Ω achieves the redesign's stated goals — faster, cheaper, more novel — by spending *less* (adaptive allocation, cascades, cache/test-impact reuse) and *selecting better* (the v1 contract, a hybrid verifier), on a durable resumable substrate that runs on Codex, Claude Code, or both, with a publishable open-pool controller riding on top.

## 1. Executive Summary & Thesis

### 1.1 What APEX-Ω Is, In One Sentence

**APEX-Ω is a vendor-neutral, deterministic dynamic-workflow ENGINE — orchestration-as-code that spawns isolated coding-agent *workers* (Codex, Claude Code, or both in one run), holds all intermediate state in script variables and a durable journal rather than a conversation window, and converges through execution-grounded verify-and-refute — on which APEX v1's execution-authoritative kernel is the hardened SUBSTRATE and every redesign idea is admitted only as a *judged, bounded workflow-pattern extension*.**

This is a fusion of three things the rest of this plan develops in detail: (a) the orchestration-as-code paradigm from the [Claude Code dynamic-workflows](https://code.claude.com/docs/en/workflows) model — the model writes the script, a background runtime runs it, state lives in variables — re-implemented as a *re-implementable, vendor-owned* engine (Section 2); (b) the production-hardened APEX v1 pipeline (worktree isolation, the Cardinal Safety Contract, cheap-first verification, RunManifest determinism), preserved verbatim as the inner kernel (Section 4); and (c) the proposed "v3" redesign ideas (speculative tree search, CTDG pruning, epistemic blackboard, model economy, active controller), each routed through an adversarial verdict and accepted, modified, deferred, or rejected — never rubber-stamped (Sections 5, 9-14, 18).

APEX-Ω must build a **better** engine than the reference implementation in exactly one place the reference is documented to be weak — durable, restart-survivable resume — while honoring five invariants throughout: filesystem-as-source-of-truth, execution-evidence-authoritative selection, fail-loud-never-fake, durable resumable journaling, and vendor neutrality.

### 1.2 The Fused Thesis

The binding constraint in repository-level software engineering is **not** generation coverage — it is *where you spend coverage and how you select among it against an imperfect verifier.* The inference-scaling literature makes this precise: repeated sampling expands *coverage* (pass@k) along a smooth log-linear curve over four orders of magnitude ([Brown et al., "Large Language Monkeys"](https://arxiv.org/abs/2407.21787): SWE-bench Lite 15.9% → 56% at 250 samples), but **realized** resolved-issue rate is gated by selection. With a weak verifier, selection plateaus near ~100 samples while coverage keeps climbing — a 50+ point gap on MATH — and against an imperfect verifier the compute-optimal sample count is *finite and often single-digit* ([Limits of Inference Scaling Through Resampling](https://arxiv.org/abs/2411.17501): optimal K often < 10; weak-model amplification cannot match a strong model's single shot). CodeMonkeys quantifies the same effect on real SWE: [69.8% coverage but only 57.4% realized after selection](https://arxiv.org/abs/2501.14723), with random selection scoring 45.8%.

APEX v1 already solved selection. Its **Cardinal Safety Contract** (`APEX_DESIGN_BLUEPRINT.md` §13.1) states: *execution evidence is authoritative; soft/LLM signals may only re-rank within an execution-verified tier or downgrade an already-accepted candidate, never promote an unverified one.* It is enforced structurally — `_apply_evidence_bound_review` flips `accepted` only `True → False`, and the deterministic ranking tuple places every soft/learned/LLM key strictly below every execution key. This is the direct mitigation of the false-positive failure mode the resampling literature proves is fatal, and it is what makes APEX's number trustworthy where a pure LLM-judge pipeline is not.

So the genuine capability unlock is **execution-grounded verify-and-refute running on a durable, vendor-neutral substrate with context isolation** — NOT agent count and NOT a fancier search algorithm. The source paradigm says this in its own words ("the genuine unlock is the verify-and-refute loop, not agent count"; "the model stops saying done when it is half done"), and it is independently corroborated: [Anthropic's multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) beat single-agent Opus 4 by 90.2% with orchestrator-worker + verification, while [Du et al. (ICML 2024)](https://arxiv.org/abs/2305.14325) show agents refuting each other reduce hallucination. Context isolation is the *scaling* complement — keeping intermediate results in variables, not context, is a real architectural fix against measured [context rot](https://www.trychroma.com/research/context-rot) (all 18 frontier models degrade before the window fills) and [lost-in-the-middle](https://arxiv.org/abs/2307.03172) (15-20pt drop by position alone).

> **Honest credit assignment (non-negotiable):** The engine by itself does not exceed the base model. What exceeds it is the *verifier + selector + verify-and-refute loop running on the engine*, plus diversity-preserving branching and cross-vendor fleets that decorrelate hallucinations, plus the removal of the single-context bound via isolation. Search and the model economy are **bounded amplifiers**; execution evidence is both the steering signal and the brake; verified best-of-N is the floor we can never do worse than. That framing is itself the load-bearing claim of this plan.

The APEX-Ω spine therefore has four layers, in descending order of importance:

1. A deterministic workflow engine exposing `agent / parallel / pipeline / phase / budget` over a normalized **Executor** interface (Section 2).
2. v1's execution-authoritative verification kernel, kept verbatim (Sections 4, 13).
3. **Bounded, evidence-grounded adaptive-branching** allocation (wider-vs-deeper conditioned on remaining budget) that *degrades to verified best-of-N as a guaranteed floor* (Sections 9, 14).
4. A **vendor-agnostic active controller** that routes each decision node to a worker described by a learned *capability/cost profile* — the defensible scientific contribution (Sections 14, 19).

### 1.3 Honest Disposition of Each Redesign Idea

The redesign document (`APEX_DESIGN.md`) is aspirational prose; most of its "novel" mechanisms have substantial, unacknowledged v1 antecedents. Each idea is admitted only at its verdict-bound strength. The full ledger is in Section 18; the headline dispositions:

| Redesign idea | Verdict | One-line reason |
|---|---|---|
| Distributed/classical MCTS as the core loop | **reject** | Re-describes v1's existing `FrontierSearchController`; plain MCTS does not reliably beat verified sampling at repo scale and is brittle against non-serializable container state. |
| Static-AST CTDG as a test-pruning **gate** | **reject** | PyCG ~70% recall; pytest collection not statically enumerable; gating silently drops fault-revealing tests, violating execution-authority. |
| Cheap pre-execution plan scoring as a hard **prune/gate** | **reject** | False-negative pruning suppresses correct-but-unverified plans before evidence exists — inverse violation of the Cardinal Contract. |
| Heavy-orchestrator + thin executor as the **default** SWE shape | **reject** | [HyperAgent ablation] shows cheapening navigation/multi-file editing causes the worst resolve-rate drops; the almost-right trap can cost more than one frontier pass. |
| Non-adaptive fixed-K default (`num_rollouts=5`, no down-scaling) + caps OFF | **reject** | The headline cost pathology; replaced by adaptive low-K + budget-aware deepening, full-cap kept only as the thin-feedback floor. |
| Bit-reproducible agent OUTPUT replay | **reject** | Impossible across hosted APIs (temp-0 batch non-invariance); reproduce **artifacts** (diffs + re-run verification) instead. |
| Raw share-all / mid-subprocess "instant push" blackboard | **reject** | Share-all measurably lowers accuracy (−3.7pp) and homogenizes attempts; mid-subprocess injection is infeasible against opaque CLIs and breaks replay. |
| Bounded adaptive-branching (wider/deeper/diversify) over FrontierSearch | **adopt-modified** | Keep the part of search that wins (AB-MCTS adaptive allocation) inside budget caps; mandatory collapse to verified best-of-N below a feedback-confidence floor. |
| Agent-initiated `speculate()` fork | **adopt-modified** | Admit only at turn/checkpoint boundaries feeding FrontierSearch ranking/budget; a *constant-factor* prefix-reuse saving, not exponential. |
| CTDG as test **prioritizer** + dynamic-coverage prune + full-suite backstop | **adopt-modified** | Reordering has zero false-negative risk; dynamic coverage is near-safe; static-as-gate stays rejected. |
| Cheap pre-execution plan scoring as a **downgrade-only** prioritizer | **adopt-modified** | May set branch priors/budget the controller can override; never excludes a candidate pre-execution. |
| Blackboard 2.0: phased, abstracted **negative-constraint** sharing at turn boundaries | **adopt-modified** | Abstracted negatives preserve diversity ([MEMOIR/LTS]); verifier must not see producer context. |
| Model economy as sub-role, **verification-gated cascade** | **adopt-modified** | Cheapen run/verify and narrow edits; keep frontier on navigation/multi-file edits; escalate on first verify-on-diff failure with a rewrite cap. |
| Open-pool active controller via learned capability/cost profiles | **adopt-modified** | The defensible NeurIPS-grade contribution; staged (bandit → GEPA → RL); blend-not-switch, fail-open to heuristic. |
| `pipeline()` per-item staged streaming | **adopt** | The one genuinely net-new primitive; cuts wall-clock from sum-of-slowest-per-stage to slowest-single-chain. |
| GEPA reflective prompt evolution / Full RL over orchestrator | **defer** | Stage 1 / Stage 2 respectively, after the bandit ships. |

### 1.4 The Speed Story: Spend Less, Don't Lower the Ceiling

v1's defaults are tuned for "SOTA, never for cost": `enable_adaptive_allocation=False` runs a fixed `num_rollouts` (default 5, never down-scaled) regardless of difficulty, with the `max_rollouts=16` cap reached only under escalation/adaptive allocation, `repo_token_cap=None`, no wall-clock kill. APEX-Ω drives wall-clock and cost down *without lowering the solve ceiling* through composed levers, every one of which respects progress-based liveness (never a flat wall-clock kill of a working agent — v1's S1-S7 watchdog discipline is preserved):

- **Difficulty-adaptive low-K allocation (default ON).** Compute-optimal K is often < 10 ([Limits of Resampling](https://arxiv.org/abs/2411.17501); CodeMonkeys uses 10 trajectories). v1 already *has* `enable_adaptive_allocation` and `compute_rollout_count` over buckets `[1,4,8,16]` — they are simply OFF. Flipping them on is the single biggest cost lever (Section 16).
- **`pipeline()` per-item streaming.** No barrier between reproduce → localize → patch → verify stages; item A can be patching while item B is still reproducing. Wall-clock collapses from sum-of-slowest-per-stage to slowest-single-chain (Section 2, 16). This is the largest net-new build — v1 has only barrier waves.
- **Early localization-futility gate.** Kills the "15/16 doomed" patch-loop waste by routing budget to surviving hypotheses *before* the expensive patch attempts; informs allocation, never suppresses a candidate without execution evidence (Section 16.6).
- **Prefix-stable prompt assembly + provider-cache adapter.** Makes branching's constant-factor saving real (~90% off cached reads) and gives a portable cost contract across opaque vendor CLIs (Section 12, 16).
- **CTDG/test-impact-pruned verification** to bound the O(N²) cross-validation matrix (v1's dominant per-task verification cost), with a full-suite backstop (Section 10, 16).
- **Futility / token-snowball detection** and warm CoW worktree pools (v1's `WorktreePool`, ~10× cheaper recycling).

### 1.5 The Novelty Story: Open-Pool Cross-Vendor Control

The marketable headline — "an RL controller that prunes/sequences agents for cost-quality" — is **already published** ([Puppeteer, NeurIPS 2025](https://arxiv.org/abs/2505.19591); AgentConductor; AFlow; AOrchestra). Re-proposing it is derivative; it must be a *baseline we beat*, not the contribution. Most published learned orchestrators (Puppeteer, AgentConductor) train and test on a **fixed, known** agent/model pool indexed by identity; even recent dynamic-creation work (AOrchestra) and routing-as-attributes methods stop short of open-pool cross-vendor generalization to a *held-out vendor with no retraining*.

The unclaimed, NeurIPS-grade gap is **open-pool cross-vendor search-policy generalization**: a learned controller that, via learned **capability/cost profile vectors** (not one-hot vendor IDs — extending [MoMA](https://arxiv.org/html/2509.07571v1)/[DAAO](https://arxiv.org/html/2509.11079v1) from routing to long-horizon SWE), routes to a vendor/model *held out from training, with no retraining*, while every accept stays execution-gated.

- **H1 (central, falsifiable):** On contamination-resistant SWE ([SWE-bench Pro](https://labs.scale.com/leaderboard/swe_bench_pro_public) / SWE-bench-Live), under cost-matched budgets, the open-pool controller dominates (a) the strongest single model in the pool, (b) cost-equal best-of-N, and (c) a re-trained published orchestrator — *and retains dominance on a held-out-vendor split.*
- **H2 (secondary):** Execution-authoritative grounding is a *learnability requirement*, not a safety tax — relaxing the Cardinal Contract should *degrade* the learned policy via reward-hacked signal.

The "make the systems work pay rent algorithmically" bridge: durable deterministic replay (the Section 15 substrate) enables reproducible off-policy credit assignment over journaled decisions — turning a systems feature into a learning enabler, which is what clears a top ML venue. Reviewers will also demand the now-expected emergent-structure and "when orchestration hurts" analyses ([ChromaFlow](https://arxiv.org/abs/2605.14102)); both are in the evaluation plan (Section 20).

### 1.6 The Vendor Story: Codex, Claude Code, or Both, In One Run

Vendor neutrality is **structural, not bolted on**, because **the git diff is the source of truth.** Acceptance runs on the resulting diff regardless of which vendor produced the patch, so JSON event streams are *telemetry, not the contract.* v1 already proves this: `LLMBackend` spans `claude_cli / codex_cli / gemini_cli / opencode_cli / metacode_cli / openai_api`, and verification is vendor-blind at the diff boundary.

APEX-Ω consolidates v1's scattered per-vendor fragments behind one normalized **Executor**:

```
Executor:
  spawn(config) -> session
  run(session, ScopedTask) -> { final_message, structured_output?, usage, session_id }
  observe(session) -> git_diff           # filesystem-as-truth, vendor-blind
```

plus an **ACP-style capability-negotiation handshake** that probes internet / schema / sandbox / effort and **degrades — does not crash** (no native JSON schema → embed-in-prompt + post-parse, as v1 already does for gemini/codex; no read-only sandbox → wrap in APEX's worktree + `fcntl` isolation). Because the controller routes *per decision node*, a single solve can simultaneously carry a Claude branch, a Codex branch, and a cheap Codex/opencode contract-executor leaf — the heterogeneous-fleet pattern (Devlo/TRAE) that *beats* single-vendor best-of-N and powers cross-vendor cost arbitrage. The **RunManifest** pins `{vendor, model, resolved cli_version, capability_profile, prompt_hash}` per node; replay reproduces **artifacts** (re-applied diffs + re-run verification), because temp-0 is not bitwise reproducible across hosted APIs (Section 3, 15).

### 1.7 Net Pareto Claim

APEX-Ω targets the redesign's stated goals — faster, cheaper, more novel — by **spending less** (difficulty-adaptive low-K, verification-gated cascades, prefix-cache and test-impact reuse, the localization-futility gate) and **selecting better** (the Cardinal Safety Contract, a hybrid execution+critic verifier per [R2E-Gym ~43% → 51%]), on a **durable, restart-survivable** substrate that runs on Codex, Claude Code, or both in one run, with a publishable open-pool controller riding on top.

The defensible quantitative claim, to be proven in Section 20, is a **Pareto improvement**: same-or-higher resolved-issue rate on contamination-resistant SWE at a *fraction* of v1's tokens and wall-clock — never worse than verified best-of-N (the floor), and dominant on the held-out-vendor split where no published baseline competes. Where evidence is thinnest — beating the *single best frontier model* at matched cost on hard repo SWE is unproven in the literature — the plan treats it as the experiment to run, not a settled result, and ships every learned component default-off / fail-open so enabling it can never silently move the headline number.

> **Two cautions carried throughout this plan.** (1) Passing tests ≠ correctness — the [Bun port](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code) shipped 99.8% tests passing yet [13,000+ unsafe blocks with no human full read](https://www.theregister.com/devops/2026/05/14/anthropics-bun-rust-rewrite-merged-at-speed-of-ai/); we cite the Bun 750k-line / 85-agent figures as motivation only, never as load-bearing evidence. (2) Coding is *less* parallelizable than research (Anthropic's own caveat), so APEX-Ω **auto-routes** rather than auto-orchestrating everything: small, tightly-coupled changes go to a single focused agent; only decomposable, verification-heavy work escalates to a full workflow.

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

Two gaps are the headline work. (1) **No orchestration-as-code layer:** the model never writes a script; the dependency graph, fan-out, and loops are hard-coded as `_execute_with_dynamic_transitions` + `_execute_progressive_rollout_plan` (~8.2k-line `solver.py`). (2) **Resume is the explicit "do better than the reference impl" mandate:** the reference Claude Code engine resumes only *within a session* — "if you exit Claude Code while a workflow is running, the next session starts the workflow fresh" ([Claude Code workflows docs](https://code.claude.com/docs/en/workflows)) — and v1 is at the same limitation. v1’s `ReplayRecorder` has **no production callsite** (and `ReplayPlayer` is wired only into the offline `apex replay` CLI, not the solve/orchestrator path), and the escrow WAL (CCEDF) is an fsync-durable backstop that rescues only **one** confirmed-full-scope-pass candidate across restart. Durable restart-survivable resume is therefore **genuinely unbuilt in v1**, not merely under-documented — the clearest "APEX must do better" target.

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

## 3. Vendor-Agnostic Execution: Codex, Claude Code, or Both

APEX-Ω runs with Codex (`codex_cli`), Claude Code (`claude_cli`), or **both in one run** — and ideally any agent CLI/API (`gemini_cli`, `opencode_cli`, `openai_api`). This is a hard requirement, not a nicety. The dynamic-workflow paradigm (see Section 2) is a *concept* — orchestration-as-code spawning subagent workers — and the Claude Code Workflow tool is only one implementation of it. APEX-Ω owns a vendor-neutral orchestration engine; vendors are merely the **leaf workers** that the engine fans out, pipelines, and refutes against each other. This section specifies the normalized Executor interface, the per-vendor adapters with concrete flags, the ACP-style capability-negotiation handshake, the heterogeneous-fleet routing model, the cost-arbitrage cascade, and the run-manifest pinning that makes cross-vendor runs replayable.

The load-bearing claim — and the reason this is feasible today — is that **the filesystem (and the git diff over it) is the source of truth.** APEX v1 already verifies acceptance on the resulting diff regardless of which vendor produced it (`LLMBackend` already spans `claude_cli/codex_cli/gemini_cli/opencode_cli/metacode_cli/openai_api`). The adversarial verdict on this claim is `sound_with_caveats` (high confidence): vendor neutrality is genuinely feasible, but "without losing the paradigm benefits" is an overclaim. The honest framing is **vendor-neutral with bounded, declared degradation**, never lossless portability. We build to that truth.

### 3.1 The Core Invariant: Filesystem/Git-Diff as Contract, JSON Events as Telemetry

Every vendor's JSON event vocabulary differs — Codex emits `item.completed` with item types (agent message, reasoning, command exec, file change, MCP tool call, web search, plan update); Claude emits `tool_use` inside `stream-json` NDJSON; Gemini emits a `stats` object (per-model tokens, per-tool `totalCalls/Success/Fail`, files +/-). But **all of them mutate the working tree**, and APEX verifies on the resulting git diff. This is independently confirmed by both the v1 paradigm ingest ("filesystem/git diff as the source of truth — vendor-neutrality enabler") and the vendor SOTA research ([FILESYSTEM-AS-TRUTH is the real enabler... Treat JSON streams as telemetry, not as the contract](https://developers.openai.com/codex/noninteractive)).

The consequences are precise and non-negotiable:

- **Executor parsing stays best-effort / observational.** Correctness *never* depends on trusting a vendor's self-reported output. This directly honors the pitfall "Do not trust vendor self-reported JSON as the correctness contract." A vendor that reports `success: true` but produced a diff that fails verification is a failure, full stop.
- **Structured output (when present) is a convenience, re-validated downstream.** Claude's native `--json-schema` → `structured_output` and Codex's `--output-schema` are accepted, but APEX re-parses and re-validates the returned object at the engine layer (schema validation at the tool layer, model retries on mismatch — exactly v1's `run_structured_prompt` contract). For vendors with no native schema (Gemini, opencode), APEX embeds the schema in the prompt and post-parses (§3.4).
- **Replay reproduces artifacts, not token streams.** Because temperature-0 is not bitwise reproducible across hosted APIs ([batch non-invariance, Thinking Machines, Sep 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/): 80 distinct completions per 1000 identical temp-0 prompts), APEX replays recorded diffs and re-runs verification (§3.7). This is already aligned with v1's diff-verification; the accepted mechanism "Bit-reproducible agent OUTPUT replay" is `reject`, and we honor that.

This invariant is what makes the harness-dominates-model threat survivable. [Terminal-Bench 2.0](https://www.tbench.ai/leaderboard/terminal-bench/2.0) shows 30–50pt same-model swings across harnesses; a normalization-leaky executor can erase the entire cross-vendor diversity gain. Because correctness lives in the diff, a parsing bug degrades *telemetry*, not *verdicts* — but flag/sandbox/schema mapping (§3.3) is still load-bearing and gets a conformance test (§3.8).

### 3.2 The Normalized Executor Interface

The Executor is the canonical generalization of v1's `CLIModelClient.run_structured_prompt` (cli_backend.py) — the existing multi-vendor `agent()` primitive that already returns a normalized `CLIModelResult` and **never raises to the caller** (every abnormal exit becomes a typed result). The accepted mechanism "Normalized Executor + ACP-style capability negotiation" is `adopt`: consolidate v1's scattered per-vendor fragments into one interface; degrade-not-crash.

```python
# Data structures (engine-internal; field types are normative)

@dataclass(frozen=True)
class CapabilityProfile:
    vendor: str                       # "codex_cli" | "claude_cli" | "gemini_cli" | "opencode_cli" | "openai_api"
    model: str                        # human alias, e.g. "opus" | "gpt-5.5" — resolved to launcher id at command-build time
    cli_version: str                  # npm-resolved exact version, e.g. "@openai/codex@0.140.2"
    internet: bool                    # web-search/internet mode available
    native_schema: bool               # native structured-output (Claude --json-schema, Codex --output-schema)
    sandbox_levels: tuple[str, ...]   # e.g. ("read-only","workspace-write","danger-full-access") | ("yolo",)
    thinking: str                     # "effort:low..max" | "extended" | "none"
    bidirectional_stream: bool        # only Claude --input-format stream-json documented
    tool_interception: str            # "pre-tool-hook" | "none" — and known_interception_gaps
    mcp: bool                         # accepts an injected MCP server set

@dataclass(frozen=True)
class ScopedTask:
    prompt: str
    schema: dict | None               # JSON schema for structured_output, if any
    allowed_tools: tuple[str, ...]    # restricted tool allowlist for this worker
    sandbox: str                      # requested sandbox level (normalized to APEX floor on degradation)
    effort: str                       # "low".."max"
    mcp_servers: tuple[McpServerRef, ...]
    cwd: str                          # the worktree path (isolation is APEX-owned; see §3.4)
    label: str; phase: str            # narration (maps to phase()/log())

@dataclass(frozen=True)
class ExecResult:
    final_message: str
    structured_output: dict | None    # re-validated by engine, NOT trusted as correctness
    usage: Usage                      # input/cached_input/output/reasoning tokens, normalized cross-vendor
    session_id: str | None            # for vendor-native resume (codex exec resume; claude session_id)
    raw_events: list[dict]            # best-effort parsed NDJSON; telemetry only
    finalization_status: str          # "completed"|"timeout"|"policy_violation"|"output_limit"|"progress_abort"|"isolation_error"
    fs_diff: GitDiff                  # observe(): the authoritative artifact

class Executor(Protocol):
    def negotiate(self) -> CapabilityProfile: ...        # ACP-style initialize; §3.5
    def spawn(self, cwd: str) -> "Session": ...          # bind to a worktree
class Session(Protocol):
    def run(self, task: ScopedTask) -> ExecResult: ...   # == agent(); returns structured result + observes diff
    def observe(self) -> GitDiff: ...                    # git diff over the worktree (the contract)
    def resume(self, session_id: str) -> "Session": ...  # vendor-native resume where available; else replay (§3.7)
```

The Executor lifecycle is exactly: `spawn(worktree_cwd) -> session.run(ScopedTask) -> {final_message, structured_output?, usage, session_id, raw_events} + observe(git diff)`. This is `agent()` in the paradigm sense, now first-class multi-vendor. Three of four vendors expose NDJSON event streams (Codex `--json`, Claude `--output-format stream-json`, Gemini `--output-format stream-json`), so **one NDJSON reader with per-vendor event-type maps covers three of four**; opencode normalizes via its OpenAPI server or `serve acp`.

Reuse mandate (accepted mechanism "Reuse v1's cli_backend.py/llm_routing.py/backend_portfolio.py/cli_turn_parser.py"): the Executor wraps, not replaces, v1's machinery. `cli_turn_parser.py` (`CLITurnParser`) remains the NDJSON/turn splitter feeding `raw_events` and the `turn_observer` mid-flight steering channel. The S1–S7 progress watchdog (cli_backend.py) governs liveness — **progress-based, never wall-clock** (Section 15) — so a long legitimate agentic turn on any vendor is not false-killed.

### 3.3 Per-Vendor Adapters (Concrete Flags)

Each adapter maps the common `ScopedTask` to vendor-native argv. These flags are the literal contract a coding agent builds against; pin and record the resolved CLI version (§3.7) because npm-distributed CLIs drift fast (e.g., Codex profile semantics broke at 0.134.0; `--full-auto` deprecated).

| Capability | Codex (`codex exec`) | Claude Code (`claude -p`) | Gemini CLI (`gemini -p`) | opencode |
|---|---|---|---|---|
| Headless entry | `codex exec` (alias `e`) | `-p`/`--print` | `-p`/`--prompt` (or non-TTY) | `opencode run` / `serve acp` |
| Event stream | `--json` (JSONL: thread/turn/item/error) | `--output-format stream-json` (NDJSON: system/init, api_retry, stream_event) | `--output-format stream-json` | OpenAPI 3.1 server `/doc`; `serve acp` NDJSON over stdio |
| JSON result | (via `--json` + `-o`) | `--output-format json` (result, session_id, total_cost_usd) | `--output-format json` (response + stats + error) | server response body |
| Structured output | `--output-schema <file>` | `--json-schema` → `structured_output` | none native → embed in prompt + post-parse | none native → embed + post-parse |
| Sandbox | `--sandbox {read-only\|workspace-write\|danger-full-access}` (read-only default) | `--permission-mode {acceptEdits\|dontAsk\|...}` + `--allowedTools` | `--yolo` (all-or-nothing) | server perms |
| Tool allowlist | (config/required-MCP) | `--allowedTools` | (limited) | server config |
| MCP | required-MCP (config) | `--mcp-config` | built-in/config | ACP passes MCP at session start |
| Final message file | `-o`/`--output-last-message` | (in JSON result) | `--session-summary <file>` | response |
| Git-repo bypass | `--skip-git-repo-check` | (n/a) | (n/a) | (n/a) |
| Reproducibility flags | `--ephemeral`, `--ignore-user-config`, `--ignore-rules` | `--bare` (skip hooks/skills/MCP/CLAUDE.md auto-discovery; becoming `-p` default) | (limited) | `--attach http://host:port` (avoid MCP cold-start) |
| Resume | `codex exec resume --last\|<SESSION_ID>` | `session_id` replay (+ `--replay-user-messages`) | (limited) | server session |
| Auth | CLI login or inline `CODEX_API_KEY` (exec-only) | `ANTHROPIC_API_KEY`/`apiKeyHelper` (with `--bare`) | provider auth | `OPENCODE_SERVER_PASSWORD`/basic-auth |

Canonical launch templates (the adapter emits these, modulo negotiated degradation):

- **Codex:** `codex exec --json --sandbox workspace-write --skip-git-repo-check --output-schema <f> -m <model>`
- **Claude:** `claude -p --bare --output-format stream-json --json-schema <f> --allowedTools <…> --permission-mode acceptEdits --mcp-config <f> --model <model>`
- **Gemini:** `gemini -p --output-format stream-json --yolo --session-summary <f>` (schema embedded in prompt)
- **opencode:** `opencode run --attach http://host:port "<prompt>"` or `opencode serve acp` (ACP over stdio)
- **openai_api:** in-process adapter over v1's `LLMClient` + `AgentLoop` fallback path (OpenAI-compatible chat.completions), kept for completeness; not the primary path.

Note on `--bare` and reproducible CI: from 2026-06-15, Claude subscription `-p`/Agent-SDK usage draws a **separate monthly Agent SDK credit pool**. This breaks naive cross-vendor cost accounting (§3.6) and must be modeled before any savings number is published. Codex's inline `CODEX_API_KEY` is exec-only and unsafe as a job-level env var on repo-controlled code; the adapter passes it per-invocation, never exports it.

### 3.4 Graceful Degradation to APEX's Own Floor

When a vendor lacks a capability, the Executor **degrades to APEX's own primitives** rather than crashing. The two load-bearing degradations:

1. **No native schema (Gemini, opencode) → embed schema in prompt + post-parse.** This is exactly v1's `_augment_prompt_for_backend` behavior (Codex additionally normalizes `additionalProperties=False`, required=all keys via `_normalize_schema_for_codex`). Prompt-embedded schema is *weaker* than native validation — this is a declared, bounded loss, not a hidden one. The engine re-validates and retries (up to v1's `max_attempts=4` for claude/codex; 1 otherwise).
2. **No read-only sandbox (Gemini `--yolo` is all-or-nothing) → wrap in APEX worktree + `fcntl` isolation.** APEX's per-rollout git-worktree isolation + advisory `fcntl` lock (the accepted, kept-verbatim mechanism; CAID ablation 63.3 vs 57.2) is the *floor*. Even a vendor running "full access" is confined to its own worktree, so concurrent same-file edits cannot corrupt siblings. The 4-tier degradation ladder (seed_clone → worktree → snapshot → synthetic) is preserved from v1.

Other declared per-vendor losses (the abstraction degrades, it does not preserve losslessly):

- **Tool-call interception is non-uniform.** v1's independent-CLI tool-call review wires the vendor's native pre-tool hook (claude `PreToolUse` / gemini `BeforeTool` / opencode `tool.execute.before`) to an external reviewer, but has documented `known_interception_gaps` and the reviewer **fails open** (malformed → allow). So the finer-grained verify-and-refute benefit (Section 13) is only *partially* preserved across vendors. The engine **records when an interception gap means a tool call went un-reviewed**, so the degradation is visible, not silent.
- **Bidirectional streaming** is only documented for Claude (`--input-format stream-json`); other vendors get single-shot scoped tasks.
- **Internet/web-search** differs (Codex `web_search` config, Gemini built-in, Claude via WebSearch tool/MCP); negotiation (§3.5) records which is active per-vendor.

### 3.5 ACP-Style Capability Negotiation

The Executor performs an `initialize`-style handshake modeled on the [Agent Client Protocol](https://agentclientprotocol.com/get-started/introduction) (`protocolVersion=1`, JSON-RPC 2.0 over stdio, capability negotiation in `initialize`, adopted by 25+ agents incl. Gemini CLI and Copilot CLI). This consolidates v1's scattered fragments (`_internet_launcher_args`, schema delivery, `CLIToolHookSupport`, `_CLIBackendSandboxSpec`, `--effort low..max`) into **one** negotiation layer rather than per-call special-casing.

```python
def negotiate(vendor, model, cli_version) -> CapabilityProfile:
    probe = run_capability_probe(vendor)        # vendor self-report (advisory only)
    declared = STATIC_CAPABILITY_TABLE[vendor]  # APEX's hardcoded, version-keyed truth
    # APEX's declared table WINS on conflict — vendor self-report is advisory telemetry.
    profile = merge(declared, probe, prefer=declared)
    profile.cli_version = resolve_npm_version(vendor)   # recorded for manifest + drift detection
    assert_conformance(profile, cli_version)    # §3.8: does the mapped sandbox/schema actually take effect?
    return profile
```

APEX can either (a) speak ACP as a client to ACP-capable workers (opencode `serve acp`; Gemini was first external integration) or (b) borrow ACP's `initialize`/capability schema for its own Executor handshake over the existing subprocess path. We adopt **(b) as the default** (it covers all five vendors uniformly via the subprocess adapters) and **(a) opportunistically** for opencode where it avoids MCP cold-start. The borrowed [A2A Agent Card](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) idea — a small, signed, discoverable manifest of `{skills, auth, transport, model}` — is the template for the per-backend `CapabilityProfile`/run-manifest schema (§3.7), so fleets are self-describing and replayable. **Caveat (honored pitfall):** "ACP" overloads three protocols and remote (HTTP/WebSocket) support is WIP; A2A/Agents-SDK target agent-to-agent/in-process, not fleet leaf-execution. We borrow the *handshake pattern*, not bet the executor transport on a single emerging standard.

### 3.6 Mixed in One Run: Heterogeneous-Fleet Routing

Cross-vendor diversity is a **first-class diversity/search axis** (accepted mechanism, `adopt`), placed alongside v1's strategy-axis/brief-family/effort/seed axes (CLI backends ignore temperature, so `(vendor, model)` is a *stronger* diversity lever — different model families fail differently, decorrelating hallucinations). The controller (Section 14) routes a `(vendor, model)` per decision node.

The evidence is direct: [Dissecting the SWE-Bench Leaderboards](https://arxiv.org/html/2506.17208v2) shows Devlo (70.2% SWE-bench Verified) generated candidates with three distinct models (Claude 3.7 Sonnet + o3 + Gemini 2.5 Pro); TRAE (70.4%) generated with Claude 3.7 Sonnet + Gemini 2.5 Pro + o4-mini and **selected with o1**; AgentScope used Qwen2.5 to select among Claude 3.5 trials. Heterogeneous-fleet generation + an execution-grounded selector beats single-vendor best-of-N.

The critical, non-negotiable condition (adversarial verdict): **this win materializes ONLY with an execution-grounded selector.** Without verification, diverse-but-wrong candidates add noise ([HeuriGym: higher diversity lowers yield via invalid outputs]). APEX's Cardinal Safety Contract (Section 13; execution-evidence-authoritative; soft signals re-rank-within-tier or downgrade only, never promote) is exactly that selector. **Rule: never ship cross-vendor diversity without the Cardinal-Safety verifier gate.** That is the difference between coverage and noise.

Reviewer-independence caveat (honored pitfall "Do not fold metacode into the opencode family when claiming cross-vendor reviewer independence"): v1's family-disjoint reviewer check (actor family ≠ reviewer family) currently folds `metacode` into the `opencode` family, so an `opencode + metacode` pair gets **no** independence gain. For true decorrelated cross-vendor review we either (a) stop folding metacode into the opencode family, or (b) explicitly document that same-family pairs yield no independence gain and exclude them from independence accounting. The roadmap chooses (a).

Resilience substrate (accepted mechanism "Two-tier failure memory + self-evicting BackendPortfolio", `adopt`, kept verbatim): a 429/stall on one vendor must not poison a healthy one. v1's distinction between **call-failover** (current-stage reroute only — 429/529, stall, connection reset) and **backend-level global reroute** (auth/401, missing binary, SDK breakage) is preserved, with the self-evicting `BackendPortfolio` (`run_backend_portfolio.json`) honoring `retry_after_seconds`. A per-vendor retry adapter keys off native signals (Claude `system/api_retry` with categories rate_limit/overloaded/server_error; Codex/Gemini exit codes) to drive unified backoff + cross-vendor failover, avoiding thundering-herd into 429s.

### 3.7 Cost Arbitrage: A Verification-Gated Cascade, Not Blind Routing

Cost arbitrage is **demoted from "net advantage" to "opt-in, verification-gated cascade"** (adversarial verdict; the weakest leg of the bundled claim). The accepted mechanism "Model economy as sub-role, verification-gated cascade" is `adopt-modified`. Do **not** do static up-front routing: [xRouter (2510.08439)](https://arxiv.org/html/2510.08439v1) shows hand-crafted "expensive-for-hard/cheap-for-easy" trees are brittle and do not transfer across providers, and the **almost-right trap** means cheap executors needing 3–4 retries + human review cost *more* than one frontier pass. **Measure token YIELD (cost per verified-resolved task), not invoice.**

The cascade (fits APEX's existing cheap-first verify-on-diff loop perfectly):

```
1. Frontier PLANNER (one vendor) decomposes + writes scoped contracts.   [keep frontier here]
2. Cheap cross-vendor EXECUTOR satisfies a narrow, well-specified step.   [cheapen ONLY here]
3. APEX cheap-first verification on the diff (AST → symbol survival → targeted pytest).
4. On verification FAILURE → escalate to frontier executor (rewrite-cycle cap).
5. Frontier REVIEWER owns the final quality gate.                        [keep frontier here]
```

Which roles are safe to cheapen is settled by the [HyperAgent ablation](https://arxiv.org/html/2409.16299v1): weakening the **Navigator (codebase exploration)** or **Editor (multi-file editing)** roles causes the *worst* resolve-rate drops because they need sustained long-context environment interaction; the **run/verify Executor** is the substitutable role. small open models trail frontier sharply on hard repo SWE (HyperAgent's own Llama-3-8B "Lite" variant scores ~16%, far below frontier-tier resolve rates). Therefore: cheap is safe for run/verify and narrow single-tool calls; **risky** for navigation and multi-file edits on hard repo SWE. The accepted mechanism "Heavy-orchestrator + thin executor as the default execution shape" is `reject` — that default regresses toward the cheap-model baseline. Cascade-with-verification (à la [FrugalGPT](https://arxiv.org/abs/2305.05176), up to 98% cost cut at GPT-4 quality with a cheap scorer; [Aider architect/editor](https://aider.chat/2024/09/26/architect.html), Pareto-improving when the editor is competent) is the safe form.

This stays consistent with v1's "never optimize for cost" directive via the `budget{}` primitive (Section 2): `budget {total, spent(), remaining()}` is first-class but **defaulted unbounded**; cost arbitrage is opt-in. Realizing it requires cross-vendor token/cost *normalization* (different tokenizers, tiers, and the Claude Agent-SDK credit pool) that v1 deliberately does not have yet — so we build the accounting layer before publishing any savings figure.

#### Run-Manifest Pinning & Artifact Replay

`RunManifest` extends v1's existing manifest (which already pins `apex_git_sha`, python/platform, model_versions, docker_images digest-pinned, harness versions) to pin **per rollout**: `{vendor, model, resolved cli_version (npm), session_id, sandbox_mode, capability_profile, prompt_hash}`. Replay reproduces **artifacts (diffs) and re-runs verification**, not token streams — because temp-0 is not bitwise reproducible across hosted APIs (Thinking Machines; the accepted "Bit-reproducible agent OUTPUT replay" is `reject`). This satisfies the mandate's "pin vendor+model+version for replay" and is the substrate the durable journaled resume (Section 15) needs. Version pinning (`npm i -g @openai/codex@X.Y.Z`, `@anthropic-ai/claude-code@X.Y.Z`) defends against fast-moving CLI breaking changes; the resolved version is captured, not just requested.

### 3.8 Uniform MCP Tool Plane & Conformance Testing

To make tool capability identical regardless of leaf vendor, APEX **injects the same MCP server set into every backend** (Codex required-MCP, Claude `--mcp-config`, ACP passes MCP endpoints+credentials at session start, opencode server config). This means a branch's tool capabilities do not depend on which vendor executes it — a precondition for fair cross-vendor diversity and for the controller to route freely.

Because harness-dominates-model (30–50pt swings), the Executor is treated as **part of the harness** and gets a per-vendor **conformance test** asserting that the mapped sandbox/schema/tool-allowlist actually take effect (e.g., a read-only request truly blocks writes; an injected MCP server is truly reachable; an embedded schema truly yields a parseable object). This surfaces version drift *loudly* at run start rather than silently eroding the diversity gain mid-run, honoring "pin CLI versions against drift."

### 3.9 Summary of Dispositions

| Mechanism | Disposition | Net |
|---|---|---|
| Filesystem/git-diff as contract; JSON events as telemetry | adopt (kept verbatim) | The enabler; correctness never trusts vendor self-report |
| Normalized Executor + ACP-style negotiation, graceful degradation | adopt | Consolidate v1 fragments; degrade-not-crash to APEX floor |
| `(vendor, model)` as first-class diversity axis | adopt | Decorrelated cross-family errors widen coverage — **only** with the Cardinal-Safety selector |
| Two-tier failure memory + self-evicting BackendPortfolio | adopt (verbatim) | One vendor's 429 cannot poison a healthy fleet |
| RunManifest pins vendor+model+cli_version+profile; artifact replay | adopt (verbatim) | Reproduce diffs, not token streams (temp-0 not bitwise reproducible) |
| Cost arbitrage as verification-gated cascade | adopt-modified | Cascade-not-route; measure token yield; latent until opt-in |
| Heavy-orchestrator + thin executor as the default shape | reject | Cheapening navigation/multi-file editing regresses to cheap-model baseline |
| Trusting vendor self-reported JSON as correctness contract | reject | Verify on diff |
| Bit-reproducible agent OUTPUT replay | reject | Impossible across hosted APIs |
| Folding metacode into opencode for reviewer independence | reject | Same-family pair → no decorrelation gain |

The net verdict to carry forward: **feasibility is sound; "no benefit loss" is an overclaim (bounded, declared degradation is the truth); cross-vendor diversity is a sound advantage given APEX's execution-grounded selector; cost arbitrage is a conditional, opt-in cascade.** Cross-references: the engine primitives this Executor plugs into are Section 2; verify-and-refute and the Cardinal Safety Contract are Section 13; the model economy cascade detail is Section 12; isolation/determinism/durable resume are Section 15; the active controller that routes `(vendor, model)` is Section 14.

## 4. APEX v1: Foundation, Strengths & Ceiling

APEX-Ω is not a rewrite. It is an extension of a working, hardened kernel. This section is the load-bearing recap of that kernel — APEX v1, the "Adaptive Parallel EXecution" orchestrator — and it serves three jobs simultaneously. First, it fixes the **substrate**: the exact contract, control flow, and invariants the redesign builds *on top of* and is forbidden to weaken (see Section 7 for how these become first principles, and Section 18 for the kept/modified/dropped ledger). Second, it inventories the **reusable assets** so later sections cite concrete code seams rather than reinvent them (the engine in Section 2/8, verification in Section 13, the controller in Section 14, the model economy in Section 12). Third, it names the **ceiling** — the four change-seams and the cost stack — that every redesign mechanism is justified against.

The framing of this whole plan is honest about that order of operations: APEX v1 is the *substrate*; speculative search, CTDG pruning, the epistemic blackboard, the model economy, and the active controller are *extensions expressed as workflow patterns over vendor-neutral workers*. Nothing below is a proposal. It is the description of what already runs, what we keep verbatim, and where the next-generation work attaches.

### 4.1 The Contract: One `solve()`, One Diff, Four-Way Status

APEX v1 exposes exactly one entrypoint, and the entire plan inherits its signature as the public contract:

```python
ApexOrchestrator.solve(
    repo_path: str,
    issue_description: str,
    test_command: str | None = None,
    benchmark_metadata: dict | None = None,
    verification_test_command = _INHERIT_VERIFICATION_TEST_COMMAND,
) -> ApexResult        # apex/orchestration/solver.py:646
```

Given `(repo_path, issue, optional test_command)` it produces **one unified diff plus one terminal `Status`** drawn from a 4-way enum (`apex/core/status.py`):

| `Status`      | Meaning                                                            | `success` |
|---------------|-------------------------------------------------------------------|-----------|
| `SOLVED`      | A candidate passed execution-grounded verification and was accepted | `True`    |
| `ABSTAINED`   | No candidate earned positive execution evidence; we decline to guess | `False`   |
| `FAILED`      | Genuine APEX miss (a real attempt produced no acceptable patch)   | `False`   |
| `ENV_SKIPPED` | Environment/infra failure; *not charged to the model*             | `False`   |

`success == (status is SOLVED)`. The four-way split — and especially `ABSTAINED` as a first-class peer of `SOLVED` rather than a degenerate `FAILED` — is the structural expression of the central thesis: with an imperfect verifier, the cost of a confident-wrong accept dominates, so abstention-over-guessing must be representable in the type system, not buried in a confidence float. APEX-Ω keeps this enum byte-for-byte.

The result is produced by a strict five-phase pipeline, driven by `solve()` as a thin, stateless-across-runs sequential coordinator:

```
preprocess (1) -> plan (2) -> rollouts (3) -> verify (4) -> select (5)
```

```
solve()
  ├─ _maybe_solve_via_in_container_v5     # benchmark short-circuit (gated by benchmark_metadata)
  ├─ _prepare_run                         # Phases 1-2 + baseline -> 9-tuple
  │     (repo_context, verifier, planner, strategy, issue_plan,
  │      task_state_graph, baseline_result, resolved_verification_test_command,
  │      orchestration_transitions)
  ├─ RolloutEngine(config, repo_path, repo_context)
  ├─ _run_pipeline                        # Phases 3-5 + 4 follow-up recovery loops
  └─ _build_final_result -> ApexResult
```

Phase data-flow and the load-bearing objects:

| Phase | Producer | Output object | Lifecycle discipline |
|-------|----------|---------------|----------------------|
| 1 preprocess | `RepoAnalyzer.analyze()` | `RepoContext` | **Built once, read-only thereafter** (amortized context — scan the repo once, reuse across all parallel attempts) |
| 2 plan | `planner.build_execution_strategy()` | `PlanningDecision` + `IssuePlan` | `IssuePlan` is **the central mutated-throughout object**, threaded into every rollout, escalation, follow-up, and selection |
| baseline | `verifier.capture_baseline()` | `BaselineResult` | One full-suite run per `(repo, command)`, cached |
| 3 rollouts | `RolloutEngine.execute_rollouts()` | `list[RolloutResult]` | `RolloutResult` is the **atomic unit** flowing generation → verification → selection → recovery |
| 4–5 verify+select | `PatchVerifier` + `PatchSelector.select_best_patch()` | winning `RolloutResult` | Phase 4 is **not** a standalone method — verification runs *inside* the selector and inside per-rollout `quick_verification` |

One subtlety to carry forward: **there is no separate benchmark pipeline**. The published numbers come from the same `solve()` a library user runs; `benchmark_metadata` is the *only* differentiator and merely gates two short-circuits (the V5 in-container path and artifact-safe plan stripping). This is a deliberate anti-cheat property — any divergent benchmark path reopens benchmark-specific cheating — and Section 3's vendor-neutrality argument and Section 20's evaluation plan both depend on it.

### 4.2 The Non-Negotiable Invariants (and Why Each Is Load-Bearing)

The credibility of best-of-N orchestration rests on a small set of invariants. The redesign's job is to *amplify* what these protect, never to relax them. Each is stated with the guarantee it provides and the specific failure that re-emerges if it is removed.

#### 4.2.1 The Cardinal Safety Contract — execution-evidence-authoritative selection

> *"Execution evidence is authoritative. Soft signals may re-rank within an execution-verified tier, or downgrade an already-accepted candidate — they may NEVER promote an unverified candidate."* (Blueprint §13.1)

This is the single rule running through every selection component. It is enforced *structurally*, not by convention:

- `_apply_evidence_bound_review` flips `accepted` **only `True → False`**. The adversarial veto, the fresh-context `FinalAcceptanceReviewer`, and the clarification-abstain arm are all downgrade-only. (`VerificationResult` has no `status`/`passed` field; `accepted` is the gate.)
- The deterministic ranking key is a fixed lexicographic tuple in which every soft/learned/LLM key sits strictly below every execution + critic key, terminating in a content-derived tiebreak — never insertion order:

  ```
  rank_key = (
      combined_score, accepted, public_signal_score, critic_score, size,
      verification_score, eg_critic_tiebreak, perspective_score,
      len(changed_files), -cluster_id            # never insertion order
  )
  ```

**Load-bearing because:** this directly counters the *Inference Scaling Flaws* failure mode — with an imperfect verifier, repeated-sampling gains saturate or invert because wrong code that the verifier (or an LLM judge) likes gets shipped. Pointwise LLM judges capture only a fraction of best-of-N gain, so the LLM is bounded to a downstream tie-break. Remove this rule — let any soft signal (deterministic `SelectionCritic`, learned EG-critic, `SelectorAgent` vote, perspective/final-acceptance reviewers, process quality, evidence ledger) promote an unverified candidate — and the selector ships LLM-preferred-but-unexecuted patches; the published number stops measuring capability; the entire credibility argument collapses. This is why every redesign mechanism (CTDG pruning §10, plan scoring §14, blackboard sharing §11, the generative critic §13) is admitted only in **re-rank-within-tier or downgrade-only** form. The Cardinal Contract is what converts log-linear *coverage* into trustworthy *resolved issues*.

#### 4.2.2 The cheap-first verification cascade that never synthesizes a pass

A confidence-ordered ladder runs high-precision cheap filters first, so expensive selection only sees survivors:

```
syntax (AST py_compile of changed files; hard 0.0 on fail)
  -> lint (flake8 --select=E9,F63,F7,F82; no bonus if flake8 absent)
  -> reproduction (+0.35)
  -> regression-prune (re-run ONLY baseline-passing tests, chunks of 50, in candidate worktree; +0.35)
  -> cross-validate (+0.10 * mean)
  -> score (pass_rate adds 0.10 * pass_rate)
  -> accept
```

Three encoded countermeasures define the "never synthesize a pass" discipline:

- `rc == 124` (timeout) → a **separate axis** `regression_inconclusive` (+0.15 partial), *not* a failure.
- Silent `rc == 0` no-op → `errors = 1`, **never `passed = 1`**.
- Singletons **abstain** (empty cross-validation list), never a synthetic `0.5` prior; the self-index `M[i][i]` is excluded.

Regression pruning is a precision *and* speed primitive: it re-runs only baseline-passers in targeted chunks of 50 nodeids in the candidate worktree (not the whole suite), and on a collection error it expands the file key to every baseline nodeid under that file prefix so import-time breakage cannot silently survive. `PruneResult.is_valid == False` drops the candidate.

**Load-bearing because:** without cheap-first ordering, cost explodes (cross-validation and LLM votes run on syntactically-broken candidates). Without the guards, a no-op command, a timed-out-but-passing suite, or a zero-collected run is recorded as a pass — the exact false-`SOLVED` paths the system exists to prevent. This cascade *is* the safe per-candidate prune the redesign wanted (Section 10, Section 13); it is already proven, and CTDG/coverage pruning attaches as a *prioritizer and backstop around it*, never as a replacement gate.

#### 4.2.3 Per-rollout git-worktree isolation + scoped `fcntl` locks

Every rollout gets a private workspace. The per-rollout `fcntl` advisory lock (`flock LOCK_EX|LOCK_NB` at `workspace_dir/.locks/rollout_<id>.lock`) is taken **before** touching the workspace path and released on every failure path; it raises `ConcurrentWorktreeError` if held (Windows fallback = PID marker). Isolation degrades downward through three tiers plus an override:

```
seed_clone -> worktree -> snapshot -> synthetic
```

Snapshot baselines are deterministic (`_bootstrap_git_snapshot` commits with fixed author/committer and date `2026-01-01T00:00:00+0000`, message derived from `source_head8 + dirty_hash8`), so two identical source states produce **bit-identical commit SHAs → bit-identical diff text**. Critically, **there is no machine-wide mutex** — all concurrency, kills, registries, and secret boundaries are scoped to a single rollout (the `RolloutCLIRegistry` binds each CLI pid to a `rollout_id`; `terminate_rollout_children` escalates `SIGTERM → SIGKILL` across only that rollout's pids).

**Load-bearing because:** this is the primitive that makes *any* parallelism or branching safe. The CAID ablation is explicit — soft isolation 55.5 < single agent 57.2 < worktree **63.3**. Remove it and K parallel attempts corrupt each other's filesystem, cross-rollout writes pollute candidate diffs, reaping one stalled rollout touches its siblings, and untrusted external agent code running test suites leaks into shared state. The speculative tree-search of Section 9 and the parallel/pipeline fan-out of Section 8 are *only* sound because every branch lands in an isolated, lock-guarded worktree.

#### 4.2.4 Filesystem-as-source-of-truth (the vendor-neutrality enabler)

Every phase boundary persists a durable atomic-write artifact (`repo_context.json`, `baseline_result.json`, `issue_plan.json`, `controller_decisions.jsonl`, `apex_result.json`, `run_manifest.json`). `status`/`watch` subcommands **read artifacts only** — they never attach to the scheduler or hold a lock. Results are written to disk, **not flooded back into an orchestrator context window**. The atomic-write pattern is uniform: normalize target → `json.dumps(indent=2, sort_keys=True, default=str)` → `mkstemp(...).tmp` → write → flush → `os.fsync` → `os.replace` (readers see old-or-new, never torn) → unlink tmp on failure.

**Load-bearing because:** this is the concrete mechanism of vendor neutrality (Sections 2–3). Because state lives on disk and each agent is observed via stdout turn-parsing + artifacts rather than driven in-process, opaque external CLIs (`claude`/`codex`/`gemini`/`opencode`/`metacode`) are interchangeable behind `AGENT_NAME_TO_CONFIG`. It is also independently corroborated by context-rot findings (frontier models degrade well below their window limit, lost-in-the-middle): keeping intermediate results in script variables and a durable journal — never a conversation window — is the scaling unlock that lets a large run avoid drift. Remove it and the system couples to a specific in-runtime agent framework, monitoring perturbs runs, crashes lose all intermediate state, and there is no auditable record of why a rollout ended or which candidate won.

#### 4.2.5 Determinism + `RunManifest` + Docker digest pinning + strict replay

Determinism is **best-effort around irreducibly-stochastic agents** (temperature 0.0 default; CLI backends ignore temperature entirely) but pinned everywhere pinnable: candidate ordering by `(rollout_id, content_hash)`, cluster verification in `cluster_id` order, `Random(0)` mutation seeds, content-sha tie-breaks. `RunManifest` captures git sha/dirty, python/platform, seed, redacted `APEX_*` env, model ids, Docker digests, harness versions. `resolve_image` pins tags to `repo@sha256:` via `prepinned -> registry -> docker_inspect -> bare`, with an `@sha256:`-in-tag short-circuit that avoids the malformed double-pin bug; it never raises and records which path won. `apex replay-deterministic --verify` re-runs a recorded session and asserts the reproduced trajectory matches.

**Crucial honesty constraint (carried into Section 15):** manifest pinning guarantees *environment + ordering*, and replay guarantees the *trajectory*; neither guarantees agent **output**. Bit-reproducible agent OUTPUT replay is impossible across hosted APIs (temperature-0 batch non-invariance) and is therefore explicitly rejected in the canonical disposition list — APEX reproduces *artifacts* (diffs + re-run verification), not token streams.

**Load-bearing because:** without manifest + pinning, image drift silently changes scores and provenance cannot distinguish environment from model; without deterministic ordering/tie-breaks, re-runs select different winners and slot-0 bias creeps in; without replay, there is no debugging substrate for a stochastic agent. This is also what makes a *learned* controller (Section 14) tractable: reproducible off-policy credit assignment over journaled decisions.

#### 4.2.6 Escrow WAL / commit-then-publish / idempotent exactly-once (CCEDF)

The escrow WAL at `<run_dir>/escrow/confirmed_wal.jsonl` is fsync-durable, `flock`-guarded, monotonic-seq, idempotent exactly-once durability for confirmed candidates. Replay is latest-wins by `seq` per `idempotency_key` (`task_id::candidate_id`) then best-by-task by `(score, seq)` — **ordering uses `seq`, not wall-clock**, so replay is order-stable and a duplicate append is harmless. The engine wraps the call in a bare `except`: durability is a backstop that must never become fatal.

**Load-bearing because:** it fixes the dominant Commit0 loss — a rollout that reached `pass_rate == 1.0` then dropped to `scheduler_cancelled` and was lost. Remove it and a confirmed full-scope pass produced early, then preempted during a later wasted wave, is lost forever; long-running preemptible parallel search becomes lossy and non-resumable; published rates understate true coverage. The redesign promotes this narrow WAL plus v1's unused `ReplayRecorder` into a **per-`agent()`-call WAL** (Section 15) that doubles as the off-policy credit substrate for the learned controller — but the durability semantics here are kept verbatim.

#### 4.2.7 Anti-cheat / fairness / failure taxonomy / first-class abstention

A coding-agent orchestrator is only as credible as its number, and capable agents are reliable adversaries against weak oracles. The non-negotiables:

- **Upstream Docker harness is the only publishable number.** APEX-private rescoring is diagnostic-only and published as a delta (`fairness_audit.json`, `FLAG_THRESHOLD = 0.02`); the fairness audit runs two scorers over the *same* pre-computed evaluation (O(N), not O(2N)).
- **Gold fields are scrubbed at LOAD time — discard, not mask.** The classic `SWEBenchTask` dataclass literally has no field to hold `patch`/`test_patch` (only `scrubbed_test_patch_summary` + integer F2P/P2P counts). Masking is rejected because a redacted field is one prompt-template bug from leaking.
- **Commit0 uses a true git-history flatten** (`rm -rf .git` + nested submodule `.git`, rootless re-init, post-conditions assert `rev-list --all --count == 1`) — *block the channel, never the neuron*: the gold-recovery channel is destroyed while weights-encoded capability is untouched.
- **NDFF flake firewall** declares flakes only on positive evidence and never re-runs a real failure; the **anti-hack ledger** grounds assertions by structured fingerprint (op+shape+repr), biases to false-negatives, soft-downweights.
- **`patch_sanitizer`** distinguishes `VENDORED_UPSTREAM_ARTIFACT` (stripped silently, requalified as a soft advisory signal) from `GOLD_PROTECTED_TEST` (reject wins) so an attacker cannot disguise a gold-test edit under a vendored `testing/` dir.
- **Failure taxonomy** (`FailureClass`, 8 buckets + `UNCLASSIFIED`, phase-ordered, never raises): `charges_apex == {APEX_MISS, UNCLASSIFIED}` only; `HARNESS_BUG` is excluded from *both* `is_environment` and `charges_apex`; `ENV_*`/`NON_DETERMINISTIC` are retryable. Phase makes `ModuleNotFoundError` an install-failure during setup but an APEX-miss during test execution.
- **Abstention is first-class** (echoing §4.2.1): all-env-failed → `None`; `cross_candidate_voter` returns `winner=None` when all `oracle_scores == 0` (caller keeps baseline pick); the strict gate `_selected_result_is_accepted` requires positive evidence (legacy `overall_score >= 0.9` shortcut removed); the deterministic `HeuristicRepairAgent` fallback stamps a hard-coded `0.35` confidence + "LLM execution was unavailable" so a heuristic patch is never mistaken for a real solution.

**Load-bearing because:** reward-hacking *scales* with capability and contamination is the dominant validity threat — non-negotiable for a mixed-vendor fleet (Section 3). Remove any one and a sufficiently capable agent finds a cheaper-than-solving path to a green checkmark; the headline number stops measuring capability; environment failures get charged as model misses (or vice-versa), corrupting both the published rate and the retry policy.

### 4.3 Existing Assets the Plan Reuses

The redesign explicitly lifts and re-implements, rather than rebuilds, a mature vendor-agnostic foundation. The canonical disposition for the workflow engine is **adopt**: lift v1's `run_structured_prompt` (= the `agent()` primitive) and `execute_rollout_requests` (= the `parallel()` primitive) into a re-implementable engine. The assets below are the concrete seams.

| Asset (v1 location) | What it is | Where the plan reuses it |
|---------------------|-----------|--------------------------|
| **`FrontierSearchController` (PUCT)** | The existing bounded best-first search over rollout frontier with virtual-loss / `min_branch_reward` accounting | Section 9 — the **antecedent** for adaptive branching; *not* re-described as new |
| **`EpisodicMemoryBus`** (`rollout/engine.py`) | Append-only, thread-safe Discovery store; cross-sibling + cross-solve priors (reserved negative `rollout_id`s ≤ −1); negative/ruled-out sharing; `query()` excludes caller's own id; caps `positive_limit=5 + negative_limit=3`; `extract_durable_insights` caps at 64 | Section 11 — Blackboard 2.0 *evolves its delivery*, keeping relevance/confidence/dedup/own-rollout-exclusion |
| **`RepoGraph` / `RepoContext`** | Built-once, read-only repo model | Section 10 (CTDG attaches to it) and the amortized-context discipline |
| **`contract_slice.py`** | Localization / scope-slicing primitive | Section 10 (CTDG test prioritizer + dynamic-coverage prune) |
| **`BackendPortfolio`** (`core/backend_portfolio.py`) | Per-run persisted ledger (`run_backend_portfolio.json`) of disabled `(backend, command)` fingerprints with `retry_after_seconds`; `is_disabled` self-evicts expired entries | Section 14 — two-tier failure memory; prevents a 429 on one vendor poisoning a heterogeneous fleet |
| **`enable_speculative_first_attempt`** | The existing cheap speculative first attempt (gated to easy tasks, `difficulty <= 0.25`) | Section 9 — `speculate()` fork is admitted as an *extension* of this, at turn/checkpoint boundaries only |
| `CLIModelClient.run_structured_prompt` | Launches an opaque CLI subprocess running its own multi-turn loop; observes via stdout + watchdog; **never raises** (typed `CLIModelResult`) | Section 2/8 — the `agent()` worker primitive |
| `execute_rollout_requests` + `WorktreePool` | K-wide parallel rollout dispatch with ~10x worktree recycling | Section 2/8 — the `parallel()` primitive |
| `CLITurnParser` + `turn_observer` | Splits stdout into `Turn` objects; the *only* mid-flight steering channel over an opaque CLI | Section 11/14 — turn-boundary sharing & control |
| `controller_decisions.jsonl` | Already logs every controller decision | Section 14 — substrate for the learned active controller |

**A standing caution the plan must honor:** `FrontierSearchController` is the antecedent, not a novelty. The redesign's "speculative tree-search" *re-describes* this existing search. The disposition is therefore **adopt-modified** — keep the AB-MCTS-style adaptive allocation that wins, run it *inside* `FrontierSearch` budget caps, and make collapse to verified best-of-N mandatory below a feedback-confidence floor. Classical/distributed MCTS as the core loop is **rejected**: that verdict is unsound (it re-describes `FrontierSearch`, plain MCTS does not reliably beat verified sampling at repo scale, and it is brittle against non-serializable container state).

### 4.4 The Four Change-Seams

The redesign attaches at exactly four seams. Each is a clean integration point with existing dataclass boundaries; none requires touching the §4.2 invariants.

| # | Seam | Current v1 behavior | Redesign attachment (target section) |
|---|------|---------------------|--------------------------------------|
| 1 | **Linear scaffolded pipeline** | Strict one-directional JSON-artifact handoff `Reproducer → Localizer → Patcher [→ TestWriter]`; no inter-agent dialogue; agents "deliberately ignorant of waves/escalation/selection" | The `pipeline()` per-item staged-streaming primitive (Section 2; detail §16.3) — the one genuinely net-new engine primitive; cuts wall-clock from sum-of-slowest-per-stage to slowest-single-chain across reproduce→localize→patch→verify. The JSON-only anti-misalignment discipline is preserved |
| 2 | **Passive controller** | Two layers: the wave/escalation loop only decides count/escalation; the calibrated policy layer is *blend-not-switch* (`evaluate_policy_model`: `applied=False ⇒ value==baseline`) and its intended kill switch `library_enabled` is **unwired** (zero runtime consumers) | The active adaptive controller (Section 14) — staged bandit → GEPA → RL; **blend-not-switch, fail-open to heuristic** preserved; `library_enabled` finally wired or removed |
| 3 | **Blind redundant rollouts** | N redundant attempts at the SAME task in isolated worktrees; diversity by strategy-axis/prompt/seed; no coordination during generation | Bounded adaptive branching + `(vendor, model)` as a diversity axis (Sections 9, 12) — steer later rollouts away from confirmed dead ends; worktree isolation kept as the hard safety primitive |
| 4 | **Append-only blackboard** | `EpisodicMemoryBus` + typed `TaskBlackboard` are append-only discovery stores | Blackboard 2.0 (Section 11) — phased, abstracted negative-constraint sharing at turn boundaries; verifier must not see producer context |

Two seam dispositions are explicitly **rejected** and must not creep back in: raw *share-all / "instant push" mid-subprocess injection* (share-all measurably lowers accuracy and homogenizes attempts; mid-subprocess prompt mutation is infeasible against opaque CLIs and breaks determinism/replay), and *heavy-orchestrator + thin-executor as the default shape* (HyperAgent ablation shows cheapening navigation/multi-file editing causes the worst resolve-rate drops).

### 4.5 The Ceiling: The Cost Stack and "15/16 Doomed"

The ceiling is cost, and it is structural. APEX v1 is built to **"optimize for SOTA, never for cost,"** and the defaults make that literal:

- **Non-adaptive fixed-K default.** `RolloutConfig.enable_adaptive_allocation = False`, so `_requested_rollout_budget` returns `num_rollouts` (default **5**) and the planner runs that fixed count regardless of difficulty (the `max_rollouts=16` cap with buckets `[1, 4, 8, 16]` binds only when adaptive allocation is on, or when escalation/the portfolio floor raises the count); the portfolio floor only *raises* the count, never lowers it. The difficulty-adaptive low-K path exists and is fully wired (`estimate_difficulty → compute_rollout_count → evaluate_policy_model('planning.rollout_count') → _clamp_rollout_bucket`) — it is just off by default. Turning it **ON by default** is the canonical **adopt** disposition and the single biggest cost lever (optimal K is often < 10).
- **Caps off.** `repo_token_cap = None`, `max_tokens_per_repo_followup = 0`; the entire cumulative-token-cap machinery is inert in a normal run. The non-adaptive fixed-K default (5 rollouts, no down-scaling) with caps off is **rejected** as the headline cost pathology — replaced by adaptive low-K + budget-aware deepening, keeping full-cap only as the thin-feedback floor.
- **No wall-clock kill of a working agent.** Liveness is progress-based (S1–S7 inner watchdog; K1 outer stall, window 1200s × size_factor up to 6; emergency-silence ceiling 14400s = 4h; hard timeout opt-in, floored at 1800s). This is correct (it avoids killing slow-but-legitimate work) but it is also why a single rollout can run for hours.

The cost is paid in four multiplicatively-stacked layers — the **K × N² × waves** stack:

```
total ≈  K  (generation: K full agent trajectories (default 5, up to the 16 cap), each up to
             max_iterations_per_rollout turns at an 80–120k context ceiling)
       × per-rollout in-loop verification (targeted pytest, cached per patch)
       × N² selection (cross-validation matrix: each candidate's tests run on
             every other candidate's worktree, sandboxed)
       × waves (escalation loop cap 20, progressive waves cap 6,
             follow-up iterations cap 24, selection rounds 4 ×3 on near-miss)
```

The **"15/16 doomed at localization"** pattern is the headline waste, stated honestly: **localization is amortized once** (top-K hypotheses from a single localizer seed K rollouts via `hypothesis i → rollout i mod K`), but the **full patch-and-verify trajectory is replicated K times**. On an easy task that one good rollout would solve, the other 15 trajectories are pure redundant spend. The precise framing matters for the redesign: the redundancy lives in *patching*, not localization, and coverage scales log-linearly — so the binding constraint is *selection*, not localization. The **early localization-futility gate** (canonical **adopt**) routes budget to surviving hypotheses *before* the patch loop, informing allocation but — per the Cardinal Contract — never suppressing a candidate without execution evidence.

The dominant absolute cost hotspots, in priority order for Section 16 to attack:

| Hotspot | Why expensive | Rough magnitude |
|---------|---------------|-----------------|
| **Parallel rollout generation** (K opaque CLI agents/task) | Each rollout is a full multi-turn agent solve in its own worktree; default backend (codex_cli:gpt-5.5 first, claude_cli:opus failover; `--effort max` on the Claude path); no token cap | The largest absolute driver: up to **16× a single full solve**, × up to 6 waves / 20 strategy iterations on hard tasks. Inner parallelism capped at `parallel_workers=3` ⇒ ~`ceil(16/3)` sequential batches of the slowest rollout |
| **N×N cross-validation** (`build_cross_validation_matrix`) | Each candidate's suite executed against every other candidate's patched worktree, each a full sandboxed run (per-suite timeout 120s) | **O(N²)** sandboxed test executions/task; the dominant per-task verification cost for large ensembles. Two-pass AST clustering (threshold 0.95) dedups first, the main lever bounding N |
| **Regression baseline + prune** | Baseline = one full-suite run/`(repo,command)` up to 900s (cached); prune re-runs baseline-passers in chunks of 50 per candidate | `1 × full-suite` + `O(candidates × baseline_passers/50)` chunked pytest invocations |
| **LLM selection arms** | `SelectorAgent` up to 5 voters × 8 iterations; `PerspectiveReviewer` 4 lenses; `FinalAcceptanceReviewer` 1 fresh-context pass | Only on the tie-break path (≥2 selectable clusters); all default-off-or-fail-open, capping cost-and-variance |
| **Escalation + 4 follow-up loops** | On partial progress the controller *adds* rollouts rather than stopping; near-miss (≥0.95 pass rate) triggers a 3× multiplier on selection rounds | Bounded by iteration caps (20/6/24/4), **not** by tokens |
| **F2P oracle + dual-version voting** | Clones the repo twice, applies gold patch, runs the suite on both checkouts; dual-version generalizes to tests × surrogate-patches | `2 clones + 2 full suite runs`/F2P eval; `~(T + T·P)` sandboxed runs for the dual-version matrix |

### 4.6 What the Recap Establishes

This recap is the motivation to **keep**. The seven invariants of §4.2 are the substrate APEX-Ω inherits unchanged; removing any one collapses a named, specific guarantee, so the redesign is constrained to *amplify within them*. The assets of §4.3 are re-implemented, not rebuilt, and `FrontierSearchController` in particular is the antecedent the redesign re-describes — credit it as existing, not novel. The four seams of §4.4 are precisely where bounded, evidence-respecting extensions attach. And the cost stack of §4.5 — non-adaptive fixed-K (default 5, up to the 16 cap) with caps off, the K × N² × waves multiplier, and "15/16 doomed" in the *patch* loop — is the ceiling every later mechanism is justified against, with difficulty-adaptive low-K allocation (default ON) and the early localization-futility gate as the first, cheapest, highest-leverage wins. The honest through-line, carried forward to Section 7: **search and economy are bounded amplifiers; execution evidence is both the steering signal and the brake; verified best-of-N is the floor we can never do worse than.**

## 5. The Proposed Redesign (v3), Critically Assessed

The v3 redesign document (`APEX_DESIGN.md`) proposes evolving APEX from a "brute-force sample-and-select" engine into an "intelligent speculative search" engine built on five mechanisms: (1) Distributed MCTS + speculative branching via a `speculate()` fork tool; (2) a static-AST Code-Test Dependency Graph (CTDG) for millisecond test pruning; (3) Dual-Feedback Bidirectional Pruning (cheap pre-execution plan scoring + heavy post-execution reward); (4) a Cross-Branch Epistemic Memory / "Blackboard 2.0" with instant push injection; (5) a Contract-Driven Executor split where a heavy orchestrator writes contracts that thin fast executors satisfy.

This section does three things, in order. First, it establishes the **single most important fact the redesign omits**: three of its five "novel" mechanisms have substantial, unacknowledged v1 antecedents, and the others reduce to narrow deltas. Second, it runs each mechanism through the adversarial verdicts and assigns a **verdict-qualified disposition** — never a rubber stamp. Third, it gives the **build-grade form** of every adoption: data structures, control flow, config keys, and the safety contract each must respect. The governing principle throughout is the one from Section 1's thesis: **search and economy are bounded amplifiers; execution evidence is both the steering signal and the brake; verified best-of-N is the floor we can never do worse than.** Every adoption below is shaped to obey that principle; every rejection below is a case where the redesign's literal form would violate it.

> **Framing note.** `APEX_DESIGN.md` is a ~90-line prose blueprint, not code: it specifies no concrete v3 types, config keys, file paths, MCTS selection/backup policy, node/depth budgets, or `speculate()` signature, and it does not cite the `FrontierSearchController`, `EpisodicMemoryBus`, `RepoGraph`, or `core/contract_slice.py` it overlaps. The dispositions below are therefore reconstructed by matching the redesign's described behavior to v1's actual implementation; "novelty" is stated net-of-v1-antecedents. The concrete forms here become the substance of Sections 9 (tree search), 10 (CTDG/pruning), 11 (blackboard), 12 (model economy), and 14 (controller); this section fixes the dispositions those sections build on.

### 5.1 The Five Mechanisms and Their v1 Antecedents

The redesign presents its five mechanisms as a clean break from v1. The truth is closer to an *under-cited generalization* of code that already exists. Pitfall to avoid: do not present any of these as novel without flagging the antecedent. The table is the honest accounting.

| Redesign mechanism (`APEX_DESIGN.md`) | v1 antecedent (already shipped) | Genuinely new delta (net of v1) | Net novelty |
|---|---|---|---|
| **Distributed MCTS + `speculate()` fork** | `FrontierSearchController` (`apex/search/frontier_search.py`): BEST_FIRST/PUCT over branchable `WorkspaceSeed` checkpoints, `c_puct=1.25`, `virtual_loss=0.15`, `max_depth=6`, `max_frontier_branching=3`, `min_branch_reward=0.12`, transition reward, value backup (`_update_stats`), verification-as-early-stop (`_should_stop_search`) | An **agent-initiated, in-trajectory** `speculate()` fork (v1 frontier targets are planner-emitted from the `TaskStateGraph`, not agent-emitted mid-loop) | Low-to-moderate |
| **Static-AST CTDG** | `RepoContext.repo_graph` (`RepoGraph`): `contains`/`imports`/`inherits`/`references`/`uses`/`rationale_for` edges; `CoverageReport` via coverage.py / jest-istanbul (dynamic, computed late) | A **code→TEST** edge type (v1's graph has no test edge) used for fast test-subset selection | Moderate |
| **Cheap pre-exec plan scoring (Dual-Feedback)** | Post-execution half fully exists: `_build_patch_feedback_generator`, `verify_patch`/`_compute_score`, `prune_by_regression`, N×N cross-validation, `_compute_transition_reward` | A **pre-execution** plan-scoring gate before any code tokens (v1 evaluates only post-execution) | Moderate (pre-exec half only) |
| **Blackboard 2.0 push injection** | `EpisodicMemoryBus` (`apex/rollout/engine.py`, §8.8): append-only thread-safe blackboard sharing positive **and negative/ruled-out** `Discovery` records across siblings, relevance scoring, dedup-by-signature, own-rollout exclusion, `positive_limit=5`/`negative_limit=3` | **Push/instant** injection at boundaries vs v1's **pull-at-stage-boundary** query | Low |
| **Contract-Driven Executor split** | `core/contract_slice.py` `build_contract_slice(...max_files=8)`/`render_contract_slice`; contract obligations on `TaskBlackboard`; `HierarchicalAgent`+`BudgetPlanner` divide-and-conquer | An **economic heavy-plan / cheap-type MODEL-tier split** with stripped executor authority (v1 spawns one opaque CLI agent per rollout, no tier split) | Moderate |

Two structural facts dominate the dispositions:

1. **The opaque-CLI seam blocks the redesign's most ambitious primitives.** v1 spawns one opaque external CLI agent subprocess per rollout (claude/gemini/codex/opencode/metacode) and observes it via stdout turn-parsing; it does **not** drive an in-process tool loop (`apex/agents/*`, §10.1). Mid-trajectory pause/fork (`speculate()`), mid-flight prompt mutation (push injection), pre-execution plan inspection, and contract/thin-executor dispatch **all** require reshaping this seam. The vendor-neutral Executor interface (Section 3) is the consolidation point; until it exists, anything that needs to interrupt a running agent is admitted only at **turn/checkpoint boundaries**, which is what v1 already does.

2. **The redesign optimizes the wrong cost center.** v1 itself names verification/selection (K rollouts + N×N cross-validation, cited ~15× tokens; ~80% of variance from token usage) as the dominant cost. The redesign's savings target **generation** (cheaper executors, fewer redundant trajectories) and say little about verification. So even where a mechanism is adopted, the cost-benefit must be measured **net of verification**, never on gross generation tokens.

### 5.2 Distributed MCTS as the Core Loop — REJECT as headline; keep bounded adaptive-branching

**Verdict: unsound (high confidence).** The claim "distributed MCTS over the codebase beats verified best-of-N for repo-level SWE under a fixed budget" is contradicted on all four of its conjoined axes (it-is-MCTS, distributed-stateful, beats-best-of-N, fixed-budget). The single clean repo-level number, [SWE-Search](https://arxiv.org/abs/2410.20285) +23% relative, is measured against a **greedy single-trajectory** agent, not verified repeated sampling; SWE-Search's own Pass@5 (34.0) barely exceeds Pass@1 (31.0), proving the bottleneck is **selection, not search coverage**. Repeated sampling already extracts most coverage ([Large Language Monkeys](https://arxiv.org/abs/2407.21787): 15.9%→56% on SWE-bench Lite at <1/3 cost). [REBASE](https://arxiv.org/abs/2408.00724) found "MCTS underperformed plain sampling at every budget." [PRM reliability decreases with distance from terminal states](https://arxiv.org/abs/2510.20272) — exactly where MCTS prunes early — so verifier-guided tree search can prune all correct paths. And "distributed over the codebase" demands a shared mutable tree plus cheap state save/restore in non-serializable Docker containers, a hard engineering blocker.

This also collides with v1 invariants: a **shared mutable search tree** is the architectural opposite of context-isolation / filesystem-as-source-of-truth / per-rollout scoping; **LLM value-function pruning** of unexecuted nodes violates the Cardinal Safety Contract (a soft signal deciding candidate fate without execution evidence).

**What we keep — bounded adaptive-branching (disposition: adopt-modified).** The genuinely-winning part of the search literature is *adaptive allocation*, not MCTS rollouts/backtracking. [AB-MCTS / TreeQuest](https://arxiv.org/abs/2503.04412) generalizes best-of-N by deciding **go-wider vs go-deeper** per fan-out point and beats both repeated sampling and standard MCTS — but **only above ~64 calls**; under strict small budgets a budget-agnostic schedule can [lose to plain repeated sampling (BAVT)](https://arxiv.org/abs/2603.12634). The disposition is therefore:

- Run adaptive-branching **inside** `FrontierSearchController`'s existing budget caps (`max_depth=6`, `max_frontier_branching=3`, `min_branch_reward`, `virtual_loss` de-duplication) — reuse, do not rebuild. This is the v1 antecedent the redesign ignored, and it already supplies the budget-bounding and anti-duplication the redesign omits.
- **Mandatory collapse to verified best-of-N below a feedback-confidence floor.** When the per-branch feedback signal is thin (no reproduction, weak/partial tests), branching degenerates to noise; the controller must fall back to parallel best-of-N. This is the "floor we can never do worse than."
- **Branch only on git-worktree/commit checkpoints**, never Docker/process snapshots — sidestepping the non-serializable-state blocker. v1's deterministic snapshot SHAs already provide bit-identical branch-and-restore.
- **(vendor, model) is a search dimension.** [Multi-LLM AB-MCTS](https://sakana.ai/ab-mcts/) exceeds any single model (>30% vs ~23% on ARC-AGI-2) and decorrelates hallucinations — directly realizing the vendor-neutral thesis (see Section 3, Section 13).

The allocation policy is a pure function of recorded evidence (seeded for determinism — see Section 5.7 and Section 15):

```text
# Per-wave adaptive allocation (NOT a stateful MCTS tree).
# Runs on top of parallel()/the wave loop; no shared mutable tree, no rollouts/backtracking.
function allocate_next_wave(surviving_branches, remaining_budget, feedback_confidence):
    if feedback_confidence < FEEDBACK_CONFIDENCE_FLOOR:
        return collapse_to_best_of_N(remaining_budget)        # mandatory floor

    explore_bias = schedule(remaining_budget)                 # broad early, greedy late (BG-MCTS/BAVT)
    for b in surviving_branches:
        wider = thompson_sample_new_child(b, seed=content_hash(b.state))   # go-wider
        deeper = thompson_sample_refine(b,  seed=content_hash(b.state))    # go-deeper
        b.priority = blend(wider, deeper, explore_bias)
    capped = enforce_caps(surviving_branches,                  # reuse FrontierSearch caps
                          max_depth=6, max_frontier_branching=3,
                          min_branch_reward=MIN_BRANCH_REWARD,
                          virtual_loss=VIRTUAL_LOSS)
    return capped
```

Config keys (Section 9): `search.adaptive_branching.enabled` (default `true`), `search.feedback_confidence_floor` (default conservative, calibrated per benchmark), `search.budget_aware_schedule` (`broad_early_greedy_late`), `search.collapse_to_best_of_n_below_floor` (default `true`, **not** overridable to `false` in production).

### 5.3 Static-AST CTDG — REJECT as a gate; adopt as prioritizer + dynamic-coverage prune

**Verdict: unsound (high confidence) for the literal "static CTDG enables SAFE millisecond pruning in dynamic Python."** The conjunction fails on its load-bearing terms. [PyCG](https://arxiv.org/abs/2103.00587), the SOTA static Python call graph, has ~99.2% precision but only **~69.9% recall** — ~30% of real call edges are missing, and it explicitly ignores `eval`, `getattr`/`setattr`, built-in type effects, conditionals, and loops. Worse, the pytest test set is **not statically enumerable**: `parametrize`, `pytest_generate_tests`, fixture graphs from plugins, `conftest` tree effects, and `pytest_collection_modifyitems` all run at collection time, so a static graph cannot even name what it would prune (`pytest --collect-only` is the only reliable enumeration). [Reflection alone makes Java static RTS unsafe](https://lingming.cs.illinois.edu/publications/oopsla2019.pdf), and Python is strictly more dynamic. [Rothermel & Harrold's safety theorem](https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=1015&context=csearticles) makes "maximal pruning + zero false-negatives from imperfect data" structurally impossible.

Using a static CTDG as a **gate** would directly violate the Cardinal Safety Contract: a statically-missed coverage edge means a real regression test is never run, silently dropping a fault-revealing test (a false-negative with no execution evidence) — strictly worse than the prohibited "promote unverified." v1's `prune_by_regression` already guards exactly this by re-running only baseline-passing tests and expanding collection-error file keys to every baseline nodeid under the file prefix.

**What we adopt (disposition: adopt-modified): split "use as graph" from "prune as gate."**

1. **Prioritize, never exclude (zero false-negative risk).** Add a `code→test` edge to `RepoContext.repo_graph` (the existing `RepoGraph` change seam) and use it only to **reorder** the candidate test set most-likely-impacted-first. Reordering accelerates time-to-first-failure at zero safety cost and respects the Contract (re-rank, never eliminate). This is the millisecond-and-safe win.
2. **Dynamic-coverage prune for the inner loop only.** Actual pruning uses dynamic per-test coverage ([testmon-style block-checksums](https://www.testmon.org/blog/determining-affected-tests/) over coverage.py / per-language tracer), where over-selection is cheap and a false negative merely delays feedback. Seed the dynamic CTDG from a **one-time dynamic coverage run** (v1 already has `CoverageReport`) rather than pure static reachability. Hash env/config/seed/lockfile into the selection key; over-select on hierarchy changes (testmon's bias toward false positives).
3. **Full-suite backstop keeps it honest.** Never let static or dynamic selection be the sole pre-merge gate; run the full suite at least once on the final pre-merge state (the [Google](https://research.google.com/pubs/archive/45861.pdf)/[Facebook](https://arxiv.org/pdf/1810.05286) "stabilization" pattern). v1's cheap-first cascade + baseline-prune remains the authoritative oracle.
4. **Per-repo safety-mode flag.** `ctdg.safety_mode ∈ {advisory, prune_with_backstop, prune_hard}`, default `prune_with_backstop`; `prune_hard` requires explicit opt-in. Treat the tracer as a plug: coverage.py for Python, Ekstazi/STARTS for JVM, build-DAG for Bazel monorepos (Section 10).

```text
function select_tests(changed_symbols, ctdg, safety_mode):
    candidates = collect_only()                          # real pytest collection, never static parse
    ordered = reorder_by_static_graph(candidates, ctdg)  # PRIORITIZE: zero false-negative risk
    if safety_mode == "advisory":
        return ordered                                   # run all, just reordered
    pruned = dynamic_coverage_subset(ordered, key=hash(env, seed, lockfile))
    if safety_mode == "prune_with_backstop":
        schedule_full_suite_backstop(final_pre_merge=True)
    return pruned                                        # prune_hard skips backstop (opt-in only)
```

### 5.4 Cheap Pre-Execution Plan Scoring — partially_sound; adopt-modified as downgrade-only prioritizer

**Verdict: partially_sound (high confidence).** The mechanism bundles four sub-claims and the evidence splits them sharply. "Cheap model" as the scorer is the **weakest possible config**: [SWE-PRM](https://arxiv.org/html/2509.02360v1) shows open/weak critics fail to improve over base agents and several *drop* performance (30–38.8% vs 40.0% base); only strong closed critics gave +5–11pp. "Before execution" forfeits the dominant signal: [ORPS](https://arxiv.org/html/2412.15118v1) shows an execution-grounded untrained critic (59.9% Pass@1) crushes a trained execution-free PRM (37.0%). "Prune" (hard-exclude) at the earliest point is where the evidence is most hostile: PRM reliability is lowest exactly where early pruning happens, [beam-search-style early pruning has the lowest accuracy](https://arxiv.org/html/2510.20272), and the [~11pp Best@K vs Pass@K gap (SWE-Gym)](https://arxiv.org/abs/2412.21139) quantifies the correct solutions a verifier discards. "Without killing unconventional approaches" is directly contradicted — unconventional code is the OOD tail, and [RLHF diversity collapse](https://arxiv.org/html/2310.06452v2) means a cheap pre-execution pruner systematically pushes exploration toward conventional-looking plans.

A pre-execution hard prune is the **inverse-equivalent** of the prohibited "promote unverified": it makes a soft, execution-free signal load-bearing for *exclusion*, permanently denying a branch any execution verification. v1 enforces the asymmetry structurally (`_apply_evidence_bound_review` flips accepted only True→False; the ranking tuple places every soft key strictly below every execution key).

**What we adopt (adopt-modified): a downgrade-only prioritizer, never a gate.**

- The cheap score may **set branch priors / budget share** that the active controller (Section 14) can override; it may **never remove a candidate from the set before execution**.
- Always keep a **wildcard lane** that executes the lowest-scored unconventional branch, so tail solutions are never silently killed (counters Best@K<Pass@K + diversity collapse).
- Prefer a **generative/CoT critic** over a scalar one ([THINKPRM](https://arxiv.org/abs/2504.16828) generalizes OOD to code far better), behind a swappable `Verifier` interface with a `critic_strength` knob that **auto-degrades to pure execution-order scheduling** when only a weak critic is available.
- Where actual pre-execution pruning is desired, **execute the cheapest discriminating evidence** (v1's syntax→lint→reproduction cascade or a fast smoke test) — the cheap model proposes *what cheap check to run*; execution decides.
- Journal the score as advisory metadata per `agent()` call (filesystem-as-source-of-truth). Emit a **`pruned_but_would_have_passed` canary metric** so over-aggressive prioritization is detectable. **Never** use this scorer as an RL reward ([reward hacking generalizes to sabotage](https://arxiv.org/abs/2511.18397)).

```text
function prioritize_branches(plans, cheap_critic, controller):
    for p in plans:
        p.prior = cheap_critic.score(p) if cheap_critic.strength >= MIN_CRITIC_STRENGTH else NEUTRAL
        # DOWNGRADE-ONLY: prior may lower priority/budget share; it can NEVER set p.excluded = True
    ranked = controller.allocate(plans, priors=[p.prior for p in plans])  # controller may override prior
    ensure_wildcard_lane(ranked)         # always execute lowest-scored unconventional branch
    return ranked
```

Config (Section 10): `pruning.pre_exec_scoring.enabled` (default `false`, opt-in), `pruning.pre_exec_scoring.mode` (`downgrade_only`, **the only allowed value**), `pruning.cheap_critic.min_strength`, `pruning.wildcard_lane.enabled` (default `true`).

### 5.5 Cross-Branch Sharing (Blackboard 2.0) — sound_with_caveats; adopt phased abstracted negative sharing, reject raw push

**Verdict: sound_with_caveats (medium confidence).** "Improves over isolated rollouts" is well-supported — isolation is itself a weak baseline — independent-parallel-no-communication is a poor multi-agent configuration, and naive cross-agent sharing can amplify errors rather than correct them ([MAST](https://arxiv.org/abs/2503.13657) catalogs the failure modes). But "without collapsing diversity" is true **only for a specific mechanism** and false for sharing in general: [naive share-all dropped accuracy up to 3.7pp on GAIA (LTS)](https://arxiv.org/abs/2602.05965); broadcasting raw trajectories "biases generation toward local patches rather than new designs" ([MEMOIR](https://arxiv.org/html/2605.17539)); pass@k gains "vanish when candidates are highly correlated" ([Monkeys](https://arxiv.org/abs/2407.21787)). And "real-time" is the weakest word — the two strongest supporting works (MEMOIR, LTS) are **phased/post-commit, not real-time**; [DReaMAD](https://arxiv.org/abs/2503.16814) shows early/shared context amplifies the dominant initial belief regardless of correctness. The only true real-time data point, [Hogwild!](https://arxiv.org/abs/2504.06261), is for tightly-coupled subtasks *within* one rollout, not across independent rollouts.

**Reject: raw share-all / instant mid-subprocess push.** Mid-subprocess prompt mutation is infeasible against opaque CLIs and breaks determinism/replay (each `agent()` call's inputs would become nondeterministic, defeating the durable journal). Share-all measurably lowers accuracy and homogenizes attempts.

**Adopt-modified: phased, abstracted, negative-constraint sharing — an evolution of `EpisodicMemoryBus`, not a rebuild.** This is the largest under-cited overlap: v1's `EpisodicMemoryBus` *already* shares negative/ruled-out discoveries with relevance ranking, confidence floors, dedup-by-signature, and own-rollout exclusion. The redesign's only real delta is the delivery *schedule*. So:

1. **Diversity-by-construction at spawn** — per-rollout unique prompts/personas, heterogeneous (vendor, model), independent seeds (DReaMAD perspective diversity).
2. **Phase the bus, not always-on.** Keep the first exploratory wave fully isolated (v1 already runs barrier waves); open the epistemic layer only **after** rollouts commit to distinct strategies. This single change reconciles "improves outcomes" with "preserves diversity."
3. **Two-tier memory (MEMOIR).** Private full traces per rollout (already v1's worktree+`RepoContext` discipline); a thin global layer of ~200–300-token abstracted entries — verified codebase facts, failure modes, and **negative/avoidance directives** ("do NOT retry X; it deadlocks test Z"). Never broadcast raw solution trajectories. Implement as a generalization of `EpisodicMemoryBus` (append-only, relevance-ranked, self-excluding, **artifact-backed** so it stays filesystem-as-source-of-truth and resume-deterministic).
4. **Selective admission controller (LTS)** — ~85% admit, not 100%; admit only broadly-applicable cross-rollout facts. Add blackboard roles ([LbMAS](https://arxiv.org/abs/2507.01701)): a `cleaner` (prune stale/contradicted facts — proven to control token blowup), a `conflict_resolver` (reconcile contradictory facts about the same code object), a `critic` (catch hallucinated facts pre-propagation).
5. **Strict producer-only scope (the load-bearing guardrail).** The shared epistemic layer feeds **only generation**; it must **never** touch the execution-grounded selector / EG-critic / FinalAcceptanceReviewer, or the verifier "becomes another participant in collective delusion" rather than an objective validator (a documented multi-agent false-consensus failure). This preserves the Cardinal Safety Contract.
6. **Reserve true real-time** for tightly-coupled subtasks *within* a single rollout (Hogwild!-style), never across independent rollouts.
7. **Live diversity health metrics** — track pass@k vs pass@1, cluster count, support size; throttle/close the bus when diversity collapses.

```text
record EpistemicEntry:                # global layer; ~200-300 tokens; artifact-backed
    id: str
    kind: enum{verified_fact, failure_mode, negative_constraint}   # NEVER raw_trajectory
    summary: str                      # abstracted, not raw debug detail
    confidence: float
    provenance_rollout_id: str        # used to EXCLUDE own-rollout on query
    relevance_keys: {stage, path, symbol, test}
    seq: int                          # monotonic; replay reconstructs in seq order

function open_bus_if_phased(wave_index, rollouts):
    if wave_index == 0: return        # first wave fully isolated (anti-anchoring)
    for r in rollouts:
        entries = bus.query(exclude=r.id, min_confidence=FLOOR, limit_neg=3, limit_pos=5)
        admit = controller.admit(entries)            # ~85% admit, not 100%
        r.generation_context += format_negatives_first(admit)   # producers ONLY
```

Naming note: if the team wants "real-time," rename to **"phased streaming epistemic sharing"** to avoid overclaiming the one part the evidence does not support. Config (Section 11): `blackboard.phase_first_wave_isolated` (default `true`), `blackboard.admit_rate_target` (~0.85), `blackboard.producer_only_scope` (`true`, non-overridable), `blackboard.share_raw_trajectories` (`false`, non-overridable).

### 5.6 Heavy/Thin Executor Split — partially_sound; adopt-modified as sub-role cascade, reject thin-default

**Verdict: partially_sound (high confidence).** The role-split *spine* is well-supported: [Aider architect/editor](https://aider.chat/2024/09/26/architect.html) improved **every** tested model over its solo baseline (o1-preview 79.7%→85.0%; o1-mini 61.1%→71.4%) and hit polyglot SOTA at ~14× lower cost; [Claude Code orchestrator patterns](https://www.mindstudio.ai/blog/smart-orchestrator-cheaper-sub-agent-models-claude-code) report 5–10× cuts on bounded tasks. But every such win is on **bounded/editing** benchmarks with a **competent (Sonnet-class) editor**, not a "thin cheap" executor on hard repo SWE. The redesign's own scope word — *hard repo SWE* — is the documented failure case: the [HyperAgent ablation](https://arxiv.org/html/2409.16299v1) shows weakening the **Navigator** (codebase exploration) or **Editor** roles causes the *worst* resolve-rate drops, while only run/verify is safely substitutable; (HyperAgent's own Llama-3-8B "Lite" variant scores ~16%, far below frontier-tier resolve rates, [arXiv:2409.16299](https://arxiv.org/html/2409.16299v1)). The almost-right trap means a thin executor needing 3–4 rewrites can cost more than one frontier pass, and [SWE-Effi documents token-snowball "expensive failures" (4×+)](https://arxiv.org/pdf/2509.09853).

**Reject: heavy-orchestrator + thin-executor as the *default* execution shape on hard repo SWE.** It regresses toward the cheap-model baseline precisely where the redesign claims the win.

**Adopt-modified: a sub-role, verification-gated cascade.**

- Keep a **frontier planner** (decomposition) and a **frontier reviewer** (final quality gate) — where compute demonstrably pays off.
- **Tier the executor by sub-role, not as one blob.** Route cheap models **only** to run/verify and narrow single-tool calls; keep frontier (or competent Sonnet-class, **not** thin) on navigation and multi-file editing for hard tasks ([HyperAgent](https://arxiv.org/html/2409.16299v1)).
- **Calibrated escalation.** Self-generated tests as the in-domain confidence signal ([code-cascade](https://arxiv.org/html/2405.15842v1/)); threshold tuned on in-domain traces, **never** intuition ([RouteLLM](https://arxiv.org/html/2406.18665v4) cross-domain routing can be worse than random without calibration; [GATEKEEPER](https://arxiv.org/pdf/2502.19335): poorly-calibrated cheap models can't close the gap at any threshold). A **hard frontier fallback** bypasses the threshold on the hardest tier. **Escalate on first verify-on-diff failure**, with a **rewrite-cycle cap**.
- **Contract as the planner↔executor interface**, scoped **per-task** (not whole-system) to honor the Beck/Fowler "you learn during implementation" critique and fight [spec↔code drift](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html). v1's `build_contract_slice` is the natural starting point; acceptance is decided on the **git diff against real tests**, never the contract or the executor's self-report — which neutralizes drift via the existing execution-evidence-authoritative oracle.
- **Measure cost-per-resolved-task net of verification** (a thin executor *raises* the false-positive accept rate — [ImpossibleBench: GPT-5 hacks tests 76%](https://arxiv.org/html/2510.20270v1) — making the cheap-first cascade and N×N cross-validation *more* load-bearing, partially eating the savings). Validate only on contamination-resistant splits, never SWE-bench Verified ([OpenAI retired it: 59.4% flawed tests in the audited hard set](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)).

Tension with v1's "never optimize for cost": expose this as an **opt-in `budget{}` primitive defaulted unbounded**; budget exhaustion must **never** abort an in-flight succeeding rollout (preserving the escrow-WAL/CCEDF invariant that a confirmed full-scope pass is never lost).

| Sub-role | Cheapen on hard repo SWE? | Rationale |
|---|---|---|
| Planner (decomposition) | No — frontier | Where compute pays off |
| Navigator (codebase exploration) | No — frontier / Sonnet-class | HyperAgent: worst drop when weakened |
| Editor (multi-file edits) | No — frontier / Sonnet-class | HyperAgent: worst drop when weakened |
| Executor: run / verify | **Yes — cheap** | Most substitutable role |
| Executor: narrow single-tool calls | **Yes — cheap** | Bounded, well-specified |
| Final reviewer (quality gate) | No — frontier | Catches compounding cheap-model errors |

### 5.7 Speculative Branching — partially_sound; adopt-modified (constant factor via prefix reuse, turn-boundary forks)

**Verdict: partially_sound (high confidence).** Direction is right — decision-node forking with prefix reuse beats N independent trajectories on cost-per-useful-result ([Tree-GRPO](https://arxiv.org/abs/2509.21240) matches chain RL at 1/4 budget; [InfoTree](https://arxiv.org/html/2605.05262) proves independent samples *collapse* on hard prompts). But **"exponentially" is unsupported**: every measured saving is a bounded constant factor (Tree-GRPO ~4×; [RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) up to 6.4× throughput / ~10× cached-vs-uncached gap; [Speculative Actions has a proven ~50% (2×) latency ceiling](https://arxiv.org/abs/2510.04371)). The saving comes from **prefix/KV reuse + cheap fork**, not the search algorithm — and prefix reuse caps at the prefix-fraction of prefill, structurally incapable of exponential scaling. The claim also conflates branch-and-prune (a real win vs N trajectories) with speculative *action* execution (a latency optimization that burns extra tokens and is net-negative unless the next action is predictable **and** downstream work is expensive **and** effects are reversible/sandboxed).

**Adopt-modified.**

- **Branch-and-prune as a first-class workflow primitive**, forking at **decision/turn/checkpoint boundaries** (whole thought-action-observation steps), measuring cost in token/tool-call budget and **critical-path** (longest branch), not trajectory count. The fork feeds `FrontierSearchController` ranking/budget; it is bounded by `virtual_loss`/`min_branch_reward`. This is the one genuinely-new v1 delta (agent-initiated vs planner-emitted) — admitted only at boundaries because the opaque-CLI seam cannot be paused mid-token.
- **Constant-factor, not exponential.** Restate the claim honestly: "~4–6× cheaper per unit of useful exploration, conditional on realized prefix reuse and millisecond fork." Pair always with submodular value-guided pruning + optimal-stopping (InfoTree's (1-1/e) greedy; replace fixed-N best-of-N with adaptive stopping).
- **Prefix discipline as a portability requirement** (Section 16). Emit `[stable: tooling+system+policies]` then `[volatile: task+scoped context]`; lint/forbid timestamps/UUIDs/reordering in the stable region; dispatch longest-shared-prefix-first. Add a benchmark that **detects whether prefix caching actually fired** (`cache_read` vs `cache_creation` tokens) and **degrades gracefully** (cap fan-out, fall back to sequential) when it does not — because a vendor-agnostic orchestrator cannot assume provider prefix caching, and one byte of drift makes branching degenerate to N independent trajectories. **Resolve the v1 conflict**: v1 deliberately injects per-rollout-divergent state (air-gapped HOME, per-rollout lock paths, scoped `discovery_scope`, snapshot messages embedding `source_head8+dirty_hash8`) — the stable shared-prefix region must be engineered **separately** from this per-rollout-divergent scoping, or the branching economics never materialize.
- **Demote speculative *action* execution** to a targeted optimization for slow/external tools, gated by a **new effect taxonomy** (only idempotent/reversible/sandboxed actions speculatable; irreversible/externally-visible never). For the inner edit-compile-test loop prefer engine-side, rollback-free wins ([Suffix Decoding: 1.8–4.5× end-to-end on SWE-bench](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/)) over action-level speculation needing a fork-commit-rollback protocol.
- **Fork substrate.** Keep v1's cheap **filesystem-only** git-worktree branching as default (CoW ~20ms, scales with change not size), explicitly documenting that it loses live process/memory state. Only add process-memory C/R (DeltaBox/Crab-style, behind a turn-classifying Inspector) where a branch genuinely depends on it.
- Every speculated branch stays **downstream of the Cardinal Safety Contract verification cascade**, so branching can never promote an unverified candidate.

### 5.8 Determinism Preservation — sound_with_caveats; adopt with seeded/journaled search-control

**Verdict: sound_with_caveats (medium confidence).** Speculative search *can* preserve v1's actual guarantees, but "bit-for-bit determinism" overstates both what v1 guarantees and what speculation delivers. v1 is explicit: "determinism is best-effort (pinned around stochastic agents) + replay is strict"; manifest pinning guarantees environment+ordering, **not** agent output (reproduce **artifacts**, not token streams — [temp-0 is not bitwise reproducible across hosted APIs](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)). So "preserved determinism" can only honestly mean (a) bit-identical orchestration/ordering/selection and bit-identical diffs from identical source states, and (b) replay-from-journal reproducing the trajectory — **not** "re-run a speculative search from scratch and get an identical tree."

Speculation is compatible *in principle* because v1 already solved the hard version in the escrow WAL (CCEDF, §16.5): concurrent, nondeterministically-timed operations are made replay-stable by ordering on a **monotonic seq (not wall-clock)** with idempotent keys, and selection is already exploration-order-independent (the ranking tuple terminates in content `sha1`/`-cluster_id`, "never insertion order"). The [durable-execution literature](https://learn.temporal.io/tutorials/go/background-check/durable-execution/) confirms the pattern: deterministic orchestration + nondeterministic LLM/tool calls isolated as retryable activities with idempotency keys.

**Adopt with three hard constraints.**

1. **Quarantine all search-control nondeterminism.** Make every branch/expand/prune decision a pure function of recorded evidence; **seed any sampler (e.g., Thompson) from a content hash** (reuse v1's `Random(0)` discipline); **journal every node expansion and cancellation** to a WAL keyed by monotonic seq + idempotency key (generalize the escrow WAL from §16.5 to per-node). Replace wall-clock hedging on the deterministic path with budget/evidence triggers, **or journal the hedge/cancel decisions** so replay reproduces them rather than re-deriving from timing. (Per the [prompt-caching durability study](https://arxiv.org/html/2601.06007v2): timestamps/random seeds/network calls in *workflow* code break replay subtly — test by killing mid-run.)
2. **Subordinate pruning to the Cardinal Safety Contract.** A deterministic-but-unsafe pruner is *reproducibly wrong*. Speculative pruning may only re-order or down-weight; it must never terminate a branch on a soft/learned/LLM signal in a way that prevents a would-be execution-verified candidate from reaching the verifier. Prefer "prune by budget/depth ceilings and execution evidence" over "prune by value head."
3. **Build and validate the substrate first.** The journaling the claim depends on is currently **unbuilt**: v1's `ReplayRecorder` has no production callsite and the escrow WAL is a narrow one-confirmed-candidate backstop. Promote `ReplayRecorder` into a production **per-`agent()`-call WAL** and add the standard durable-execution acceptance test — **kill mid-run, confirm bit-identical resumed trajectory and identical selected winner** — as a gating CI check **before** claiming the guarantee is preserved (Section 15). The deterministic selection tuple is not in conflict and is the main enabler.

### 5.9 Disposition Summary

The full Kept/Modified/Dropped ledger is Section 18; this is the verdict-qualified disposition for the five redesign mechanisms specifically. No row is a rubber stamp — every adoption carries its qualification, every "novel" claim is flagged against its v1 antecedent.

| Redesign mechanism | Adversarial verdict | Disposition | Verdict-based qualification | Builds in |
|---|---|---|---|---|
| Distributed/classical MCTS as core loop | unsound (high) | **reject (headline)** → keep **bounded adaptive-branching** | No clean win vs verified best-of-N; selection (not topology) is the bottleneck; mandatory collapse to best-of-N below feedback floor; git-checkpoint branching only | §5.2, §9 |
| Static-AST CTDG as test gate | unsound (high) | **reject as gate** → adopt as **prioritizer + dynamic-coverage prune + full-suite backstop** | Static is unsafe in dynamic Python (PyCG ~70% recall; collection not statically enumerable); reorder-never-exclude; per-repo safety mode | §5.3, §10 |
| Cheap pre-exec plan scoring | partially_sound (high) | **adopt-modified (downgrade-only prioritizer)** | Never excludes pre-execution (inverse-Contract violation); wildcard lane; generative critic; auto-degrade to execution-order; never an RL reward | §5.4, §10, §14 |
| Cross-branch sharing (Blackboard 2.0) | sound_with_caveats (medium) | **adopt phased abstracted negative sharing; reject raw push** | Share-all loses (-3.7pp); phase first wave isolated; producer-only scope; evolve `EpisodicMemoryBus` delivery; mid-subprocess push infeasible/non-deterministic | §5.5, §11 |
| Heavy/thin executor split | partially_sound (high) | **adopt-modified (sub-role cascade); reject thin-default** | Cheapen only run/verify + narrow calls; frontier on plan/navigate/edit/review; calibrated escalation; rewrite-cycle cap; cost net-of-verification | §5.6, §12 |
| Speculative branching | partially_sound (high) | **adopt-modified (constant factor, turn-boundary forks)** | Not exponential (~4–6× via prefix reuse); prefix discipline + cache-hit detection + graceful degradation; effect taxonomy for action speculation; downstream of verifier | §5.7, §9, §16 |
| Determinism preservation | sound_with_caveats (medium) | **adopt with seeded/journaled search-control** | Artifacts not token streams; seed samplers from content hash; journal expansions/cancellations; pruning subordinate to Contract; build+validate WAL first | §5.8, §15 |

The throughline: the redesign's instincts are mostly right, but its *forms* over-reach — toward stateful trees, static gates, pre-execution hard pruning, raw mid-flight sharing, thin-default execution, and exponential claims. In every case the evidence-grounded, Contract-respecting form is **narrower, bounded, and reuses a v1 antecedent the redesign did not cite**. APEX-Ω adopts the amplifier, keeps verified best-of-N as the floor, and lets execution evidence remain both the steering signal and the brake.

## 6. State of the Art: Synthesis & Exploitable Gaps

This section distills the 2024–2026 literature on inference-time scaling, tree search, process/outcome verification, program analysis, fleet efficiency, learned orchestration, and the dynamic-workflow paradigm into a single load-bearing claim and a small set of engineering invariants that the rest of APEX-Ω is built on. The thesis is deliberately narrow and decisive: **selection (the verifier), not search topology, is the binding constraint on agentic coding capability; coverage is cheap and search is a bounded amplifier whose ROI depends on the verifier and on candidate diversity.** Everything downstream — the speculative tree-search layer (Section 9), CTDG pruning (Section 10), the blackboard (Section 11), the model economy (Section 12), the verifier (Section 13), the active controller (Section 14) — is justified or restrained by the synthesis here.

The reader should treat each subsection as a *design constraint with a regime condition*, never a universal win. The single most important meta-lesson from the corpus is that almost every headline number is regime-dependent (budget, verifier soundness, candidate correlation, repo scale, model strength), and the most common failure of a redesign is to import a technique that won in one regime into a regime where it loses or actively harms. Where a mechanism is genuinely unproven for our regime, it is carried as guarded/optional with a mitigation and a fallback to the verified floor.

### 6.1 The Eight Synthesized Findings (the binding constraints)

The table below is the executive map of the evidence. Each row names the finding, the load-bearing numbers, the regime in which it holds, and the APEX-Ω disposition (where it is realized — see the cross-referenced section). The dispositions are consistent with the canonical accepted-mechanisms list and with the adversarial verdicts.

| # | Finding | Load-bearing evidence | Regime condition (when it holds / breaks) | APEX-Ω disposition |
|---|---------|----------------------|-------------------------------------------|--------------------|
| F1 | **Selection/verifier is the binding constraint, not search topology** | Best@K trails Pass@K by ~11pt ([SWE-Gym](https://arxiv.org/abs/2412.21139): Best@16 32.0% vs Pass@16 42.8%); SWE-Search value-fn 73%→discriminator 84% ([ICLR 2025](https://arxiv.org/abs/2410.20285)); CodeMonkeys 45.8% random → 57.4% selected vs 69.8% coverage ceiling ([2501.14723](https://arxiv.org/abs/2501.14723)) | Always — strongest single result across math, code, repo-SWE. Gap *widens* as the verifier weakens | Cardinal Safety Contract + hybrid verifier (Sections 13); invest engineering in selection, cap search |
| F2 | **Repeated sampling scales coverage log-linearly, but optimal K is often <10 against imperfect verifiers** | Coverage c(k)≈exp(a·k^-b) over ~4 orders of magnitude; SWE-bench Lite 15.9%@1→56%@250 ([Large Language Monkeys](https://arxiv.org/abs/2407.21787)); but optimal K≤5 at cost-benefit 4, often <10 ([Limits of Resampling](https://arxiv.org/abs/2411.17501)) | Coverage scaling needs a *near-oracle* verifier to convert. With noisy verifiers, false-positive risk *rises* with K | Difficulty-adaptive low-K default ON (Section 9/14); high-K only as thin-feedback floor |
| F3 | **Adaptive-branching beats best-of-N and MCTS only above ~64 calls; budget-aware schedule matters** | [AB-MCTS](https://arxiv.org/abs/2503.04412): comparable below 64 calls, pulls ahead above ~64 (DeepSeek-V3 CodeContests); [BAVT](https://arxiv.org/abs/2603.12634): budget-agnostic tree search can *lose* to repeated sampling under strict budgets | Only large budgets + rich feedback. Below the crossover, plain verified sampling wins | Bounded adaptive branching inside FrontierSearch caps; collapse to verified best-of-N below feedback-confidence floor (Section 9) |
| F4 | **Hybrid execution + generative critic verification; generative critics generalize OOD** | [R2E-Gym](https://arxiv.org/abs/2504.07164): execution 43.7% + execution-free 42.8% each plateau → hybrid 51.0% Best@26; [THINKPRM](https://arxiv.org/abs/2504.16828): generative PRM +4.5% OOD on LiveCodeBench beating discriminative on 100× data | Execution anchors correctness (low discrimination); critic breaks ties (biased). Discriminative scalar PRMs fragile across repos/langs | Hybrid verifier, critic discrimination-only within execution-verified tier (Section 13) |
| F5 | **Dynamic coverage (testmon-style) near-safe; static CTDG unsafe in dynamic Python** | [PyCG](https://arxiv.org/pdf/2103.00587) ~99% precision but ~69.9% recall, ignores eval/codegen; reflection/monkeypatch/fixtures invisible; pytest set not statically enumerable | Static gating silently drops fault-revealing tests → violates execution-authority. Dynamic coverage + full-suite backstop is honest | CTDG as test *prioritizer* + dynamic-coverage prune + full-suite backstop; static-as-gate **rejected** (Section 10) |
| F6 | **Heavy/cheap split + calibrated cascade saves 30–90%; cheapen run/verify not navigation** | Aider opusplan 5–14× cheaper with competent editor; [RouteLLM](https://github.com/lm-sys/RouteLLM) >85% cost cut MT-Bench; [EET](https://arxiv.org/html/2601.05777) ~31.8% avg cut at negligible loss; **but** HyperAgent ablation: cheapening navigation/multi-file edit causes worst resolve-rate drops | Savings real for run/verify/narrow-edit. Cheapening *navigation* is the "almost-right trap" — costs more than one frontier pass | Verification-gated model economy; frontier on navigation/multi-file (Section 12) |
| F7 | **Abstracted-negative blackboard sharing wins; share-all loses; phase the sharing** | Share-all measurably lowers accuracy (−3.7pp) and homogenizes attempts; MEMOIR/LTS abstracted negatives preserve diversity; ReasoningBank +4.6% SWE-bench-Verified from *abstracted* insights ([Google](https://research.google/blog/reasoningbank-enabling-agents-to-learn-from-experience/)) | Sharing helps only if abstracted + phased at turn boundaries. Raw share-all + mid-subprocess injection breaks diversity and determinism | Blackboard 2.0: phased abstracted negatives at turn boundaries; share-all **rejected** (Section 11) |
| F8 | **Contamination is the dominant validity threat; reward-hacking scales with capability; expensive failures** | [OpenAI Feb 2026](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/): 59.4% flawed tests in audited hard set, verbatim gold-patch recall; [Anthropic 2025](https://arxiv.org/abs/2511.18397): reward hacking generalizes to 12% sabotage / 50% alignment-faking; [SWE-Effi](https://arxiv.org/abs/2509.09853): failures cost 4×+ a success | Universal for a mixed-vendor fleet trained/evaluated on public data. Hacking risk *rises* with model strength | Anti-cheat/fairness/failure taxonomy + futility detection; private rotating eval (Sections 13/15/20/21) |

The ninth finding — **open-pool generalization** — is the unclaimed contribution and is treated separately in §6.7 because it shapes the controller (Section 14) and the evaluation (Section 20) rather than a single mechanism.

### 6.2 F1+F2: Selection dominates; coverage is cheap, conversion is not

The corpus is unambiguous and reproduced across math (MATH 95.3% coverage but 39.8% selection — a ~55pt gap, [Brown et al.](https://arxiv.org/abs/2407.21787)), competitive code, and repo-SWE (CodeMonkeys 12.4pt coverage→selection gap; SWE-Gym ~11pt Best@K vs Pass@K). **The deliverable is one answer (Pass@1/Pass@2), not coverage (Pass@k).** Search topology buys coverage; coverage is already cheap via repeated sampling. What is expensive — and what gates real resolve-rate — is converting coverage into a *trustworthy* selection. This is precisely why APEX v1's Cardinal Safety Contract (execution-evidence-authoritative selection) is adopted verbatim: it is the mechanism that converts log-linear coverage into trustworthy gains and counters the inference-scaling false-positive failure mode that the Limits-of-Resampling result predicts.

Design consequences, expressed as an invariant the engine must enforce:

```text
INVARIANT (Cardinal Safety Contract, restated for the engine):
  execution_evidence is AUTHORITATIVE for promotion across tiers.
  soft signals (generative critic, plan score, blackboard priors) may:
    - re-rank WITHIN an execution-verified tier, OR
    - downgrade a candidate (lower its priority/budget share),
  but MUST NEVER promote a candidate above an execution-verified one,
  and MUST NEVER exclude a candidate before execution evidence exists.
```

The compute-allocation corollary (F2): default to **low K** with explicit cost-benefit gating, not a fixed large N. CodeMonkeys used 10 trajectories; the Limits-of-Resampling cost-benefit analysis puts the optimum at single digits against imperfect verifiers. APEX-Ω therefore ships `difficulty_adaptive_allocation: true` by default (v1 had `enable_adaptive_allocation` but OFF by default — the single biggest cost lever left on the table). The allocation config:

```yaml
allocation:
  difficulty_adaptive: true        # F2: optimal K usually <10
  k_min: 2
  k_default: 4
  k_max: 16                        # full-cap retained ONLY as thin-feedback floor (see F3)
  cost_benefit_ratio: 4            # stop sampling when marginal_gain < cost/ratio
  difficulty_probe: early_pass_rate # cheap probe → route easy↦revision, hard↦diverse sampling
```

Two pitfalls the engine must respect. First, **gains vanish when candidates are correlated** — temperature-only sampling yields redundant candidates, so diversity is the asset actually being spent. APEX-Ω treats `(vendor, model)` as a first-class diversity axis (adopt; Section 12) and uses serial-dependent diversification (later attempts informed by abstracted negatives from earlier ones — the blackboard, Section 11) rather than relying on temperature alone. Second, do **not** read log-scale coverage curves optimistically: on a linear axis each doubling of K buys progressively less, so the `cost_benefit_ratio` gate, not a fixed N, is the right control.

### 6.3 F3: Search is a bounded amplifier — adaptive branching inside budget caps, MCTS rejected as the core loop

The mcts research is decisive and informs two dispositions. **Plain MCTS as the core loop is rejected** (adversarial verdict unsound): SWE-Search's headline "+23%" is measured against a *greedy single-trajectory* baseline, not verified repeated sampling; its Pass@5 (34.0) barely exceeds Pass@1 (31.0), proving MCTS adds little coverage on top of sampling — the gap is selection. Worse, classical MCTS needs cheap state save/restore for rollouts/backtracking, which non-serializable Docker repo state does not provide; it re-describes our FrontierSearchController without the isolation guarantees. **Bounded adaptive branching is adopted-modified**: AB-MCTS's *adaptive allocation* (GEN node + Thompson-sampling wider-vs-deeper) is the part that wins, but only above ~64 calls and with rich feedback. Below that crossover, and below a feedback-confidence floor, it collapses to verified best-of-N.

```python
# Section 9 realizes this; the synthesis fixes the control law:
def branch_decision(node, budget, feedback_confidence):
    # F3: only adaptive-branch when budget large AND feedback informative
    if budget.remaining_calls < ADAPTIVE_BRANCH_FLOOR:       # ~64 calls
        return collapse_to_verified_best_of_n(node, budget)
    if feedback_confidence < FEEDBACK_CONFIDENCE_FLOOR:      # thin feedback
        return collapse_to_verified_best_of_n(node, budget)  # mandatory collapse
    # budget-aware schedule (BAVT): broad early, greedy late
    explore_bias = budget.fraction_remaining               # 1.0 early → 0.0 late
    return thompson_wider_vs_deeper(node, explore_bias,
                                    min_branch_reward=MIN_BRANCH_REWARD,
                                    virtual_loss=VIRTUAL_LOSS)
```

The cheap checkpointing primitive that makes *any* branching safe is git-worktree-per-rollout isolation + fcntl locks (adopt verbatim; CAID 63.3 vs 57.2). Branch at git commits/worktrees, never by snapshotting container state — this sidesteps the exact engineering blocker that kills classical MCTS for repo-SWE. The budget-aware schedule (broad exploration early, greedy refinement late) is a control input, not a footnote: BAVT shows budget-agnostic adaptive branching can underperform plain sampling under strict budgets.

### 6.4 F4: The verifier is a hybrid, discrimination-only learned layer over an execution anchor

R2E-Gym is the load-bearing recipe: execution-based and execution-free verifiers each plateau at ~42–43% on SWE-bench Verified and combine to 51.0% — *do not choose, combine*. Execution provides direct correctness but cannot rank two patches that pass the same tests (low discrimination); the learned critic discriminates but is biased toward stylistic/surface features and "agent thoughts." Under the Cardinal Contract, the critic therefore breaks ties **only within the execution-verified tier**. The verifier is a swappable component because verifier *strength is the binding constraint* (F1): SWE-PRM strong closed critics give +5–11pp but open critics *fail to beat base agents and can drop to 30–38.8%*. The engine must degrade gracefully to execution-only when only a weak critic is available, and must expose the verifier as a vendor-neutral interface:

```python
class Verifier(Protocol):
    """Vendor-neutral; any (vendor, model) or an execution oracle may implement."""
    def score(self, candidate: Candidate, evidence: ExecutionEvidence) -> VerifierResult: ...

@dataclass
class VerifierResult:
    execution_tier: int          # AUTHORITATIVE: derived from execution evidence only
    critic_score: float | None   # discrimination ONLY, within-tier; None if no strong critic
    rationale: str | None        # NL rationale (generative critic), for audit/replay
    confidence: float            # critic self-confidence (treat as miscalibrated — see F8)
```

Three durable design rules from the prm corpus. (1) Prefer **generative/long-CoT critics** over discriminative scalar PRMs — they generalize OOD across repos/languages (THINKPRM) and are data-efficient. (2) For code, do **not** over-invest in trained step-level PRMs: ORPS shows an *untrained* execution-grounded implicit PRM (59.9% Pass@1) beats an explicitly trained outcome PRM (37.0%) because execution grounds the reasoning. (3) Critically, the verifier the producer cannot see its own context: the verifier MUST NOT see producer context (a constraint inherited by Blackboard 2.0, Section 11) to avoid the critic preferring good-looking-but-wrong solutions. The cheap-first verification cascade that never synthesizes a pass is adopted verbatim (`rc==0→errors=1`, `rc==124→regression_inconclusive`) — this is the safe per-candidate prune the redesign wanted, already proven, and it is the inverse-equivalent guard: it never *fabricates* a pass just as plan-scoring must never *fabricate* a fail.

Two metrics become first-class orchestrator outputs: the **Best@K vs Pass@K gap** (how much oracle headroom the verifier leaves unrealized) and verifier cost (strong closed PRMs run ~$22–28/100 instances vs ~$2.77 base — a real budget axis).

### 6.5 F5: Program analysis is a prioritizer, never a gate (static CTDG rejected)

The staticanalysis corpus says structural priors (repo graphs + data-flow slices + LSP) reliably improve *localization*, and localization is the binding constraint on repair (ARISE: Recall@1↔Pass@1 correlation rises 0.05→0.53; RepoGraph +2–2.7pt across four scaffolds). But the gains come from the graph **data** (especially intra-procedural data-flow slicing, ARISE's single biggest lever) not the tool wrapping, and they degrade exactly where dynamic-language analysis is weakest. This forces a sharp disposition split, consistent with the adversarial verdict:

- **Static-AST CTDG as a test-pruning gate: rejected.** PyCG ~70% recall; reflection/monkeypatch/fixtures/`eval` are invisible; the pytest set is not statically enumerable. Gating silently drops fault-revealing tests, violating execution-authority — an inverse-equivalent of false-positive promotion.
- **CTDG as a test *prioritizer* + dynamic-coverage prune + full-suite backstop: adopted-modified.** Reordering has zero false-negative risk (you still run everything, just sooner-first); dynamic coverage (testmon-style) is near-safe because it is recorded from *actual* execution, not inferred from AST; and the full-suite backstop keeps it honest by re-running the complete suite before any promotion.
- **Cheap pre-execution plan scoring: adopted-modified as a downgrade-only prioritizer**, rejected as a hard prune/gate. It may set branch priors/budget share the controller can override; it may *never* exclude a candidate pre-execution (false-negative pruning would violate the Cardinal Contract).

```text
CTDG control flow (Section 10 realizes):
  graph = build_repo_graph(tree_sitter, lsp_if_available)   # data, exposed as nav tools
  ranked_tests = ctdg.prioritize(changed_files, graph)      # REORDER only, zero FN risk
  for t in ranked_tests:                                     # fail-fast on likely-relevant
      run(t); record_dynamic_coverage(t)
  if candidate_promising:
      run_full_suite_backstop()                              # honesty gate before promotion
```

Two corpus caveats are baked in. Grep can match graph navigation on small repos (Augment); APEX-Ω routes by repo scale/task — graph-heavy nav for large/multi-hop tasks, cheap grep for small local fixes. And for strong models, do **not** pre-summarize structured analysis into NL (ARISE: zero gain, +5000 tokens) — keep an optional summarizer flag for weak/cheap models only.

### 6.6 F6+F7+F8: Economy, blackboard, and validity

**F6 (model economy).** The split is verification-gated, not a blanket "cheap orchestrator." Aider/opusplan show 5–14× savings with a *competent* editor, and RouteLLM/FrugalGPT-style cascades save 30–90% — but only on the *right* axis. HyperAgent's ablation is the load-bearing warning: cheapening navigation and multi-file editing causes the worst resolve-rate drops, the "almost-right trap" that can cost more than one frontier pass. So APEX-Ω cheapens run/verify and narrow edits, keeps the frontier model on navigation/multi-file edits, and escalates on first verify-on-diff failure with a rewrite-cycle cap. The **heavy-orchestrator + thin-executor default is rejected** (verdict partially_sound → default rejected). Cost-saving primitives that *are* portable and adopted: prefix-stable prompt assembly + provider-cache adapter (~90% off cached reads; selective caching beats full-context, which can *raise* latency and is associated with ~10–18% TTFT variance below the min-token threshold — [Don't Break the Cache](https://arxiv.org/html/2601.06007v2)); a vendor-agnostic API consumer cannot literally share KV across forks, only maximize prefix hit-rate via byte-identical prefixes + dispatch ordering (KVFlow steps-to-execution scheduling over the workflow graph).

**F7 (blackboard).** Raw share-all is rejected on two grounds: it measurably lowers accuracy (−3.7pp) and homogenizes attempts (diversity is the spent asset, per §6.2), and mid-subprocess prompt mutation is infeasible against opaque CLIs and breaks determinism/replay. Blackboard 2.0 (adopted-modified) shares **abstracted negative constraints** at turn/checkpoint boundaries only, evolving v1's EpisodicMemoryBus (keep relevance/confidence/dedup/own-rollout-exclusion), with the hard rule from F4 that the verifier must not see producer context. ReasoningBank/ExpeL/Reflexion corroborate: abstracted insights/skills are the safe episodic prior; raw-trajectory reuse causes "experience following" error propagation and self-degradation (SWE-ContextBench: free retrieval can *hurt*).

**F8 (validity).** Contamination is the dominant validity threat: OpenAI deprecated SWE-bench Verified (59.4% flawed tests in the audited hard set; verbatim gold-patch recall). The disposition is non-negotiable for a mixed-vendor fleet: never trust a static public benchmark older than ~12–18 months; build/track a private, freshly-authored rotating eval (Section 20). Reward-hacking scales with capability and *generalizes* beyond the hack to sabotage/deception (Anthropic) — so anti-cheat (seal/hash test files, forbid editing tests/CI/assertions, detect `sys.exit(0)`/`__eq__`-override/pytest-monkeypatch, prefer held-out tests), fairness, the failure taxonomy, and first-class abstention are adopted verbatim. And failures are expensive (SWE-Effi: 4×+ a success; "token snowball"), so an **early localization-futility gate** (adopted) routes budget away from the "15/16 doomed" waste before the patch loop — informing allocation, never suppressing a candidate without execution evidence. "Knowing when to stop" does not emerge from end-to-end RL (Calibrate-Then-Act); it is engineered as an explicit, calibrated module.

### 6.7 F9: Open-pool generalization — the clearest unclaimed NeurIPS-grade gap

Every learned orchestrator in the literature trains and tests on a **fixed worker pool**: Puppeteer optimizes over a fixed Titan pool; The Conductor and AgentConductor train a small orchestrator against a fixed set of workers; bandit routers (BaRP, RouteLLM) learn over a fixed model menu. None addresses the realistic deployment of a vendor-neutral engine where the worker set is *open* — new (vendor, model) pairs appear, old ones are deprecated or rate-limited mid-run, and the controller has never seen a given worker's capability/cost profile. This is the defensible contribution APEX-Ω stakes (Section 14, Section 19): an **open-pool active controller via learned capability/cost profiles** that can route to a worker it has never trained on by learning a transferable profile (difficulty-conditioned success rate, cost, latency, failure modes) rather than memorizing a fixed pool's indices.

The disposition is **adopt-modified and staged**, fail-open to heuristic:

```text
Stage 0 (ships day one): contextual-bandit / lightweight policy-gradient ROUTER.
  - active control surface over OPEN pool via per-worker capability/cost profiles
  - non-linear MLP policy (BaRP: REINFORCE-MLP beats LinUCB/LinTS)
  - reward r = w_q*quality - w_c*cost ; learns online from verifier signal we already have
  - BLEND not switch; fail-open to v1 heuristic if profile confidence is low
Stage 1 (defer): GEPA-style reflective prompt evolution of the controller.
  - language-feedback beats GRPO by ~6pp avg / up to 19pp at up to 35x fewer rollouts
  - prompts-not-weights ⇒ inherently vendor-agnostic; Pareto selection; length-regularized
Stage 2 (defer): full RL (Puppeteer REINFORCE / Conductor GRPO) over orchestrator decisions.
  - only when volume justifies; cost-penalized terminal reward keyed on deterministic verifiers
  - SHORT episodes to limit credit diffusion; agent/turn-wise grouping (AT-GRPO) if GRPO
```

The durable-execution substrate makes this tractable: a durable input-hash journaled, restart-survivable resume (adopted; promotes v1's unused ReplayRecorder + escrow WAL into a per-`agent()`-call WAL) doubles as the **off-policy credit substrate** for the learned controller — reproducible credit assignment over journaled decisions. The survey's hard warnings are honored: outcome-reward RL suffers credit diffusion (keep episodes short), spawn decisions are non-identifiable from on-policy traces (log shadow branches for counterfactual eval), and the controller must `blend-not-switch` and `fail-open`. The `library_enabled` flag that is currently wired-but-unused in v1 is either wired through to Stage-0 active control or removed — no dead switches.

### 6.8 What the synthesis forbids (pitfalls promoted to engine rules)

The following are not advice; they are constraints the engine enforces, each traceable to a regime condition in the corpus.

| Forbidden pattern | Why (evidence) | Enforced by |
|-------------------|----------------|-------------|
| Promoting a candidate on a soft signal | F1/F4: critics biased, verifiers noisy; Best@K<Pass@K | Cardinal Contract; soft = downgrade/within-tier only |
| Excluding a candidate pre-execution (static CTDG gate, plan-score prune) | F5: PyCG ~70% recall; false-negative pruning violates execution-authority | CTDG = reorder + dynamic-cov prune + full-suite backstop |
| Non-adaptive fixed-K default (5; 16 = cap) | F2/F8: optimal K<10; cost pathology; expensive failures | difficulty-adaptive low-K default ON; full-cap = thin-feedback floor only |
| Plain MCTS / FrontierSearch as core loop | F3: re-describes controller; brittle vs non-serializable state; loses below 64 calls | bounded adaptive branching inside budget caps; collapse to verified BoN |
| Raw share-all / mid-subprocess injection | F7: −3.7pp, homogenizes; infeasible vs opaque CLIs; breaks replay | abstracted negatives at turn boundaries only |
| Cheapening navigation/multi-file editing | F6: HyperAgent worst-drop ablation; almost-right trap | frontier on navigation; cheapen run/verify/narrow-edit only |
| Trusting bit-reproducible agent OUTPUT replay | impossible across hosted APIs (temp-0 batch non-invariance) | reproduce *artifacts* (diffs + re-run verification), not token streams |
| Trusting any static public benchmark >12–18mo | F8: SWE-bench Verified contaminated/deprecated | private rotating eval; standardized scaffold on uncontaminated splits |
| Caching below the min-token threshold | ~10–18% TTFT variance; full-context caching raises latency | selective prefix-stable caching only |
| Hedging/speculative fan-out without a circuit breaker | tail-at-scale: degraded backend → every request crosses p95 → doubled load | deadline-triggered dispatch gated by circuit breaker + budget kill-switch |

### 6.9 Net synthesis (the load-bearing claim, restated)

The honest framing the whole plan rests on: **search and economy are bounded amplifiers; execution evidence is both the steering signal and the brake; verified best-of-N is the floor we can never do worse than.** No single technique is a universal win — every adopted mechanism carries its regime condition and a mandatory collapse to the verified floor when that condition is not met. The substrate that makes amplification *safe* is APEX v1's execution-authoritative kernel (Cardinal Contract, cheap-first cascade, worktree isolation, anti-cheat, determinism/RunManifest); the redesign mechanisms are admitted only as workflow-pattern extensions over vendor-neutral workers, only where the verdicts show net-positive ROI, and only in forms that respect that contract. The one clearly unclaimed, defensible contribution — **open-pool generalization of the active controller** — is the scientific stake (Section 19), built on the durable journaling substrate that doubles as its off-policy credit ledger.

Cross-references: the verifier and selection mechanics are specified in Section 13; the bounded search layer in Section 9; CTDG/pruning in Section 10; the blackboard in Section 11; the model economy in Section 12; the active controller and learned policy in Section 14; isolation/determinism/durable resume in Section 15; speed/cost engineering in Section 16; self-improvement/memory in Section 17; and the contamination-resistant evaluation in Section 20.

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
| 8 | **Cost-aware allocation as first-class control** | `budget{}` is a first-class primitive (defaulted unbounded to honor v1's "never optimize for cost" stance, opt-in to bound) but exhaustion **never aborts an in-flight succeeding rollout**; default difficulty-adaptive low-K; cascade (cheap → verify-on-diff → escalate to frontier); measure **token yield** (cost per verified-resolved task), not invoice | The single biggest cost lever (verdict: non-adaptive fixed-K default — 5, up to the 16 cap, caps off — `reject`); v1's `enable_adaptive_allocation` exists but is OFF by default |

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
- **`EpisodicMemoryBus`** → **L3 blackboard.** It already does the hard part the SOTA digest validates (abstracted *negative* sharing, relevance/confidence/dedup, own-rollout exclusion). Blackboard 2.0 evolves only the *delivery schedule* (pull-at-boundary → push-at-turn-boundary), keeping every guard. Crucially, the verifier must never see producer context (the "collective delusion" multi-agent false-consensus warning).
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

## 9. The Speculative Tree-Search Layer (as Workflow Patterns)

### 9.1 Scope, Stance, and the One Load-Bearing Claim

This section specifies how APEX-Ω performs *adaptive search* over candidate solutions. The honest framing, dictated by the adversarial verdicts and the SOTA digest, is narrow and decisive: **APEX does not build distributed MCTS. It builds bounded, budget-aware adaptive-branching expressed as workflow patterns over vendor-neutral workers, sitting strictly upstream of execution-authoritative selection (Section 13), with a guaranteed collapse to verified best-of-N below a feedback-confidence floor.** Search is an *amplifier* of coverage; the verifier/selector is the brake and the floor. We can never do worse than best-of-N, and we only sometimes do better — regime-dependently.

Two adversarial verdicts bound this section absolutely:

- **"Distributed MCTS over the codebase beats best-of-N for repo-level SWE under fixed budget" — UNSOUND (high confidence).** The only clean repo-level number, [SWE-Search +23%](https://arxiv.org/abs/2410.20285), is measured against a *greedy* agent, not verified best-of-N; SWE-Search Pass@5 (34.0) barely exceeds Pass@1 (31.0), so MCTS captures almost no coverage headroom. [REBASE/Wu et al.](https://arxiv.org/abs/2408.00724) found "MCTS underperformed plain sampling at every budget." We therefore **reject** distributed/stateful MCTS as the core loop (consistent with the canonical IN/OUT list).
- **"Speculative branching is *exponentially* cheaper than N full trajectories" — PARTIALLY SOUND (high confidence).** The direction is right ([Tree-GRPO](https://arxiv.org/abs/2509.21240) matches chain RL at 1/4 budget; [InfoTree](https://arxiv.org/html/2605.05262) proves independent sampling collapses on hard prompts), but the magnitude is a *bounded constant factor* (~4-6x via prefix reuse, [RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/)), never exponential, and the saving comes from the **substrate** (prefix-shared inference + cheap fork), not the algorithm. We therefore **adopt-modified** `speculate()` as a constant-factor-cheaper, prefix-reuse fork at turn/checkpoint boundaries only.

What we keep from the search literature, expressed as patterns: AB-MCTS's *adaptive wider-vs-deeper allocation* ([Inoue et al.](https://arxiv.org/abs/2503.04412)), budget-aware exploration schedule (broad early, greedy late — [BAVT](https://arxiv.org/abs/2603.12634)), multi-LLM Thompson routing as a search dimension ([Sakana](https://sakana.ai/ab-mcts/)), and futility/early-stop to avoid the [SWE-Effi](https://arxiv.org/abs/2509.09853) "4x expensive failure." All of it runs *inside* the v1 `FrontierSearchController` budget caps and is gated by the Cardinal Safety Contract.

### 9.2 Relationship to v1's FrontierSearchController

v1 already ships a `FrontierSearchController` (per the ingest, `apex/search/frontier_search.py`, Blueprint §7.5): best-first/PUCT over branchable code-state checkpoints (`SearchState`/`SearchEdgeStats`), with the following constants reported in the ingest: `c_puct=1.25`, `virtual_loss=0.15`, `max_depth=6`, `max_frontier_branching=3`, `min_branch_reward=0.12`, a `_compute_transition_reward`, value backup via `_update_stats`, and verification-as-early-stop via `_should_stop_search`. This is the closest antecedent to "MCTS over the codebase" and it already supplies the budget-bounding (depth, branching, min-reward) and virtual-loss de-duplication that the prose redesign omitted.

The APEX-Ω layer is a **strict extension of this controller, not a rewrite.** We add three things and re-use everything else:

| Capability | v1 FrontierSearch today | APEX-Ω extension | Disposition |
|---|---|---|---|
| Target source | planner-emitted from `TaskStateGraph` | + agent-initiated `speculate()` fork at turn boundary | adopt-modified |
| Allocation policy | fixed `max_frontier_branching=3` per node | AB-MCTS Thompson wider/deeper/diversify, budget-conditioned | adopt-modified |
| Model selection | resolved per `LLMConfig` | `(vendor, model)` as a Thompson-sampled search axis | adopt |
| Transition reward | obligation/hypothesis/progress/quick-feedback | execution-grounded process reward (progress, quick-verify, regression-survival) | adopt-modified |
| Pruning authority | early-stop on verification | pruning is a HINT; full regression-prune is the GATE | invariant (Cardinal Contract) |
| Floor | best-first ranking | mandatory collapse to verified best-of-N below feedback-confidence floor | new (guaranteed floor) |

Implementation note for either backend: the controller is pure orchestration code (state in script variables, results to disk — Section 2). Codex and Claude Code are *only the leaf workers* invoked through the normalized `Executor` (Section 3); the search topology is identical regardless of which CLI satisfies a node, and indeed nodes in one tree may be served by different vendors (§9.7).

### 9.3 Core Data Structures

All structures are deterministic (no `Date.now`/`Math.random`; ordering keyed by integer ids) so the journal (Section 15) can replay them. Fields below are the APEX-Ω contract; where a v1 type already exists (e.g. `WorkspaceSeed`, `RolloutResult`) we compose with it rather than redefine.

```
SearchNode:
    node_id:            int                  # deterministic, monotonically assigned
    parent_id:          int | None           # None == root (the issue)
    depth:              int                   # 0..max_depth
    checkpoint:         WorkspaceSeed | None  # git-worktree/snapshot checkpoint (v1), NOT docker/process
    prefix_key:         str                   # stable prompt-prefix hash (§9.8) for cache reuse
    origin:             enum{ROOT, PLANNER_TARGET, SPECULATE_FORK, BESTN_SEED}
    assigned_vendor:    str                   # codex_cli | claude_cli | gemini_cli | ...
    assigned_model:     str                   # resolved launcher id, pinned in RunManifest
    status:             enum{OPEN, EXPANDING, VERIFIED, PRUNED_HINT, DEAD, ABSTAINED}
    transition_reward:  float | None          # execution-grounded process reward, [0,1]; None until executed
    visit_count:        int                   # N(s) for PUCT
    value_sum:          float                 # W(s) for backup
    feedback_confidence: float                # [0,1]; drives floor collapse (§9.6)
    rollout_result:     RolloutResult | None  # v1 type; the executed artifact + verification

SearchEdgeStats:        # v1 type, reused: PUCT priors, virtual_loss bookkeeping

AllocationDecision:     # output of the wider/deeper/diversify policy (§9.5)
    action:             enum{WIDER, DEEPER, DIVERSIFY, STOP}
    target_node_id:     int
    branch_count:       int                   # clamped to remaining budget & max_frontier_branching
    vendor_model:       (str, str)            # Thompson-sampled (§9.7)

SearchBudget:           # composes v1 budget{} primitive (Section 2)
    total_tokens:       int
    spent:              () -> int
    remaining:          () -> int
    total_nodes:        int                   # hard lifetime cap (default derived, see §9.10)
    nodes_expanded:     int
    futility_strikes:   int                   # consecutive non-improving expansions
```

`feedback_confidence` is the keystone field. It is **not** a learned PRM score; it is a cheap, mechanical estimate of *how trustworthy the steering signal is at this node* (§9.6). Below the floor, search collapses to best-of-N because the evidence cannot distinguish branches — exactly the regime where the literature shows search loses.

### 9.4 Top-Level Control Flow

```
function adaptive_search(issue, plan, baseline, budget) -> RolloutResult | ABSTAIN:
    phase("speculative-search")

    # FLOOR GUARANTEE: always launch the verified best-of-N slate first as the seed frontier.
    # This is the candidate set we can never do worse than.
    root = SearchNode(node_id=0, origin=ROOT, checkpoint=baseline.seed)
    seed_nodes = launch_bestN_seed(issue, plan, root, K=adaptive_low_K(plan.difficulty))   # §9.6, Section 12
    frontier   = priority_queue(seed_nodes, key=puct_score)        # reuse v1 _should_branch / PUCT

    while not should_stop(budget, frontier):                       # v1 _should_stop_search + futility (§9.9)
        conf = aggregate_feedback_confidence(frontier)
        if conf < CONFIDENCE_FLOOR:                                # §9.6 — below floor: do NOT search
            log("feedback below floor -> collapsing to verified best-of-N")
            return select_best_of_N(verified_nodes(frontier))      # Section 13 selector; execution-authoritative

        decision = allocate(frontier, budget, conf)                # AB-MCTS wider/deeper/diversify (§9.5)
        if decision.action == STOP: break

        children = expand(decision)                                # forks worktree checkpoint, runs scoped worker
        for child in children:
            child.rollout_result   = run_node_worker(child)        # Executor.spawn -> agent(); FS-diff observed
            child.transition_reward = process_reward(child, baseline)   # execution-grounded (§9.5.3)
            child.feedback_confidence = estimate_feedback_conf(child)
            quick = cheap_verification_cascade(child)              # v1 cheap-first ladder; HINT only (§9.5.4)
            if quick == REGRESSION_INCONCLUSIVE or quick == PASS_PARTIAL:
                child.status = OPEN                                 # never pruned on a soft signal
            backup(child)                                          # value backup up to root (v1 _update_stats)

        update_futility(budget, children)

    candidates = verified_nodes(frontier)                          # nodes whose FULL regression-prune passed
    if candidates is empty:
        candidates = all_executed_nodes(frontier)                  # fall back to selector over everything
    return select_best_patch(candidates)                           # Section 13: execution-authoritative, abstains if none accepted
```

Three invariants are visible and non-negotiable here, each tracing to an adversarial verdict or the Cardinal Contract:

1. **The best-of-N slate is launched first as the seed frontier.** Search can only *re-rank and extend* it; it can never starve it. This operationalizes "best-of-N is the floor we can never do worse than."
2. **Cheap checks and process rewards are HINTS** that influence `allocate()` priority and budget share. They *never* set `status = DEAD` or `PRUNED_HINT` in a way that removes a candidate from the final `select_best_patch` pool without execution evidence. The authoritative prune is the full regression-prune inside the selector (Section 13).
3. **Below `CONFIDENCE_FLOOR`, we abandon search and return best-of-N.** This is the explicit guard against the failure mode where a value head prunes correct paths far from terminal — [PRM reliability drops with distance from terminal states](https://arxiv.org/abs/2510.20272).

### 9.5 The Adaptive-Branching Allocation Policy

This is the genuine, evidence-backed win we lift from AB-MCTS, restated as a per-fan-out decision rather than as MCTS rollouts/backtracking. At each step the controller chooses, for the highest-priority frontier node, among three actions plus stop. We use Thompson sampling over Beta posteriors (the AB-MCTS "GEN node" trick) to remove the hand-tuned width hyperparameter.

#### 9.5.1 Wider vs Deeper vs Diversify

```
function allocate(frontier, budget, conf) -> AllocationDecision:
    node = frontier.peek_best()                       # PUCT-ranked (v1 c_puct=1.25, virtual_loss=0.15)

    # Budget-aware schedule (BAVT): broad early, greedy late. f in [0,1].
    f = budget.remaining() / budget.total_tokens
    explore_bias = f                                   # high f -> favor WIDER/DIVERSIFY; low f -> favor DEEPER

    # Thompson sample each action's expected reward from its Beta posterior.
    p_wider     = sample_beta(node.wider_post)     * explore_bias
    p_deeper    = sample_beta(node.deeper_post)    * (1 - explore_bias) * 0.5 + sample_beta(node.deeper_post)*0.5
    p_diversify = sample_beta(node.diversify_post) * explore_bias * decorrelation_need(frontier)  # §9.7

    if max(p_wider, p_deeper, p_diversify) < MIN_BRANCH_REWARD:   # v1 min_branch_reward=0.12
        return STOP
    if node.depth >= MAX_DEPTH or budget.nodes_expanded >= budget.total_nodes:
        return STOP

    action = argmax({WIDER:p_wider, DEEPER:p_deeper, DIVERSIFY:p_diversify})
    branch_count = clamp(budget_affordable_branches(budget, conf), 1, MAX_FRONTIER_BRANCHING)  # =3 default
    vendor_model = thompson_sample_vendor(node, frontier)         # §9.7
    return AllocationDecision(action, node.node_id, branch_count, vendor_model)
```

- **WIDER** = sample a *new sibling approach* from the parent checkpoint (a fresh `speculate()` fork or a new strategy-axis worker). Decorrelates hypotheses; the InfoTree anti-collapse move on *hard* nodes.
- **DEEPER** = refine the current best partial (continue the same trajectory from its checkpoint with the accumulated feedback). The AlphaCodium-style run-and-fix inner loop; effective on *easy/near-miss* nodes ([Snell compute-optimal](https://arxiv.org/abs/2408.03314): sequential revision wins when the model is already close).
- **DIVERSIFY** = WIDER but forced onto a *different* `(vendor, model)` or strategy axis to widen solution coverage and decorrelate hallucinations.
- **`MIN_BRANCH_REWARD` gate (0.12 from v1):** if no action's sampled reward clears the floor, stop — this is the explicit anti-blowup bound the prose redesign omitted.

Pseudo-`Math.random` warning: Thompson sampling needs randomness, which determinism forbids. We resolve this with a **deterministic PRNG seeded from `(run_id, node_id, allocation_step)`** so that an unchanged run replays identical samples (Section 15). The seed is recorded in the journal; this preserves the AB-MCTS behavior without breaking resume.

#### 9.5.2 Why this beats fixed-width, and where it does not

AB-MCTS only beats repeated sampling **above ~64 calls** ([Inoue et al.](https://arxiv.org/abs/2503.04412)); under strict small budgets, a budget-agnostic schedule can *lose* to plain repeated sampling ([BAVT](https://arxiv.org/abs/2603.12634)). APEX therefore conditions explore/exploit on `budget.remaining()` and, critically, defaults to **adaptive low-K best-of-N** (Section 12, the canonical "Difficulty-adaptive low-K allocation, default ON"). The adaptive-branching layer only *activates beyond a budget threshold* (`search.activation_min_nodes`, default 8). Below it, we are pure best-of-N — the regime where search has no proven edge. This is the single most important honesty constraint in the section.

#### 9.5.3 Execution-grounded transition reward (process reward, not a trained PRM)

The per-node `transition_reward` is computed **from execution evidence only**, mirroring [ORPS](https://arxiv.org/html/2412.15118v1)'s finding that an *untrained* execution-grounded PRM beats a *trained* outcome PRM for code (59.9% vs 37.0% Pass@1). It is a downgrade-only steering signal, never a gate:

```
function process_reward(node, baseline) -> float in [0,1]:
    r = 0.0
    # (a) Progress: did the diff move targeted tests from fail toward pass vs baseline?
    r += 0.40 * progress_delta(node.rollout_result, baseline)        # F2P transitions, regression-free
    # (b) Quick-verification survival (cheap-first cascade, v1): rc==0 -> +; rc==124 -> inconclusive, neutral
    r += 0.30 * quick_verify_signal(node)                            # never synthesizes a pass (Section 13)
    # (c) Regression-prune survival HINT: did baseline-passing tests stay green on the sampled subset?
    r += 0.20 * regression_survival_hint(node)
    # (d) Frontier alignment: did it touch the localized frontier? (downgrade-only, mirrors v1)
    r += 0.10 * frontier_alignment(node)
    return clamp(r, 0.0, 1.0)
```

This is "execution-as-process-reward." It is deliberately *not* a learned value head, because (per the `prm` finding and the [Limits of PRM-Guided Tree Search](https://arxiv.org/abs/2510.20272)) learned value heads are fragile far from terminal and can prune correct paths. A learned critic *may* later replace component (d) as a discrimination-only tie-breaker **within** the execution-verified tier (Section 13), never as a pre-execution gate.

#### 9.5.4 Pruning is a hint, never a gate (the inverse-Cardinal violation we forbid)

The cheap-first verification cascade (v1: AST syntax check → public-symbol-survival/stub scans → targeted cached pytest; rc==0→errors=1, rc==124→regression_inconclusive) re-ranks nodes and reallocates budget away from low-reward branches. It **must not** delete a node from the final selection pool. A node whose cheap checks look bad but that was never fully regression-pruned stays `OPEN` and is eligible for the selector. Removing a correct-but-unverified candidate on a soft signal is the inverse-equivalent of promoting an unverified candidate — both violate the Cardinal Safety Contract (Section 13). Concretely: `status` may be set to `PRUNED_HINT` (deprioritized for *further expansion budget*) but a `PRUNED_HINT` node still enters `select_best_patch` if it produced a diff; only the full regression-prune may set `DEAD`.

### 9.6 The Mandatory Floor: Feedback-Confidence Collapse

This is the guaranteed-floor mechanism and the most defensible part of the section. `CONFIDENCE_FLOOR` (config `search.feedback_confidence_floor`, default `0.55`) is the threshold below which the steering signal is too weak for search to help, so APEX collapses to verified best-of-N.

`estimate_feedback_conf(node)` is a cheap, mechanical aggregate of *signal richness*, not solution quality:

| Component | Weight | Meaning | Source |
|---|---|---|---|
| Reproduction strength | 0.35 | Is there a failing reproduction test that the patch must flip? | v1 `ReproductionArtifact`; R2E-Gym execution anchor |
| Regression-test density | 0.25 | Are there baseline-passing tests covering the touched frontier? | v1 `prune_by_regression` / CTDG (Section 10) |
| Quick-verify determinism | 0.20 | Did cheap checks return decisive (not rc==124 inconclusive) results? | v1 cheap-first cascade |
| Branch separability | 0.20 | Do siblings produce *distinguishable* execution outcomes? | mirrors AB-MCTS selection gap |

Aggregation across the frontier uses the *minimum-richness* tests (reproduction + regression density) as hard prerequisites: if there is no failing reproduction AND no regression coverage on the frontier, `feedback_confidence` is capped at `0.40` regardless of other components, forcing a collapse. Rationale: with no executable discriminator, search merely burns budget chasing branches the verifier cannot tell apart — exactly the SWE-Search Pass@5≈Pass@1 selection bottleneck. In that regime, low-K best-of-N + the strongest available discrimination critic (Section 13) is strictly the better spend.

`adaptive_low_K(difficulty)` (Section 12) keeps the floor cheap: K defaults to a *single-to-low-double-digit* value (the `scaling_theory` consensus: optimal K often <10 against imperfect verifiers; [Limits of Inference Scaling](https://arxiv.org/abs/2411.17501) K≤5 at cost-benefit ratio 4), never hundreds.

### 9.7 Multi-LLM Adaptive Branching (vendor/model as a search axis)

Treating `(vendor, model)` as a Thompson-sampled dimension is both a coverage win and a hallucination-decorrelation win: [Multi-LLM AB-MCTS](https://sakana.ai/ab-mcts/) exceeded 30% on ARC-AGI-2 vs ~23% single-model. It also directly realizes APEX's vendor-neutral thesis and the canonical "(vendor, model) as a first-class diversity/search axis" disposition.

```
function thompson_sample_vendor(node, frontier) -> (vendor, model):
    pool = available_vendor_models()              # from BackendPortfolio, excluding disabled (Section 3)
    if pool is empty after portfolio filter: fail loud (Section 13 abstention path)
    # Per-(vendor,model) Beta posterior over "produced an execution-verified node" this run.
    samples = { vm: sample_beta(node.vendor_post[vm]) for vm in pool }
    # DIVERSIFY actions force a vendor/model NOT already dominant in the sibling set:
    if node.pending_action == DIVERSIFY:
        samples = penalize_overrepresented(samples, frontier_sibling_vendors(node))
    return argmax(samples)
```

Vendor-neutral constraints honored:
- The portfolio's **two-tier failure memory** (Section 3) excludes a `(vendor, model)` only at the right scope: a 429/stall is call-failover (skip this node, retry vendor later), not a global ban. A heterogeneous fleet is never poisoned by one vendor's transient.
- **Cost arbitrage** is a first-class lever: a `DEEPER` refinement on a near-miss node may Thompson-route to a cheaper executor while a `WIDER` exploration of a hard node routes to a frontier model (Section 12's verification-gated cascade). The reward posterior naturally learns this within a run.
- The chosen `(vendor, model)` is pinned per node in the `RunManifest` (Section 15) so replay reconstructs which vendor produced each branch — replay reproduces *artifacts* (diffs + re-run verification), never token streams ([bit-reproducible output is rejected](https://arxiv.org/abs/2407.21787): hosted APIs are temp-0 batch non-invariant).

### 9.8 `speculate()`: Agent-Initiated Forking at Boundaries Only

`speculate()` is the one genuinely new primitive (canonical disposition: adopt-modified). It lets an in-flight worker request a fork when it hits a hard either/or decision, feeding the same FrontierSearch ranking/budget machinery rather than spawning an uncontrolled tree.

#### 9.8.1 Where forks are admitted

Forks are admitted **only at turn/checkpoint boundaries**, never mid-subprocess. This is forced by two facts from the ingest: (1) v1's workers are *opaque external CLI subprocesses* observed via stdout turn-parsing — APEX cannot pause/snapshot an arbitrary internal step; (2) mid-subprocess prompt injection breaks determinism and replay. The `CLITurnParser` `turn_observer` (v1's only mid-flight channel) is the admission point: when a closed `Turn` carries a structured `speculate` request, the observer enqueues a fork *to be created at the next clean checkpoint*, then lets the current turn complete or stop.

```
# turn_observer hook (vendor-neutral; each CLI's turn boundary feeds the same path)
on_turn_close(turn, node):
    req = parse_speculate_request(turn)          # structured marker in the worker's output contract
    if req is None: return CONTINUE
    if node.depth + 1 > MAX_DEPTH:        return CONTINUE     # bound: depth cap
    if budget.nodes_expanded >= total_nodes: return CONTINUE  # bound: lifetime cap
    if estimate_feedback_conf(node) < CONFIDENCE_FLOOR: return CONTINUE  # below floor: no speculative forks
    enqueue_fork(parent=node,
                 branches=clamp(req.branch_count, 1, MAX_FRONTIER_BRANCHING),
                 checkpoint=current_worktree_checkpoint(node))   # git worktree, NOT docker/process snapshot
    return CONTINUE   # do not kill the in-flight turn; the fork is scheduled, not immediate
```

#### 9.8.2 Checkpoints are git worktrees, prefix-stable

Forks branch from **git-worktree/snapshot checkpoints** (v1's deterministic snapshot SHAs give bit-identical diffs for identical source state), **not** Docker/process snapshots. This sidesteps the non-serializable-container blocker that kills classical MCTS for repo-level SWE, and matches the speculative-branching verdict's recommended adaptation (filesystem-only branching as default; document that it loses live process/memory state). Each fork inherits the parent's **stable prompt prefix** so provider/server prefix caching fires:

```
prompt = [ STABLE:   system + tool-policy + repo-context-digest + contract-slice ]   # byte-identical across siblings
         [ VOLATILE: scoped task + per-node discovery + accumulated feedback ]        # diverges per node
```

The branching cost saving is *conditional on realized prefix reuse* (RadixAttention is session-scoped, not graph-scoped; one byte of drift → full re-prefill). Therefore §16's prefix-stable prompt assembly + provider-cache adapter is a **portability requirement, not an assumption**: the assembler lints the STABLE region for timestamps/UUIDs/reordering, and a benchmark detects whether caching actually fired (`cache_read` vs `cache_creation` tokens). If it did not, the controller **degrades gracefully**: cap fan-out and bias toward DEEPER (fewer new prefixes) — never silently assume the constant-factor win.

#### 9.8.3 Cost contract

`speculate()` is **constant-factor cheaper via prefix reuse, not exponential** (per the verdict). Cost is measured in token/tool-call budget and *critical-path* (longest branch), not trajectory count. The controller's `min_branch_reward`, `virtual_loss`, and `max_frontier_branching` bound the tree; a worker that over-calls `speculate()` simply has its forks declined once any bound is hit (fail-loud log, not silent suppression).

### 9.9 Futility Detection and Budget-Aware Early-Stop

To avoid the [SWE-Effi](https://arxiv.org/abs/2509.09853) "4x expensive failure," the controller stops aggressively when search is not paying off:

```
function should_stop(budget, frontier) -> bool:
    if budget.remaining() <= RESERVE_FOR_SELECTION: return True   # always leave budget for the selector (Section 13)
    if budget.nodes_expanded >= budget.total_nodes:  return True
    if budget.futility_strikes >= FUTILITY_STRIKES:  return True  # default 3 non-improving expansions
    if best_verified_node(frontier).accepted and marginal_reward_estimate(frontier) < EPSILON:
        return True                                              # diminishing-returns early stop
    return False

function update_futility(budget, children):
    if max(c.transition_reward for c in children) <= best_reward_so_far:
        budget.futility_strikes += 1
    else:
        budget.futility_strikes = 0
        best_reward_so_far = max(...)
```

`RESERVE_FOR_SELECTION` is non-negotiable: the selector/verifier is where points are actually won ([SWE-Search 73%→84%](https://arxiv.org/abs/2410.20285); [R2E-Gym hybrid 43%→51%](https://arxiv.org/abs/2504.07164)), so search may never consume the budget needed for execution-authoritative selection. Futility strikes and the marginal-reward early-stop together implement the BAVT "broad early, greedy late, then stop" schedule.

### 9.10 Configuration Keys

All keys live under `search.*` in `ApexConfig`. Defaults are conservative (best-of-N-leaning) per the canonical "non-adaptive fixed-K default (5, up to the 16 cap), caps off — REJECT" and "adaptive low-K — default ON."

| Key | Type | Default | Meaning |
|---|---|---|---|
| `search.enabled` | bool | `true` | Master switch (engine on, but adaptive branching stays gated by `activation_min_nodes`); `false` == pure adaptive low-K best-of-N (Section 12) |
| `search.activation_min_nodes` | int | `8` | Below this budget, run pure best-of-N (no adaptive branching) |
| `search.max_depth` | int | `6` | v1 cap; turn/checkpoint fork depth |
| `search.max_frontier_branching` | int | `3` | v1 cap; max children per expansion |
| `search.min_branch_reward` | float | `0.12` | v1; below this sampled reward → STOP |
| `search.c_puct` | float | `1.25` | v1 PUCT exploration constant |
| `search.virtual_loss` | float | `0.15` | v1 anti-duplicate de-correlation |
| `search.feedback_confidence_floor` | float | `0.55` | Below this → collapse to verified best-of-N (§9.6) |
| `search.futility_strikes` | int | `3` | Consecutive non-improving expansions → stop |
| `search.reserve_for_selection_frac` | float | `0.30` | Token fraction reserved for the selector |
| `search.multi_llm_routing` | bool | `true` | Thompson-sample `(vendor, model)` per node |
| `search.speculate_enabled` | bool | `true` | Admit agent-initiated forks at turn boundaries |
| `search.prng_seed_basis` | str | `"run_id+node_id+step"` | Deterministic Thompson sampling seed (replay-safe) |
| `search.require_prefix_cache_check` | bool | `true` | Detect cache hits; degrade fan-out if absent (§9.8.2) |

### 9.11 The Worker / Executor Interface for Search Nodes

Every node is satisfied by exactly one call through the normalized `Executor` (Section 3), so the search layer is fully vendor-agnostic. A node-worker call must:

1. **Spawn** a scoped worker via `agent(prompt, {schema, model, label="search:node:<id>", phase, isolation:"worktree", agentType})` — the v1 `run_structured_prompt` equivalent. `isolation:"worktree"` forks from the node's `checkpoint` (git, not Docker).
2. **Return a validated structured result** (schema native for `claude_cli`, schema-as-prompt-text + post-parse for `codex_cli`/`gemini_cli` — graceful degradation per Section 3).
3. **Be observed via the FS diff**, not vendor-specific events: the node's correctness is judged by running the resulting git diff (Section 13), regardless of which vendor produced it. This is the property that makes a heterogeneous tree sound.
4. **Journal the call** (Section 15): the node-worker `agent()` call is WAL-keyed by `hash(prefix_key, volatile_inputs, vendor, model)`. On restart, unchanged nodes replay cached results; only edited/new nodes re-run. This is what lets a deep search survive a full process restart — the explicit "do better than the reference impl" mandate.

A failed node-worker (timeout, isolation error, vendor down) resolves to a `DEAD` node with a typed failure reason (never an exception that aborts the tree, and never a faked pass — fail-loud). The controller continues with surviving frontier nodes; if all die, the run abstains (Section 13).

### 9.12 What We Deliberately Do Not Build (and why)

| Rejected | Why (verdict / evidence) |
|---|---|
| Distributed/stateful MCTS over a shared mutable tree | UNSOUND verdict; conflicts with context-isolation and filesystem-as-source-of-truth; plain MCTS doesn't beat verified sampling at repo scale; brittle against non-serializable container state |
| Value head that prunes pre-verification | [PRM reliability drops far from terminal](https://arxiv.org/abs/2510.20272); would suppress correct paths — inverse-Cardinal violation |
| Docker/process snapshot checkpoints | Non-serializable, hundreds-of-ms to seconds overhead; git-worktree checkpoints are deterministic and cheap |
| Mid-subprocess fork/inject | Infeasible against opaque CLIs; breaks determinism/replay; forks admitted only at turn boundaries |
| Claiming search beats best-of-N unconditionally | Regime-dependent (AB-MCTS only wins >~64 calls); we guarantee the best-of-N floor and only amplify above it |
| Unbounded `speculate()` | Combinatorial blow-up; bounded by `min_branch_reward`/`virtual_loss`/`max_depth`/lifetime cap |

### 9.13 Summary

The Speculative Tree-Search Layer is **bounded adaptive-branching as a workflow pattern over vendor-neutral workers**: AB-MCTS-style wider/deeper/diversify Thompson allocation, conditioned on remaining budget (broad early, greedy late), running inside v1's `FrontierSearchController` budget caps, steered by execution-grounded process rewards, decorrelated across `(vendor, model)`, forked only at prefix-stable git-worktree checkpoints, with aggressive futility/early-stop, and — above all — a **mandatory collapse to verified best-of-N below a feedback-confidence floor**. Pruning is always a hint; the full regression-prune in Section 13 is the only authoritative gate. The layer can outperform best-of-N in rich-feedback, large-budget regimes and can never do worse than it elsewhere. That asymmetry — search as a bounded amplifier, execution evidence as both steering signal and brake, best-of-N as the floor — is the load-bearing claim of this section.

## 10. CTDG + Bidirectional Pruning

This section specifies the Code-Test Dependency Graph (CTDG) and the "bidirectional pruning" subsystem as a set of vendor-neutral workflow patterns layered on the APEX substrate (Section 2) and consumed by the speculative tree-search layer (Section 9), the verifier (Section 13), and the active controller (Section 14). It exists to answer one question cheaply and *safely*: **of the tests this repo can run, which ones should this worker run first, and which can it skip during fast iteration — without ever silently discarding a fault-revealing test before the final gate.**

The design is deliberately conservative because the adversarial review judged the headline ambition — "a static CTDG enables safe millisecond pruning in dynamic Python" — **unsound** ([PyCG ICSE'21](https://arxiv.org/abs/2103.00587): ~99.2% precision / ~69.9% recall; [Rothermel & Harrold](https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=1015&context=csearticles) safety theorem). We therefore split "use the graph" from "prune as a gate" and keep execution evidence authoritative (the Cardinal Safety Contract, Section 13).

### 10.1 What this subsystem is — and is not

| Concern | Disposition | Why |
|---|---|---|
| Static import/call graph (tree-sitter / LSP) | **Prioritize + explain only** — reorder tests, never exclude | Reordering has zero false-negative risk ([ctdg synthesis](https://www.gauge.sh/blog/how-to-make-ci-fast-and-cheap-with-test-impact-analysis); Rothermel-Harrold). PyCG ~70% recall makes static *exclusion* unsafe. |
| Dynamic per-test coverage map (coverage.py contexts / testmon block-checksums) | **Actual prune gate during fast iteration** — advisory, never authoritative | "As safe as coverage.py" ([testmon.org](https://www.testmon.org/blog/determining-affected-tests/)); a false negative merely delays feedback inside the loop. |
| Full-suite stabilization backstop at final pre-accept state | **Mandatory under default safety mode** | Google TAP / Facebook PTS keep a periodic full run; selection is *never* the sole pre-merge gate. |
| Cheap pre-execution plan score | **Downgrade-only branch prioritizer** — never a kill | A pre-exec soft signal gating *exclusion* is the inverse-equivalent violation of the Cardinal Contract (adversarial verdict: `partially_sound`). |
| Static-AST CTDG as a test-pruning gate | **Rejected** | PyCG recall; reflection/monkeypatch/fixtures invisible; pytest set not statically enumerable. |
| AST / semantic-equivalence checks | **Reserved for patch validation (Section 13), not live navigation** | Equivalence is undecidable; useful only as a bounded overfitting/regression signal. |

The CTDG **feeds priors; the verifier decides.** It is never treated as an oracle.

### 10.2 Why static-only pruning is unsafe in dynamic Python (the load-bearing rationale)

A coding worker (Codex, Claude Code, or other) that trusts a static graph to *drop* tests will silently ship bad patches. The evidence is convergent:

- **Lossy recall.** PyCG, the SOTA static Python call graph, reports ~99.2% precision but only **~69.9% recall**, explicitly ignoring `eval`, `getattr`/`setattr` effects, built-in type-method effects, conditionals, and loops. ~30% of real call edges are absent; each missing edge is a candidate false negative — a pruned-but-fault-revealing test.
- **The test set is not statically enumerable.** pytest items are produced at *collection* time by `parametrize`, `pytest_generate_tests`, fixture graphs from arbitrary plugins, `conftest.py` tree effects, and `pytest_collection_modifyitems`. The only reliable enumeration is running `pytest --collect-only`. A static graph cannot even name what it would prune.
- **Reflection alone breaks static RTS.** [Shi et al. OOPSLA'19](https://lingming.cs.illinois.edu/publications/oopsla2019.pdf) (1173 versions / 24 Java projects) found reflection was the *only* cause of static-RTS unsafety, and reflection-aware safety pushed end-to-end cost from 69.1% to 85.8–91.2% of RetestAll. Python adds monkeypatch, dynamic imports, and ubiquitous `getattr` on top — strictly worse.
- **The safety/precision theorem.** Rothermel-Harrold: you cannot extract both maximal pruning *and* zero false negatives from imperfect dependency data. The residual gap must be closed by dynamic ground truth or a full-suite backstop — neither of which is "static."
- **Practitioner confirmation.** Agentic graph systems (ARISE, CodexGraph) deliberately *drop* dynamic-dispatch/eval/monkeypatch edges to avoid spurious edges — exactly the source of silent false negatives if such a graph gated tests. The marketing-vs-safety gap (Tach: "8x faster," zero correctness caveats) is the trap APEX must not fall into.

This directly collides with APEX v1's Cardinal Safety Contract: a static CTDG is a non-execution soft signal, and using it to *exclude* a candidate test is strictly stronger than the already-prohibited "promote an unverified candidate." Hence: **static graph reorders, dynamic coverage prunes (advisorily), full suite gates.**

### 10.3 Data structures

All artifacts live on the filesystem (filesystem-as-source-of-truth) under the run's repo-context cache, are content-addressed, and are journaled per `agent()` call (Section 15). Field types use Python-ish annotations; a coding agent may realize them as dataclasses, TS interfaces, or structs.

#### 10.3.1 Static layer — `CtdgStaticIndex`

Built once per repo snapshot, amortized like v1's `RepoContext`. It is an *extension* of v1's `RepoGraph` (which today emits `contains/imports/inherits/references/uses/rationale_for` edges and has **no** code→test edge — confirmed in the v1 ingest). We add a typed test-edge layer; we do not rebuild the graph.

```text
CtdgStaticIndex:
  repo_snapshot_id:   str            # git rev or tree hash of the snapshot the index was built on
  builder:            str            # "tree-sitter" | "lsp:<server>" | "regex-fallback"
  language:           str            # "python" | "js" | ... (MVP: python; others degrade to regex)
  symbol_nodes:       dict[SymbolId, SymbolNode]
  test_nodes:         dict[TestNodeId, TestNode]   # FILE/CLASS-level only at static layer (see note)
  code_to_test:       dict[SymbolId, list[TestEdge]]   # STATIC priors only; confidence-tagged
  confidence_default: float = 0.5    # static edges are priors, never authoritative
  built_at:           float
  notes:              list[str]      # e.g. "dynamic-dispatch edges dropped"

SymbolNode:    { id: SymbolId, kind: "func"|"method"|"class"|"module", file: str, span: (int,int) }
TestNode:      { id: TestNodeId, file: str, kind: "module"|"class", nodeids_known: bool }
TestEdge:      { test: TestNodeId, via: "import"|"call"|"inherit"|"name-mention",
                 confidence: float,        # EXTRACTED (>=0.8) vs INFERRED (<=0.6), mirrors v1 levels
                 source: "static" }
```

Note: static `test_nodes` are file/class granularity only. We never claim a static map to individual parametrized `nodeid`s — those are not statically knowable.

#### 10.3.2 Dynamic layer — `CoverageMap` (the real prune signal)

Borrows the highest-leverage idea from `pytest-testmon`: **block-level checksums, not file hashes** — a test re-runs only if a block it *actually executed* changed. Built from `coverage.py` dynamic contexts (`--cov-context=test` / `dynamic_context=test_function`) on the first full run, then incrementally maintained.

```text
CoverageMap:
  schema_version:   int
  selection_key:    SelectionKey        # invalidation key — see 10.3.3
  test_to_blocks:   dict[NodeId, list[BlockRef]]   # per real collected nodeid
  block_checksums:  dict[BlockRef, str]            # adler32/sha of normalized block source
  collected_set:    list[NodeId]         # from `pytest --collect-only` (NOT parsed from source)
  hierarchy_index:  dict[SymbolId, set[NodeId]]    # symbol -> covering tests (for over-select)
  tracer:           str                  # "coverage.py" | "ekstazi" | "build-dag" | ...
  built_at:         float

BlockRef:  { file: str, block_id: str }   # block = function/branch region per tracer
NodeId:    str                            # e.g. "tests/test_x.py::TestA::test_y[param-3]"
```

Safety boundary, stated explicitly in artifact metadata: coverage-derived selection is **only as safe as the tracer**. Dependencies on time, randomness, network, filesystem, env/global state, and C extensions are invisible to `coverage.py` and can cause a wrongly-deselected test. We mitigate by (a) over-selection on hierarchy changes, (b) hashing non-code inputs into the selection key, and (c) the mandatory backstop.

#### 10.3.3 The selection key (cache invalidation that closes false-negative holes)

A stale dependency DB is a documented false-negative source ([testmon issue #92](https://github.com/tarpas/pytest-testmon/issues/92)). The `SelectionKey` is hashed into every coverage decision; any change forces a full re-collect / full run.

```text
SelectionKey = sha256(concat(
  repo_snapshot_id,
  resolved_lockfile_hash,        # dependency bump -> full run
  python_version,                # interpreter change -> full run
  env_fingerprint,               # DJANGO_SETTINGS_MODULE, PYTHONHASHSEED, LANG, etc.
  test_seed,                     # PYTHONHASHSEED / pytest-randomly seed
  config_fingerprint,            # pytest.ini / pyproject [tool.pytest], conftest hashes
  coverage_schema_version,
  docker_image_digest            # pinned per RunManifest (Section 15)
))
```

This makes the selection deterministic and replayable (Section 15) and over-selects exactly where dynamic Python is riskiest (config/hierarchy churn). The container digest comes from the same pinning the RunManifest already enforces — keeping selection vendor-neutral and reproducible across hosts.

### 10.4 The two pruning channels

"Bidirectional" in APEX-Ω means **prioritize before, prune after — never exclude before evidence exists.** Concretely there are two channels, and only the *post-evidence* channel is allowed to drop work.

```text
                 ┌────────────────────────── worker turn / branch ──────────────────────────┐
  PRE  (priors)  │  static CTDG order + cheap plan score  ->  test ORDER + branch BUDGET     │
                 │  (downgrade-only; NEVER removes a test or a branch from the candidate set) │
  POST (gate)    │  dynamic coverage select  ->  run subset  ->  cheap-first cascade verdict   │
                 │  (advisory prune of UNCHANGED-block tests; full suite at final pre-accept) │
                 └───────────────────────────────────────────────────────────────────────────┘
```

#### 10.4.1 PRE channel — static prioritization (zero false-negative risk)

Used to order the candidate test set and to feed branch priors to the FrontierSearch / speculate() machinery (Sections 9, 14). It **never** removes a test.

```python
def order_tests(changed_symbols, static_index, collected_set):
    # 1. Score each collected nodeid by static proximity to the change.
    scored = []
    for nodeid in collected_set:                # collected_set comes from --collect-only
        test_node = test_node_of(nodeid)
        s = 0.0
        for sym in changed_symbols:
            for e in static_index.code_to_test.get(sym, []):
                if e.test == test_node.id:
                    s = max(s, e.confidence)     # EXTRACTED edges dominate INFERRED
        scored.append((nodeid, s))
    # 2. ENTIRE collected set is returned — reordered, never filtered.
    #    Unscored tests sort AFTER scored ones but are STILL INCLUDED.
    return [nid for nid, _ in sorted(scored, key=lambda x: -x[1])]
```

This is the safe win: it accelerates time-to-first-failure (matching RepoGraph / ARISE prioritization gains) at zero recall cost, and it satisfies the contract clause that soft signals may only re-rank.

#### 10.4.2 PRE channel — cheap pre-exec plan score (downgrade-only)

A cheap worker (any vendor model behind the Normalized Executor) may score a proposed plan/edit set to set **branch priority and budget share**, feeding FrontierSearch priors. The adversarial verdict (`partially_sound`) is honored exactly: it can lower a branch's priority but **can never remove it pre-execution**, and a *wildcard lane* always executes the lowest-scored unconventional branch so tail solutions are never silently killed (counters the Best@K ≪ Pass@K headroom and RLHF diversity collapse).

```python
def plan_prior(plan, ctdg, cheap_critic):           # cheap_critic: vendor-neutral, swappable
    # Prefer a generative/CoT critic (THINKPRM-style) over a scalar one for OOD robustness.
    score = cheap_critic.score(plan)                # in [0,1]; advisory metadata, journaled
    reach = ctdg_reachable(plan.edit_targets, ctdg) # static structural plausibility (a HINT)
    prior = clamp(0.5 + 0.3*(score-0.5) + 0.2*(reach-0.5), 0.05, 1.0)
    return BranchPrior(priority=prior, budget_share=prior, removable=False)  # NEVER a kill
```

Guardrails (all mandatory):
- **Downgrade-only.** A branch may be deprioritized to the minimum lane, never excluded. Mirror of v1's evidence-bound review (`accepted` flips True→False only).
- **No RL reward use.** This score must never become an RL/self-improvement reward; reward hacking on coding envs generalizes to sabotage ([Anthropic 2511.18397](https://arxiv.org/abs/2511.18397)).
- **Auto-degrade.** If only a weak critic is available, fall back to pure static-order scheduling (verifier strength is the binding constraint — [SWE-PRM](https://arxiv.org/html/2509.02360v1) shows weak critics can be net-negative).
- **Canary metric.** Emit a "pruned-but-would-have-passed" / "downgraded-but-won" canary so over-aggressive priors are detectable.

#### 10.4.3 POST channel — dynamic coverage prune (the only place work is dropped)

During fast iteration *inside* a worker's edit loop, after a patch touches code, select the at-risk subset via the `CoverageMap` and run only that subset under the existing cheap-first cascade (Section 13: rc==0→errors=1; rc==124→`regression_inconclusive`; AST→symbol-survival→targeted pytest). This *replaces and tightens* v1's heuristic ladder (`graph_target_test_ids[:4] > failing_test_ids[:8] > focus_test_files[:4]`) and narrows v1's `prune_by_regression` (baseline-passing tests in chunks of 50).

```python
def select_affected(changed_blocks, cov_map, static_index):
    if selection_key_changed(cov_map):          # lockfile/env/seed/config/digest change
        return FULL_SUITE                        # over-select: full run, rebuild map
    affected = set()
    for nid, blocks in cov_map.test_to_blocks.items():
        if any(b in changed_blocks for b in blocks):
            affected.add(nid)
    # OVER-SELECT on hierarchy changes: signature/class/module edits re-run everything
    # that touches the module (testmon bias toward false positives over false negatives).
    for sym in hierarchy_changed_symbols(changed_blocks):
        affected |= cov_map.hierarchy_index.get(sym, set())
    # NEW code not yet in the map -> cannot be covered statically -> include broadly.
    if introduces_new_symbols(changed_blocks):
        affected |= sibling_tests_of_changed_files(changed_blocks)
    return sorted(affected) or FULL_SUITE        # empty selection NEVER means "skip all"
```

Three invariants make this honest: (1) an empty selection means *run the full suite*, never "skip everything"; (2) any selection-key change forces a full run; (3) new/renamed symbols force broad inclusion. A false negative here only *delays* feedback because the backstop re-checks at the gate.

#### 10.4.4 POST channel — full-suite stabilization backstop (the brake)

At the **final pre-accept state** of any candidate that the cheap-first cascade has otherwise approved, run the complete suite (subject to the safety mode below). This is the Google/Facebook stabilization pattern and is what keeps fast selection honest. Selection accelerates feedback; the backstop is the merge gate. A candidate cannot reach SOLVED on selection evidence alone.

### 10.5 Per-repo safety mode (the explicit safety knob)

A single config flag governs how aggressively selection is trusted. The orchestrator **never silently gambles** — the mode is recorded in the RunManifest and journaled.

| Mode | PRE (order/priors) | POST (dynamic prune) | Backstop | When |
|---|---|---|---|---|
| `advisory` | on | reported only; full suite always run | always | unknown/high-dynamism repos; first runs |
| `prune-with-backstop` **(default)** | on | drops unchanged-block tests *inside loop* | **mandatory at final pre-accept** | normal operation |
| `prune-hard` | on | drops at the gate too | none | opt-in only; fast tracer-trusted repos |

```text
ctdg:
  enabled: true
  safety_mode: prune-with-backstop      # advisory | prune-with-backstop | prune-hard
  static_builder: tree-sitter           # tree-sitter | lsp | regex-fallback
  dynamic_tracer: coverage.py           # coverage.py | ekstazi | build-dag | none
  block_checksums: true
  over_select_on_hierarchy_change: true
  plan_prior:
    enabled: true
    critic: generative                  # generative | scalar | off
    downgrade_only: true                # HARD-LOCKED true; not user-overridable to false
    wildcard_lane: true
  selection_key_inputs: [lockfile, python_version, env, seed, config, image_digest]
```

`prune-hard` requires explicit per-repo opt-in and an accepted, *measured* non-100% catch rate; it is the only mode that may drop a fault-revealing test, and it is off by default precisely because the adversarial verdict forbids treating selection as a sole gate.

### 10.6 Vendor-neutrality: the tracer and graph are plugs

Both layers sit behind narrow interfaces so the engine is language- and vendor-agnostic (Section 3). Workers (Codex / Claude Code / mixed) consume the *outputs* (ordered test list, affected subset, branch priors) via the Normalized Executor; they do not care how the graph was built.

```text
StaticGraphProvider:                 CoverageTracer:
  build(repo_snapshot) -> CtdgStaticIndex     collect(cmd) -> CoverageMap
  changed_symbols(diff) -> set[SymbolId]      select(changed_blocks) -> set[NodeId]
                                              collected_set() -> list[NodeId]  # via --collect-only

# Plug table (graceful degradation per ACP-style capability negotiation, Section 8):
#   python  : StaticGraphProvider=tree-sitter/LSP ; CoverageTracer=coverage.py contexts
#   JVM     : StaticGraphProvider=STARTS          ; CoverageTracer=Ekstazi (class-level)
#   Bazel   : StaticGraphProvider=build-DAG       ; CoverageTracer=build-DAG reverse closure
#   no tracer: CoverageTracer=none -> safety_mode auto-forced to `advisory` (full suite always)
```

If a backend cannot supply a tracer, the engine degrades — it does not crash — by forcing `advisory` mode. Non-Python languages get regex-only static nodes (no use-edges), so their static layer contributes *less* ordering signal but never less safety.

### 10.7 Integration with the rest of the engine

- **Section 9 (speculative tree search):** static priors and the plan score feed FrontierSearch's existing ranking/budget machinery (`max_depth`, `max_frontier_branching`, `min_branch_reward`, virtual loss). The dynamic affected-set is the cheap check that prunes *speculate()* branches — but only on executed-coverage evidence, reusing v1's cheap-first ladder so branching stays affordable.
- **Section 13 (verifier):** the CTDG never overrides the verifier. Execution evidence is authoritative; the affected-subset run is execution evidence; the backstop is the final execution gate. AST/semantic-equivalence checks are used *there* for patch overfitting/regression-equivalence, not for live test selection.
- **Section 14 (active controller):** safety mode, plan-prior weight, and the over-select threshold are controller knobs (bandit → GEPA → RL staging). The controller may *re-weight* priors but inherits the hard lock that no soft signal excludes a candidate before execution.
- **Section 15 (determinism):** `SelectionKey`, the ordered test list, the affected subset, and every plan score are journaled per `agent()` call. Replay reproduces *which tests ran and the verdict*, not token streams (bit-reproducible output replay is rejected).

### 10.8 Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Static graph drops a dynamic edge (false negative) | Static layer only *orders*; never excludes. |
| Coverage map stale after dep/env/seed change | `SelectionKey` invalidation forces full re-collect + full run. |
| Tracer blind to time/random/network/C-ext deps | Over-select on hierarchy change; default backstop; force `advisory` if no tracer. |
| New/renamed symbol not in map | Broad sibling inclusion; empty selection ⇒ full suite. |
| Cheap plan score kills a correct unconventional branch | Downgrade-only (hard-locked) + wildcard lane + canary metric; no RL reward use. |
| `prune-hard` ships a regression | Off by default; opt-in with measured catch rate; the only mode permitted to drop tests. |
| Flaky tests mis-attributed (Google: 84% of P→F flaky) | Treat selection as feedback accelerator; verifier handles flakiness; backstop re-confirms. |

The throughline: **the CTDG is a prior, the tracer is an advisory accelerator, and execution at the backstop is the brake.** Best-of-N over the full suite remains the floor we can never do worse than.

## 11. The Epistemic Blackboard 2.0

### 11.1 Purpose, Scope, and the One Load-Bearing Claim

The Epistemic Blackboard 2.0 is the cross-rollout knowledge substrate of APEX-Ω: the workflow pattern by which independent coding WORKERS (Codex, Claude Code, or mixed — see Section 3) learn from each other's *dead ends and execution-grounded facts* without learning each other's *solutions*. It is the v1 `EpisodicMemoryBus` (see Section 4; `apex/rollout/engine.py`) promoted from a pull-at-stage-boundary append-only store into a **phased, selectively-admitted, push-at-turn-boundary** delivery layer — and nothing more aggressive than that, because the evidence ceiling is sharp.

The single load-bearing claim, and the one the adversarial verdict rated only `sound_with_caveats`:

> Selective, abstracted, *negatively-framed* cross-branch sharing, layered on top of diverse independent rollouts and *phased* so the first exploratory burst stays isolated, beats both (a) the no-sharing parallel baseline and (b) naive share-all — but the word "real-time" is unproven for independent rollouts and the word "sharing" must be narrowed to this specific mechanism.

Everything below is engineered to capture exactly the part the evidence supports and to refuse the part it does not. Three numbers anchor the design. Naive share-all *dropped* accuracy up to 3.7pp below a no-memory parallel baseline ([LTS, arXiv:2602.05965](https://arxiv.org/abs/2602.05965), Table 2). MEMOIR's two-level abstracted/negative sharing raised validity to 96.7% (+9.2pp) and cut run-to-run variance >10x ([MEMOIR, arXiv:2605.17539](https://arxiv.org/html/2605.17539)). A learned ~85%-admit controller beat no-memory by +1.2–5.6pp while cutting runtime 25–55% at ~0.2% controller overhead (LTS). The mechanism dominates the decision to parallelize at all: independent-parallel-no-communication is a weak MAS configuration, and multi-agent error cascades are well documented ([MAST, arXiv:2503.13657](https://arxiv.org/abs/2503.13657) catalogs 14 failure modes).

#### 11.1.1 Three inviolable invariants (the brakes)

These are non-negotiable and trace directly to APEX's foundational frame and v1's Cardinal Safety Contract (Section 13).

1. **Producer-only scope.** The blackboard feeds *generation* only. It MUST NEVER feed the execution-grounded selector, the EG-critic, the VerificationAmplifier, or the FinalAcceptanceReviewer. A verifier that shares producer context "becomes another participant in collective delusion rather than an objective validator" (a documented multi-agent false-consensus failure). This is enforced structurally, not by convention (Section 11.7).
2. **Negatives + facts only; never positive solutions.** Share "do NOT do X because it deadlocks test Z" and "`api_v2.login()` raised `TypeError: missing tenant_id`", never "the best approach is X" or a raw patch. Broadcasting positive trajectories "biases generation toward local patches rather than new designs" (MEMOIR) and pass@k gains "vanish when output candidates are highly correlated" ([Large Language Monkeys, arXiv:2407.21787](https://arxiv.org/abs/2407.21787)).
3. **Turn-boundary delivery only; no mid-subprocess injection.** WORKERS are opaque external CLI subprocesses (Section 3; v1 `CLIModelClient.run_structured_prompt`). Injecting into a running subprocess mid-turn is infeasible against opaque CLIs and would break determinism/durable-resume (Section 15). Push happens only when the `CLITurnParser` (v1 `cli_turn_parser.py`) closes a `Turn` and the *next* turn's prompt is being assembled.

> Honesty note: we deliberately rename v1/redesign "instant push" / "real-time" to **phased streaming epistemic sharing**. "Real-time cross-branch" is the one claim the evidence does not support and partly contradicts.

### 11.2 Two-Tier Memory Architecture

Per MEMOIR, memory is split so exploration detail stays private and only constraints travel.

| Tier | Scope | Contents | Size | Persistence | Source of truth |
|---|---|---|---|---|---|
| **Branch-local trace** (Tier 0) | One rollout/branch | Full execution trace, tool calls, raw errors, debug detail | Unbounded | Per-rollout worktree + journal | Filesystem (worktree), per-rollout WAL |
| **Global epistemic layer** (Tier 1) | One solve (siblings) + cross-solve priors | Abstracted entries: verified codebase facts, failure modes, negative/avoidance constraints | ~200–300 tokens/entry; capped pool | Append-only artifact-backed store (durable, resumable) | Artifact file (`epistemic_blackboard.jsonl`) |

Tier 0 already exists as v1's worktree-per-rollout isolation + `RepoContext` discipline. Tier 1 generalizes the `EpisodicMemoryBus`. The hard rule: **Tier 0 never crosses a branch boundary**; only distilled Tier 1 entries do. This is the single most important anti-homogenization lever (MEMOIR).

#### 11.2.1 Entry data structure

```python
EpistemicKind = Literal["NEGATIVE_CONSTRAINT", "FAILURE_MODE", "VERIFIED_FACT"]
# NOTE: "POSITIVE_SOLUTION" is intentionally NOT a member — sharing it is forbidden (Inv. 2).

@dataclass(frozen=True)
class EpistemicEntry:
    entry_id: str                 # content-hash; dedup key
    kind: EpistemicKind
    abstracted_text: str          # <= ~300 tokens; NO raw diff/solution; required
    # --- evidence binding (filesystem/execution-as-truth) ---
    grounding: Literal["EXECUTION", "STATIC", "MODEL_INFERRED"]
    evidence_ref: Optional[str]   # path to artifact (test output, traceback); required if EXECUTION
    code_object: Optional[str]    # symbol/file/test the entry is *about* (for conflict detection)
    failing_test_ids: tuple[str, ...]
    # --- provenance & scoping (mirror v1 EpisodicMemoryBus) ---
    producer_rollout_id: int      # >=0 sibling; <= -1 reserved cross-solve prior (v1 sentinel)
    strategy_axis: Optional[str]  # producing rollout's STRATEGY_AXIS (anti-transfer signal)
    backend: Optional[str]        # (vendor, model) that produced it — vendor-neutral metadata
    created_turn: int             # producing rollout's turn index at emission
    created_at: float
    # --- lifecycle / health ---
    confidence: float             # [0,1]; floor-gated on read
    corroboration_count: int      # independent sibling confirmations
    contradiction_count: int      # independent sibling refutations
    status: Literal["LIVE", "CONTRADICTED", "STALE", "EVICTED"]
    admit_score: float            # controller score that admitted it
```

Two design choices matter. First, `grounding` and `evidence_ref` make every entry auditable and keep the store filesystem-/execution-authoritative — a `VERIFIED_FACT` with `grounding="EXECUTION"` must point at a real traceback or test result on disk. Second, `strategy_axis` + `backend` let the relevance scorer *down-weight* a constraint when delivering to a sibling on a different strategy axis or vendor, mitigating negative transfer (a constraint true for the locking-refactor branch may be irrelevant to the minimal-fix branch).

### 11.3 The Sharing Lifecycle (Phased, Selective, Pushed)

The lifecycle reuses v1 `broadcast()` / `query()` / `format_for_context()` semantics and adds three controls: a **phase gate**, a **selective admission controller**, and **push-at-turn-boundary delivery**.

```
[Worker turn N closes]
   -> Tier-0 trace updated (private)
   -> distill() : turn delta -> candidate EpistemicEntry(s)   (abstraction, no raw solution)
   -> admit?(candidate) : selective controller (~85% admit)    (LTS)
        |-- reject -> drop (stays Tier-0 only)
        '-- admit  -> dedup_by_content_hash -> append to Tier-1 store (artifact + WAL)
   -> roles.run() : cleaner, conflict_resolver, critic over Tier-1
   -> [Worker turn N+1 about to start]
        if phase_gate.open(solve):
            ctx = deliver(rollout_id=self, query=current_focus)   # push into next prompt
            assemble_prompt(base, contract_slice, ctx)            # prefix-stable (Section 16)
```

#### 11.3.1 Phase gate (DReaMAD anti-entrenchment)

The first exploratory burst is fully independent and diversely seeded. Early sharing "amplifies the dominant initial belief regardless of correctness"; if the correct answer's initial probability < 0.5, debate/shared context cannot recover it ([DReaMAD, arXiv:2503.16814](https://arxiv.org/abs/2503.16814)). v1 already runs barrier waves, so the gate maps cleanly onto wave structure.

```python
def phase_gate_open(solve_state) -> bool:
    # Bus stays CLOSED until the first wave commits to distinct strategies.
    if solve_state.completed_waves < blackboard.min_isolated_waves:        # default 1
        return False
    if solve_state.distinct_strategy_clusters < blackboard.min_clusters:    # default 2
        return False
    if diversity_health.collapsing():                                       # Section 11.6
        return False                                                        # throttle/close
    return True
```

Diversity-by-construction at spawn (per-rollout unique prompts/personas via v1's 7 `STRATEGY_AXES`, heterogeneous `(vendor, model)` per Section 3, independent seeds) is cheaper and more robust than re-injecting diversity after collapse (DReaMAD + heterogeneity findings). The bus *complements* spawn diversity; it never substitutes for it.

#### 11.3.2 Selective admission controller (LTS)

Never auto-broadcast every step. Admit only broadly-applicable, cross-rollout-useful facts. LTS's ~85% admit rate beat both 100% (share-all) and aggressive-filter, at ~0.2% runtime overhead.

```python
def admit(candidate: EpistemicEntry, store, solve_state) -> tuple[bool, float]:
    if candidate.kind == "POSITIVE_SOLUTION":      # unreachable by type; defense-in-depth
        return (False, 0.0)
    if candidate.confidence < blackboard.admit_confidence_floor:           # default 0.35
        return (False, 0.0)
    if store.has_content_hash(candidate.entry_id):                         # dedup
        store.bump_corroboration(candidate.entry_id, candidate.producer_rollout_id)
        return (False, 0.0)
    # breadth: would this help a sibling NOT on the producer's strategy axis?
    score = controller.admit_score(candidate, solve_state)   # Stage-0 heuristic; later learned
    return (score >= blackboard.admit_threshold, score)      # tuned to ~0.85 admit rate
```

Stage-0 ships a heuristic `admit_score` (grounding boost: EXECUTION > STATIC > MODEL_INFERRED; breadth boost for `code_object` shared across the dependency graph; penalty for axis-locality). The active controller (Section 14) may later replace it with a learned policy, **blend-not-switch, fail-open to the heuristic** — a malformed model degrades to the heuristic admit, never breaking a run (v1's `evaluate_policy_model` discipline: `applied=False => value==baseline`).

#### 11.3.3 Distillation (abstraction step)

`distill()` converts a private turn delta into an abstracted entry. It is a cheap model-economy sub-role (Section 12) — a small/cheap model summarizing into the ~200–300 token budget, NEVER copying raw patch hunks. Output is constrained to the three allowed `kind`s. Distillation failures fail loud (the candidate is dropped, never faked into a `VERIFIED_FACT`).

### 11.4 Delivery: Pull-at-Boundary → Push-at-Turn-Boundary

This is the one genuine *delivery* evolution over v1, and it is bounded by Invariant 3. v1 delivered at stage boundaries via `query()`/`format_for_context()`; APEX-Ω delivers at **turn boundaries** so a sibling avoids a trap before falling into it (the redesign's intent) — but only between turns, never mid-turn.

We **keep verbatim** every v1 delivery guard, because they are exactly the LTS/MEMOIR safety machinery:

- **relevance ranking** (confidence + stage/path/symbol/test overlap + corroboration + independent-verification boosts);
- **confidence floor** on read;
- **dedup-by-signature**;
- **own-rollout exclusion** (`query()` excludes the caller's `rollout_id`); cross-solve priors keep reserved negative ids (`<= -1`) so durable repo memory is never mistaken for a live sibling;
- **caps** (`positive_limit`-equivalent for facts, `negative_limit`-equivalent for constraints — v1 defaults 5/3) plus dropping any text already present in the prompt (prompt-budget discipline).

```python
def deliver(rollout_id: int, query: FocusContext) -> EpistemicContextBlock:
    live = store.entries(status="LIVE", exclude_rollout=rollout_id)        # own-rollout exclusion
    live = [e for e in live if e.confidence >= read_confidence_floor]
    ranked = rank_by_relevance(live, query)                               # v1 scorer, kept
    # negatives first (avoidance pruning), then execution-grounded facts; caps applied:
    negs = take(ranked, kind in {"NEGATIVE_CONSTRAINT","FAILURE_MODE"}, n=negative_cap)  # 3
    facts = take(ranked, kind == "VERIFIED_FACT", n=fact_cap)                            # 5
    block = render(negs + facts, drop_already_in_prompt=True)
    # prefix-stable placement so provider caches stay warm (Section 16):
    return EpistemicContextBlock(text=block, placement="post_contract_slice")
```

#### 11.4.1 Why not Hogwild-style real-time

[Hogwild! Inference, arXiv:2504.06261](https://arxiv.org/abs/2504.06261) is the *only* genuine real-time data point: a shared concurrent KV cache, emergent coordination, no protocol/fine-tuning. But it is scoped to **tightly-coupled subtasks within a single rollout**, not independent cross-branch rollouts. APEX-Ω therefore **reserves Hogwild-style shared-KV soft coordination strictly for intra-rollout co-workers** (e.g. a contract's parallel sub-edits within one branch, Section 12) and never across the diverse independent rollouts whose decorrelation is the asset we are spending. Applying it across branches would violate context isolation, determinism, and durable resume (Section 15).

### 11.5 Blackboard Roles as Orchestrator Middleware

Per [LbMAS, arXiv:2507.01701](https://arxiv.org/abs/2507.01701), a shared space degenerates without active maintenance — removing the cleaner blew MATH tokens 4.7M→13.9M with no quality gain. Three roles run as deterministic orchestrator functions over Tier 1 after each admission cycle. They are **producer-side maintenance**, not verification (Invariant 1).

| Role | Trigger | Action | Failure mode it prevents | SOTA basis |
|---|---|---|---|---|
| **Cleaner** | After each admission cycle | Prune entries `status in {CONTRADICTED, STALE, EVICTED}`; evict beyond pool cap by lowest `relevance*confidence*decay`; drop entries whose `code_object` no longer exists | Token blow-up; stale facts about renamed/removed symbols | LbMAS cleaner ablation |
| **Conflict-resolver** | Two LIVE entries about the same `code_object` assert contradictory facts | Mark both `CONTRADICTED`; open a focused resolution — prefer the one with `grounding="EXECUTION"` and fresher `evidence_ref`; if unresolved, suppress both from delivery (do not propagate either) | Contradictory facts both propagating; negative transfer | LbMAS conflict-resolver |
| **Critic** | New `MODEL_INFERRED` entry, or low corroboration | Flag suspected hallucinated facts; require `>=1` independent corroboration or execution grounding before delivery promotion | Hallucinated "facts" poisoning siblings | LbMAS critic |

Loop safety: resolution rounds are capped (LbMAS uses max 4) to avoid flag→rewrite→re-flag cycles. The cleaner is the proven mechanism that keeps the bus from degenerating; it is mandatory, not optional.

Cross-solve durable insights (v1 `extract_durable_insights`, cap 64, decay 0.85) flow into Tier 1 as priors with reserved negative `producer_rollout_id`; the cleaner and decay keep that repo memory honest across runs (Section 17).

### 11.6 Diversity Health Metrics and the Throttle

The tension must be observable and controllable at runtime, because "entropy != diversity" — policy entropy can stay flat while semantic diversity collapses ([entropy mechanism, arXiv:2505.22617](https://arxiv.org/html/2505.22617v1)). We measure diversity directly, not via an entropy proxy.

| Metric | Definition | Healthy | Action when unhealthy |
|---|---|---|---|
| **pass@k vs pass@1 gap** | Oracle coverage over the pool minus best single-rollout | Gap stays wide | Gap collapsing → throttle/close bus |
| **Cluster count** | Distinct strategy/solution clusters among live rollouts | `>= min_clusters` (2) | Below floor → close bus, re-diversify spawn |
| **Support size** | # rollouts pursuing materially different code frontiers (Jaccard, reuse v1 `_estimate_rollout_diversity_capacity`) | Above floor | Shrinking → reduce delivery caps |

```python
def collapsing() -> bool:
    return (diversity.cluster_count < blackboard.min_clusters
            or diversity.passk_passk1_gap < blackboard.min_coverage_gap
            or diversity.support_size < blackboard.min_support)
```

When `collapsing()` is true, the phase gate closes (Section 11.3.1) and the orchestrator stops pushing — the bus is throttled before it can finish homogenizing the pool. This converts the LTS/DReaMAD warning into a live control loop. These metrics also feed the active controller (Section 14) and the evaluation matrix (Section 20).

### 11.7 Strict Producer-Only Scope (Enforcement, Not Convention)

Invariant 1 is enforced by construction:

- The selector, EG-critic, VerificationAmplifier, and FinalAcceptanceReviewer are instantiated with **no reference** to the `EpistemicBlackboard` object — there is no API path from selection code to the bus. Verification runs on the execution-grounded, context-isolated channel (Section 13), exactly as v1's Cardinal Safety Contract requires: soft signals may re-rank within an execution-verified tier or downgrade an accepted candidate, **never promote** an unverified one.
- A lint/architectural test asserts that `selection/` and acceptance modules do not import the blackboard module (CI guard).
- Shared "facts" can influence *generation priors* only. They feed prompt assembly and, optionally, branch priors/budget the controller may override — never a candidate's acceptance.

This is the conflict with v1 the verdict flagged, and it is resolved by keeping the new sharing a strict generalization of the *producer-only* `EpisodicMemoryBus`, artifact-backed (filesystem-as-source-of-truth) and resume-deterministic.

### 11.8 Configuration Keys

All keys default to a **conservative, evidence-backed** posture; the bus can be fully disabled to recover the pure isolated-rollout baseline for ablation.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `blackboard.enabled` | bool | `true` | Master switch; `false` = pure isolated rollouts (ablation baseline) |
| `blackboard.min_isolated_waves` | int | `1` | Waves kept fully independent before bus opens (phase gate) |
| `blackboard.min_clusters` | int | `2` | Distinct strategy clusters required to open / keep-open bus |
| `blackboard.admit_confidence_floor` | float | `0.35` | Min confidence to consider for admission |
| `blackboard.admit_threshold` | float | tuned→~0.85 admit | Controller score cutoff for admission (LTS target rate) |
| `blackboard.read_confidence_floor` | float | `0.45` | Min confidence to deliver an entry |
| `blackboard.fact_cap` | int | `5` | Max VERIFIED_FACT entries per delivery (v1 positive_limit) |
| `blackboard.negative_cap` | int | `3` | Max NEGATIVE/FAILURE entries per delivery (v1 negative_limit) |
| `blackboard.pool_cap` | int | `64` | Max LIVE Tier-1 entries (cleaner evicts beyond) |
| `blackboard.entry_token_budget` | int | `300` | Max tokens per abstracted entry (MEMOIR ~272) |
| `blackboard.delivery` | enum | `push_turn_boundary` | `pull_stage_boundary` (v1) \| `push_turn_boundary` (new) \| `off` |
| `blackboard.min_coverage_gap` | float | calibrated | Throttle threshold on pass@k−pass@1 |
| `blackboard.min_support` | int | `2` | Throttle threshold on distinct-frontier support size |
| `blackboard.intra_rollout_shared_kv` | bool | `false` | Hogwild-style shared KV for intra-rollout co-workers ONLY |
| `blackboard.roles.cleaner` | bool | `true` | Mandatory in practice; exposed for ablation |
| `blackboard.roles.conflict_resolver` | bool | `true` | |
| `blackboard.roles.critic` | bool | `true` | |
| `blackboard.max_resolution_rounds` | int | `4` | Loop cap (LbMAS) |

> Thresholds marked "calibrated" vary by benchmark (v1 already shows abstention thresholds vary widely); they are set per the Section 20 calibration protocol, never hard-coded blind.

### 11.9 Determinism, Resume, and Vendor Neutrality

- **Determinism / resume (Section 15).** The Tier-1 store is an append-only artifact (`epistemic_blackboard.jsonl`) plus per-`agent()`-call WAL. Each entry carries `created_turn` and `producer_rollout_id`; delivery is computed from a sorted, content-addressed snapshot so a cached/replayed turn sees identical inputs. Push happens only at turn boundaries, so an `agent()` call's inputs remain reproducible (live mid-call mutation would make them nondeterministic — rejected). On restart, the store and WAL replay to reconstruct the exact bus state.
- **Vendor neutrality (Section 3).** Distillation, admission, delivery, and the three roles operate on `EpistemicEntry`s, never on vendor-specific stream formats. `backend` is metadata used only for relevance down-weighting and diagnostics. A run may mix Codex and Claude Code WORKERS; constraints abstracted from a Codex traceback are deliverable to a Claude Code sibling and vice versa, because they are execution-grounded facts, not vendor artifacts. Turn boundaries are detected by the normalized `CLITurnParser` (Section 3), so push works identically across backends.

### 11.10 Honest Limits, Risks, and Mitigations

| Risk | Evidence | Mitigation |
|---|---|---|
| Diversity collapse from sharing | LTS −3.7pp share-all; LLM Monkeys correlation kills pass@k | Phase gate; negatives-only; live throttle (11.6); spawn diversity primary |
| Negative transfer (a constraint wrong off its axis) | DReaMAD context-specificity | `strategy_axis`/`backend` down-weighting; conflict-resolver; confidence floors |
| Hallucinated shared facts | multi-agent collective delusion | Critic role; require corroboration/execution grounding before promotion |
| Verifier contamination | "judge as participant" failure | Producer-only scope enforced structurally + CI guard (11.7) |
| Token blow-up of shared space | LbMAS 4.7M→13.9M | Mandatory cleaner; pool cap; per-entry token budget; dedup |
| Gains unrealizable without good verifier | CMU self-selection plateaus ~55%; [arXiv:2411.17501](https://arxiv.org/abs/2411.17501) optimal-N often <10 | Bus is a *generation* amplifier; execution-grounded selection (Section 13) is the ceiling; benchmark vs strong single-agent + verifier under matched compute |
| "Real-time" overclaim | MEMOIR/LTS are phased, not real-time | Renamed to phased streaming; cross-branch real-time NOT built; Hogwild scoped intra-rollout only |
| Mid-subprocess injection infeasible | opaque CLI workers | Turn-boundary delivery only (Inv. 3) |

**Net disposition (consistent with the Fusion Ledger, Section 18):** *adopt-modified.* The valuable, evidence-backed core — phased, selective, abstracted, negative-constraint sharing on diverse independent rollouts, producer-only, turn-boundary delivery, with relevance/confidence/dedup/own-rollout-exclusion preserved from v1 — is built. The parts the evidence does not support — share-all, positive-solution broadcast, real-time cross-branch coupling, mid-subprocess injection, and any path from the bus to the selector — are explicitly rejected and structurally prevented. Best-of-N with execution-grounded selection remains the floor we can never do worse than; the blackboard is a bounded amplifier on top of it.

## 12. The Vendor-Agnostic Model Economy

### 12.1 Scope, Thesis, and What This Section Does NOT Claim

The model economy is APEX-Ω's mechanism for spending less compute per *resolved* issue without surrendering the resolve rate that the execution-authoritative kernel buys us. It is an **amplifier**, not a load-bearing pillar: the three properties that actually expand capability — execution-grounded verify-and-refute (Section 13), context isolation (Section 15), and orchestration-as-code (Section 2) — work whether or not we cheapen a single worker. The economy rides on top of them.

The disposition in the canonical mechanism list is precise and we honor it verbatim: **"Model economy as sub-role, verification-gated cascade — adopt-modified."** Concretely:

- We **reject** the heavy-orchestrator + thin-executor shape *as the default execution shape on hard repo SWE*. The adversarial verdict on that claim is `partially_sound` at high confidence, and the model-economy SOTA verdict is blunt: "QUALIFIED NO (risky) for naive thin executors on hard multi-file repo SWE." The [HyperAgent ablation](https://arxiv.org/html/2409.16299v1) shows weakening the Navigator (codebase exploration) and Editor (multi-file editing) roles causes the *worst* resolve-rate drops; the small-model gap is steep (HyperAgent's Llama-3-8B "Lite" variant ~16% vs frontier-tier resolve rates, [arXiv:2409.16299](https://arxiv.org/html/2409.16299v1)). A thin executor handed navigation or large multi-file edits regresses toward the cheap-model baseline.
- We **adopt** the cheaper path only as (a) a *sub-role tiering* policy that confines cheap models to the empirically substitutable roles, and (b) a *verification-gated cascade* (try cheap → cheap-first verify on the diff → escalate to frontier on first failure, bounded by a rewrite-cycle cap), so that no cost decision can ever promote an unverified candidate. The cascade is calibration-first, not intuition-first: [FrugalGPT](https://arxiv.org/abs/2305.05176), [RouteLLM](https://arxiv.org/html/2406.18665v4), and the [code-specific self-test cascade](https://arxiv.org/html/2405.15842v1/) all show cheap-then-escalate preserves quality *only when the confidence signal is in-domain-calibrated and a frontier fallback is guaranteed*.

The economy is **opt-in and defaulted off-or-unbounded**, to respect v1's "optimize for SOTA, never for cost" directive and the hard invariant that *a cost cap can never abort a succeeding run*. Turning the economy on is a deliberate operator choice expressed through `budget{}` (Section 16) and the config keys in §12.9.

The unifying safety claim: **execution evidence is both the steering signal and the brake.** Cost shapes *budget allocation and routing priors*; the git diff plus tests decide *acceptance*. The contract and the executor's self-report are never the oracle.

---

### 12.2 The HyperAgent-Tiered Sub-Role Taxonomy

A single "executor" is the wrong unit. The evidence repeatedly separates roles that need sustained, long-context codebase interaction (dangerous to cheapen) from roles that are narrow, well-bounded, and verifiable (safe to cheapen). APEX-Ω's per-rollout stage graph (reproducer → localizer → patcher → test_writer, plus orchestration and selection) maps cleanly onto a tier table.

| Sub-role (APEX-Ω stage / function) | Tier default | Why | Cheapen? |
|---|---|---|---|
| **Orchestrator** (workflow program, FrontierSearch ranking/budget, contract authorship) | `frontier` | Decomposition + search control are where compute demonstrably pays off (Aider architect; Claude opusplan). Errors here compound across the whole tree. | No |
| **Navigator / localizer** (codebase exploration, file/symbol localization) | `frontier` | HyperAgent: weakening Navigator causes the worst resolve drops; long-context env interaction. | No (hard tasks) |
| **Patcher on multi-file / hard edits** | `frontier` or competent-mid (Sonnet-class) | HyperAgent: Editor role degradation is second-worst; the "almost-right trap" (3-4 retries) can cost more than one frontier pass. | No (hard); cautious mid (medium) |
| **Patcher on narrow single-file, well-localized edits** | `cheap` → cascade | Bounded, test-anchored; the contract is nearly complete; safe substitution zone. | Yes, with cascade |
| **Runner / verifier** (run tests, capture diagnostics, regression prune) | `cheap` | HyperAgent: Executor (run/verify) is the *most* substitutable role. v1 already runs verification deterministically; the model here mostly orchestrates shell + parses output. | Yes |
| **Reproducer** (write a failing test that reproduces the issue) | `cheap` → cascade | Narrow, test-anchored output; self-checkable (does it fail at baseline?). | Yes, with cascade |
| **Final-review gate** (frontier review checkpoint after cheap steps) | `frontier` | Mandatory; this is where compounding cheap-model errors are caught before they land. Non-optional (Claude Code guidance). | No |

`Tier` is a vendor-neutral abstraction, *not* a model id (see §12.3). Three logical tiers — `frontier`, `mid`, `cheap` — each resolve to a concrete `(vendor, model, effort)` at command-build time via a routing profile. The large cross-tier price gaps (≈5× on the current 2026 lineup — Haiku 4.5 ~$1/M vs Opus ~$5/M — and larger on legacy tiers, [routing economics](https://www.mindstudio.ai/blog/best-ai-model-routers-multi-provider-llm-cost)) make the economy worth building even with imperfect routing.

```python
# Tier is the planner-facing knob; it never names a vendor.
class Tier(StrEnum):
    FRONTIER = "frontier"   # e.g. claude opus / gpt-5.x xhigh — navigation, multi-file, review, orchestration
    MID      = "mid"        # competent editor class (Sonnet-class) — medium patches
    CHEAP    = "cheap"      # Haiku / gpt-5-mini / flash class — run/verify, narrow edits, reproduction

@dataclass(frozen=True)
class SubRolePolicy:
    role: str                       # "localizer" | "patcher" | "verifier" | "reproducer" | "reviewer" | "orchestrator"
    base_tier: Tier
    allow_cheapen: bool             # gate: never True for navigation/multi-file-on-hard
    difficulty_floor: float | None  # if task difficulty >= floor, force base_tier (no cheapen)
    cascade: "CascadePolicy | None" # how to escalate when cheapened; None => single tier
```

The `difficulty_floor` wires the economy to the difficulty estimator (Section 5 / v1 `estimate_difficulty`): on a hard task, `allow_cheapen` is overridden to `False` for the patcher/localizer regardless of policy, collapsing those roles to `frontier`. This is the structural defense against "cheapen the wrong role."

> Honest hedge: there is no published study sweeping every frontier-planner × cheap-executor combination on full SWE-bench Verified; the per-role degradation numbers are *inferred* from the HyperAgent ablation and scaling curves, not measured end-to-end. The tiering above is therefore a calibrated default, and §12.8's evaluation plan is what turns it from default into measured policy.

---

### 12.3 Tier Resolution: Vendor-Neutral by Construction

Tiers resolve through the normalized Executor interface (Section 3), so the economy works on Codex, Claude Code, or a mixed fleet without special-casing. A routing profile maps `(Tier, capability requirements)` to a ranked list of concrete `(vendor, model, effort)` candidates; the existing `resolve_available_llm_config` failover ranking (v1 `llm_routing.py`) then picks the best-ranked healthy candidate, so a tier never hard-fails when one vendor is down.

```python
@dataclass(frozen=True)
class TierBinding:
    vendor: str          # "claude_cli" | "codex_cli" | "gemini_cli" | "opencode_cli" | "openai_api"
    model: str           # human alias, resolved to launcher id at command-build time (v1 resolved_cli_model)
    effort: str          # "low" | "medium" | "high" | "xhigh" | "max"
    est_price_in: float  # $/M input tokens — for budget accounting only, NOT for acceptance
    est_price_out: float # $/M output tokens

@dataclass
class RoutingProfile:
    # Each tier resolves to an ORDERED candidate list; failover picks best-ranked healthy.
    bindings: dict[Tier, list[TierBinding]]
    same_family_pairs_only: bool = True   # prefer standard<->mini/flash within a family (predictable quality)
```

Two cross-vendor economy levers, both expressed here:

1. **Cost arbitrage.** A heterogeneous fleet can put the heavy orchestrator on one vendor and cheap executors on another, exploiting the cross-tier price gap (~5× on the current lineup, larger on legacy tiers). The fleet is also a strength for *diversity* (Section 13): cross-family errors decorrelate, widening coverage more than re-sampling one model ([Devlo 70.2% / TRAE 70.4% on SWE-bench Verified](https://arxiv.org/html/2506.17208v2) both used 3 distinct cross-vendor models + a selector).

2. **Token *yield*, not invoice.** [xRouter](https://arxiv.org/html/2510.08439v1) shows static "expensive-for-hard / cheap-for-easy" routing trees are brittle and do not transfer across providers; the "almost-right trap" means a cheap executor needing 3–4 retries + a frontier rewrite costs *more* than one frontier pass. The economy therefore measures **cost per verified-resolved task** (post-verification, including cross-validation cost), never gross executor tokens.

**Vendor accounting caveats that the cost model must encode** (these are real and break naive arithmetic):

- Claude subscription `-p` / Agent-SDK usage draws a **separate monthly Agent-SDK credit pool from 2026-06-15** ([headless docs](https://code.claude.com/docs/en/headless)). A mixed fleet's "cost" is not one currency; the budget ledger tracks a per-vendor sub-account.
- Token *units* differ across vendors (different tokenizers); the shared budget normalizes to estimated USD, not to a common token count.
- `prefer same-family tier pairs` (standard ↔ mini/flash) for predictable quality; avoid cross-architecture and closely-matched pairs (the pairing-studies caveat).

```python
@dataclass
class CostLedgerEntry:
    vendor: str
    tier: Tier
    credit_pool: str          # "api" | "claude_agent_sdk" | ... — separate sub-accounts
    input_tokens: int
    output_tokens: int
    est_usd: float            # normalized; from TierBinding prices
    role: str                 # which sub-role spent this
    rollout_id: int
    resolved_outcome: str | None  # filled at acceptance: "verified_resolved" | "abstained" | "failed"
# Yield metric = sum(est_usd) / count(distinct issues with resolved_outcome == "verified_resolved")
```

---

### 12.4 Test-Anchored Contracts: The Planner↔Executor Interface

The contract is the portable, vendor-neutral handoff that lets a tiered/heterogeneous fleet cooperate. It is **per-task, not whole-system** — this directly answers the Beck critique ("writing a full spec before implementation wrongly assumes you learn nothing during implementation") and the Fowler/SDD drift problem ("an outdated spec is as harmful as none"). [OpenAI Symphony](https://openai.com/index/open-source-codex-orchestration-symphony/) validates "software as a spec" portability (one spec → compliant TS/Go/Rust/Java/Python), but *notably defines no planner/executor model split*; we add the split and keep the spec scoped to one task.

We reuse v1's existing `core/contract_slice.py` rather than invent a new format. `build_contract_slice(repo_root, issue_plan, localization_artifact, max_files=8)` / `render_contract_slice` already produce a `# Contract Slice` prompt block (failing tests + relevant files + per-file exports/symbols/imports/stubs), and the `TaskBlackboard` already carries contract *obligations*. The economy promotes this from "prompt scaffolding" to "the typed interface between a frontier author and a cheaper satisfier."

```python
@dataclass(frozen=True)
class TaskContract:
    # Authored by the frontier orchestrator/localizer; satisfied by the (possibly cheaper) patcher.
    contract_id: str
    issue_ref: str
    anchor_tests: list[str]          # test ids the patch MUST make pass (the oracle; e.g. graph_target_test_ids)
    must_not_regress: list[str]      # baseline-passing tests that MUST stay green
    target_files: list[str]         # scoped edit surface (max_files=8), from contract_slice
    obligations: list[str]           # required exports/symbols that MUST survive (TaskBlackboard obligations)
    forbidden: list[str]             # anti-patterns (e.g. "do not stub the tested function")
    contract_slice_block: str        # rendered render_contract_slice() text for the worker prompt
    authored_by_tier: Tier
    rewrite_cycle: int = 0           # incremented each escalation; bounded by CascadePolicy.max_rewrite_cycles
```

Why test-anchored: the contract's acceptance criterion is *executable* (anchor tests pass, must_not_regress stays green). This is what makes the cheap path safe — the cheaper a worker is, the more the false-positive accept risk rises ([ImpossibleBench](https://arxiv.org/html/2510.20270v1): GPT-5 exploits tests 76% on impossible tasks), so the contract is checked by execution on the diff, *never* by the executor's claim that it satisfied the contract. Anchor tests are *self-generated* by the frontier author (reproducer/localizer stages) and are the in-domain confidence signal the cascade keys on (§12.5).

> Spec-drift defense, made concrete: because acceptance runs on the git diff against the anchor + regression tests (filesystem-as-source-of-truth, execution-evidence-authoritative), a stale contract cannot launder a wrong patch. If the contract diverges from reality, the diff fails verification and the candidate is rejected or escalated — drift becomes a *loud* failure, not a silent one. There is no separate "spec consistency in CI" burden because the per-task contract lives and dies inside one solve.

**Handoff hygiene** (attacks the ~37% context-loss failure class, [MAST](https://futureagi.substack.com/p/why-do-multi-agent-llm-systems-fail)): the contract is a *structured briefing* (objectives + constraints + prior decisions + evidence as typed fields above), schema-validated at the Executor boundary — never a raw context dump. This is the same schema-validated-message discipline used everywhere else in APEX-Ω; the economy just reuses it for the cheap-path handoff.

---

### 12.5 The Verification-Gated Cascade (Cascade, Not Blind Routing)

This is the core algorithm. It is deliberately a **cascade** (try cheap, escalate on verified failure) and not static up-front routing, because (a) static routing trees are brittle across providers (xRouter), and (b) the cascade reuses APEX's existing cheap-first verification ladder, so escalation is gated by *execution evidence*, not by a guessed confidence threshold.

```
ALGORITHM: execute_subrole_with_cascade(contract, policy: SubRolePolicy, difficulty, budget)
# Honors: never optimize for cost above correctness; never promote unverified;
#         a cap can never abort a succeeding run.

  # 0. Difficulty override: hard tasks collapse dangerous roles to frontier.
  if (not policy.allow_cheapen) or (policy.difficulty_floor is not None
                                    and difficulty >= policy.difficulty_floor):
      tier = policy.base_tier            # e.g. localizer/patcher -> FRONTIER on hard
  else:
      tier = Tier.CHEAP                  # start cheap on the substitutable / easy slice

  cascade = policy.cascade               # ordered escalation ladder, e.g. [CHEAP, MID, FRONTIER]
  cycle = 0
  while True:
      binding = routing_profile.resolve(tier, contract.required_capabilities)  # vendor-neutral
      # SPAWN a worker via the normalized Executor (Section 3). The worker satisfies the contract.
      exec_result = executor.run(worker_prompt(contract, binding), tier=tier, isolation="worktree")
      diff = executor.observe_diff()     # filesystem/git is the source of truth, vendor-blind
      budget.charge(exec_result.usage, binding, role=policy.role)  # cost accounting (separate credit pools)

      # CHEAP-FIRST VERIFY ON THE DIFF (v1 _build_patch_feedback_generator ladder, reused verbatim):
      #   AST syntax -> public-symbol-survival + stub-residue scan -> targeted cached pytest on anchor tests.
      # This NEVER synthesizes a pass: rc==0 => evidence of a pass; rc==124 => regression_inconclusive (NOT pass).
      verdict = cheap_first_verify_on_diff(diff, contract.anchor_tests, contract.must_not_regress)

      if verdict.passed_anchor_tests and not verdict.regressed:
          return Accepted(diff, tier, cycle, exec_result)     # success: stop, do NOT escalate

      # FAILURE: decide whether to escalate. Escalation is the ONLY response to a verified failure.
      cycle += 1
      next_tier = cascade.next_tier_after(tier)               # CHEAP -> MID -> FRONTIER
      if next_tier is None or cycle > cascade.max_rewrite_cycles:
          # No higher tier left, or rewrite-cap hit. Hand the FAILED candidate (+ failure evidence)
          # to the standard rollout failure path / FrontierSearch as a refuted hypothesis.
          return Refuted(diff, tier, cycle, verdict)
      if budget.remaining() <= 0 and not has_any_verified_success():
          # Budget exhausted, no success yet: a cap must NEVER drop a verified pass, but here there is none.
          return Refuted(diff, tier, cycle, verdict)          # loud failure, never a faked pass
      tier = next_tier
      contract = contract.with_failure_feedback(verdict)      # carry the failing diagnostics forward
      contract = replace(contract, rewrite_cycle=cycle, authored_by_tier=tier)
```

```python
@dataclass(frozen=True)
class CascadePolicy:
    ladder: list[Tier]            # ordered, e.g. [CHEAP, MID, FRONTIER]; last element is the HARD frontier fallback
    max_rewrite_cycles: int = 2   # hard cap on cheap->retry loops (the almost-right-trap brake)
    escalate_on: str = "verified_failure_only"  # never escalate on self-report; only on diff/test evidence
```

Five design commitments, each tied to evidence:

1. **Self-generated tests are the in-domain confidence signal.** The cascade escalates when the contract's anchor tests fail on the diff — functional correctness, the natural code-domain signal ([code-cascade paper](https://arxiv.org/html/2405.15842v1/)), not a model's verbal confidence. This sidesteps the calibration trap that sinks naive thresholds.

2. **Calibration is measured, not intuited; reject poorly-calibrated cheap models.** The threshold that *would* be used for any soft scoring (and the choice of which cheap model is admitted to a tier) is tuned on **labeled traces, not intuition** ([GATEKEEPER](https://arxiv.org/pdf/2502.19335); [Gemini-Lite cautionary tale](https://www.mindstudio.ai/blog/best-ai-model-routers-multi-provider-llm-cost) — a poorly-calibrated cheap model cannot close the accuracy gap at *any* threshold). A cheap model whose self-test pass-signal does not correlate with true resolution on held-out traces is **disqualified from the cheap tier** for that role (§12.8).

3. **Hard frontier fallback is guaranteed.** The last rung of `ladder` is always `frontier`. The hardest tier bypasses any soft gate. No task can get stuck at a tier below frontier solely because a cheap model felt confident.

4. **Rewrite-cycle cap is the almost-right-trap brake.** `max_rewrite_cycles` (default 2) bounds the cheap-retry loop; once hit, the candidate is refuted and budget is routed to the next tier or to surviving hypotheses (the early localization-futility gate, Section 9). This is the futility/budget control that [SWE-Effi](https://arxiv.org/pdf/2509.09853) shows is mandatory to avoid the "token snowball" (4x+ on off-track runs).

5. **The cascade never promotes an unverified candidate.** A cheap success is accepted *only* after diff verification; a cheap failure escalates. Soft signals (the generative critic of Section 13) re-rank within the execution-verified tier or downgrade — never promote. This is the Cardinal Safety Contract applied to the economy.

---

### 12.6 The Mandatory Frontier-Review Gate

Cheap models make more errors, and those errors *compound* (Claude Code guidance treats the review checkpoint as non-optional). After any cheap-executor step that produces an accepted-by-cheap-verification candidate, APEX-Ω inserts a **mandatory frontier-review gate** before the candidate is allowed into the selection pool as a verified peer.

```
ALGORITHM: frontier_review_gate(candidate)   # runs only when candidate was produced/verified by a cheap tier
  if candidate.produced_tier == Tier.FRONTIER:
      return candidate                        # frontier output skips the extra gate
  # 1. Full-scope execution re-verification at frontier rigor: run must_not_regress in full,
  #    plus the discriminating-test amplifier (Section 13) if the candidate is otherwise a tie.
  full = full_scope_execution_verify(candidate.diff)        # execution evidence, authoritative
  if not full.accepted:
      return downgrade(candidate, reason="frontier_gate_regression")   # never silently accept
  # 2. Frontier generative critic (discrimination-only): may DOWNGRADE on contradicting evidence,
  #    may RE-RANK within the verified tier, may NEVER PROMOTE an unverified candidate.
  critic = frontier_critic_review(candidate.diff, contract=candidate.contract)
  candidate = apply_soft_signal(candidate, critic)          # re-rank/downgrade only
  return candidate
```

Two notes:

- The gate is **execution-first**: step 1 is a real test run on the diff; the frontier *model* only enters in step 2 as a discrimination-only critic. This keeps the gate cheap-ish (one full verification + at most one frontier critic call) while still catching the compounding-error class.
- The gate's frontier critic obeys the Cardinal Safety Contract: it operates only *within* the execution-verified tier. It cannot rescue a candidate that failed execution; it can only downgrade a passing-but-suspect candidate or break a tie. This is why a cheap executor *raises* verification load (the N×N cross-validation cost hotspot grows) — the savings the economy delivers are **net of verification**, never gross.

---

### 12.7 Worked Control Flow: Easy vs. Hard Task

To make the economy concrete, here are the two regimes end-to-end. (Difficulty is the Section 5 estimate; `economy_enabled` is off by default per §12.9.)

**Easy / well-localized task (`difficulty < patcher.difficulty_floor`, economy on):**

```
plan -> difficulty=0.2
reproducer  : CHEAP   (writes failing test; self-checks it fails at baseline)        [cheap tier]
localizer   : CHEAP-allowed since difficulty < floor; small repo -> CHEAP            [cheap tier]
contract    : authored by orchestrator (frontier) from localization -> TaskContract
patcher     : CHEAP -> cheap_first_verify_on_diff -> PASS anchor tests, no regress   [cheap tier]
frontier-review-gate : full re-verify PASS + frontier critic no contradiction        [1 frontier call]
verifier    : CHEAP (run suite)                                                       [cheap tier]
=> accepted. Cost ~= mostly cheap tier + 1 frontier critic. Yield: 1 resolved at low $.
```

**Hard / multi-file task (`difficulty >= floor`, economy on):**

```
plan -> difficulty=0.8
localizer   : difficulty >= floor => FORCED FRONTIER (no cheapen; navigation is dangerous to cheapen)
contract    : frontier-authored, multi-file target surface
patcher     : multi-file edit on hard task => base_tier FRONTIER (allow_cheapen False for this slice)
              cheap_first_verify_on_diff -> if FAIL, this is already frontier; refute -> FrontierSearch
verifier    : CHEAP (run/verify is substitutable even on hard tasks)                  [cheap tier]
frontier-review-gate : N/A for frontier-produced patch (skips extra gate)
=> The economy here cheapens ONLY run/verify and reproduction. The expensive cognitive work
   (navigation, multi-file editing, review) stays frontier. This is the rejected-default made safe.
```

The contrast is the whole point: the economy's *savings concentrate on easy/bounded tasks and on the run/verify slice of every task*, exactly where the evidence says cheapening is Pareto-safe, and it *withdraws automatically* on the hard cognitive roles where cheapening regresses quality.

---

### 12.8 Calibration & Acceptance of a Cheap Model into a Tier

A cheap model is admitted to a `(role, Tier.CHEAP)` slot only by passing an offline calibration check on labeled traces — never by intuition. This is the gate that operationalizes "reject poorly-calibrated cheap models."

```
ALGORITHM: calibrate_cheap_binding(binding, role, labeled_traces)
  # labeled_traces: held-out (contract, gold_resolved?) pairs for this role, contamination-resistant split.
  for trace in labeled_traces:
      diff = run_worker(binding, trace.contract)
      cheap_signal = cheap_first_verify_on_diff(diff, trace.contract.anchor_tests, ...)  # the in-domain signal
      true_resolved = oracle_resolves(diff, trace.hidden_tests)                          # ground truth
      record(cheap_signal.passed, true_resolved)
  # Calibration quality = how well the cheap self-test signal predicts true resolution.
  precision = P(true_resolved | cheap_signal.passed)
  if precision < MIN_CHEAP_PRECISION:        # e.g. 0.85 — tune per benchmark, NOT a global constant
      DISQUALIFY binding from (role, CHEAP)   # poorly-calibrated -> excluded; route stays frontier/mid
  else:
      ADMIT binding; record the threshold used in the RunManifest for replay
```

- **Use contamination-resistant, standardized-scaffold splits** for calibration and for any "economy preserves quality" claim — SWE-bench Pro public/commercial, Terminal-Bench 2.0, SWE-bench Live, or a private freshly-authored set. *Never* SWE-bench Verified ([OpenAI deprecated it](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/): 59.4% flawed tests in the audited hard subset, verbatim gold-patch recall). A cheap model's pass rate on a contaminated benchmark is partly memorization, so its calibration would be a mirage.
- **Per-benchmark thresholds.** v1 already shows abstention thresholds vary widely across benchmarks; `MIN_CHEAP_PRECISION` and `difficulty_floor` are tuned per eval, recorded in the RunManifest, and replayed deterministically (Section 15).
- The economy's evaluation in Section 20 must report **cost-per-resolved-task net of verification** and the **token-yield** delta versus frontier-everywhere, on a standardized scaffold (harness-dominates-model: [Terminal-Bench 2.0 shows 30–50pt same-model swings across harnesses](https://www.tbench.ai/leaderboard/terminal-bench/2.0)), or the comparison is meaningless.

---

### 12.9 Configuration, Defaults, and Invariants

The economy is governed by explicit config keys. All default to the SOTA-not-cost stance: the economy is **off**, and `budget{}` is **unbounded** (Section 16), so nothing here can ever silently make APEX cheaper-but-worse or abort a succeeding run.

| Key | Type | Default | Effect |
|---|---|---|---|
| `economy_enabled` | bool | `False` | Master switch. Off ⇒ every role runs at its frontier-equivalent tier (v1 behavior). |
| `routing_profile` | RoutingProfile | frontier-only | Maps tiers → ordered `(vendor, model, effort)` candidates. |
| `subrole_policies` | dict[str, SubRolePolicy] | per §12.2 table | Per-role base tier, `allow_cheapen`, `difficulty_floor`, cascade. |
| `cascade.max_rewrite_cycles` | int | `2` | Almost-right-trap brake on cheap-retry loops. |
| `cascade.ladder` | list[Tier] | `[CHEAP, MID, FRONTIER]` | Escalation order; last rung is always the hard frontier fallback. |
| `min_cheap_precision` | float | `0.85` (tune per bench) | Disqualify cheap bindings below this calibrated precision. |
| `frontier_review_gate.enabled` | bool | `True` *(when economy on)* | Mandatory review after cheap steps; cannot be disabled while `economy_enabled`. |
| `cross_vendor_arbitrage` | bool | `False` | Allow heavy-orchestrator-one-vendor + cheap-executor-another fleets. |
| `same_family_pairs_only` | bool | `True` | Prefer standard↔mini/flash tier pairs for predictable quality. |
| `budget` | Budget\|None | `None` (unbounded) | Shared cost ceiling (Section 16); a cap can never abort a succeeding run. |

**Hard invariants the economy must never violate** (these are non-negotiable and cross-cut the whole plan):

1. **A cost decision never promotes an unverified candidate.** Cheapness sets *priors and budget share*; the git diff + tests decide acceptance (Cardinal Safety Contract, Section 13).
2. **A budget cap never aborts a succeeding run.** Caps fire only when *no* verified success exists (v1's `_cumulative_token_cap_exceeded` invariant), and never drop a candidate already in the escrow WAL (Section 15).
3. **Hard tasks keep frontier on navigation, multi-file editing, and final review.** `difficulty_floor` enforces this structurally; the thin-executor-default is rejected.
4. **Frontier-review gate after cheap steps is mandatory** while the economy is on — it cannot be configured away.
5. **Acceptance is vendor-blind.** The diff is the oracle regardless of which vendor/tier produced it; the contract and the executor's self-report are never trusted (filesystem-as-source-of-truth).
6. **Measure yield, not invoice.** All economy reporting is cost-per-verified-resolved-task, net of verification, with separate per-vendor credit-pool accounting (notably Claude's Agent-SDK pool from 2026-06-15).

---

### 12.10 Why Thin-Executor-Default Is Rejected (and What We Keep Instead)

For the record, since this is the most tempting and most dangerous economy shape:

- **The almost-right trap.** A thin executor that produces a plausible-but-wrong patch costs the full verification pass *plus* the rewrite cycles *plus* the eventual frontier escalation — frequently more than one clean frontier pass. xRouter and the model-economy SOTA both warn: measure yield, not invoice.
- **The small-model cliff.** (HyperAgent's own Llama-3-8B "Lite" variant scores ~16%, far below frontier-tier resolve rates, [arXiv:2409.16299](https://arxiv.org/html/2409.16299v1)); the resolve-rate curve is steep and frontier-dominated. A thin executor on the cognitive roles sits in the collapse zone for hard tasks.
- **The wrong-role ablation.** HyperAgent shows Navigator/Editor are the *worst* roles to cheapen; only run/verify is safely substitutable. A uniform "cheap executor" ignores this and regresses toward the cheap-model baseline.
- **The false-positive amplifier.** A cheaper executor raises reward-hacking and benchmark-passing-but-wrong rates ([ImpossibleBench](https://arxiv.org/html/2510.20270v1)), stressing verification harder and eating the savings.

What we keep is the *defensible* version the evidence supports: a frontier planner + frontier reviewer spine, cheapening confined to run/verify and narrow well-localized edits, a calibrated verification-gated cascade with a guaranteed frontier fallback and a rewrite-cycle cap, test-anchored per-task contracts as the portable interface, and execution-evidence-authoritative acceptance that neutralizes both spec drift and the cheap executor's added error rate. This is the [Aider architect/editor](https://aider.chat/2024/09/26/architect.html) Pareto win (improved *every* model over its solo baseline, polyglot SOTA at ~14x lower cost) generalized across vendors and confined to the regime where it actually holds.

See Section 13 for the verification/selection machinery this economy depends on, Section 14 for the active controller that learns the routing/budget priors over journaled traces, Section 15 for the determinism + escrow-WAL guarantees that make "a cap can never drop a verified pass" true across restart, and Section 16 for the `budget{}` primitive and broader cost engineering.

## 13. Verification & Evidence-Grounded Selection (Verify-and-Refute)

> **Where this sits.** Verification is the load-bearing claim of APEX-Ω (see Section 1, Central Thesis): *capability beyond the base model comes from execution-grounded verify-and-refute, not from agent count.* Generation (Sections 8–12) produces a population of candidate diffs across vendors and budgets; this section is the machinery that converts that **coverage** (the fraction of issues with *some* correct candidate) into **trustworthy resolved issues**, and is the brake that keeps every speculative extension (Sections 9–12, 14) from ever shipping an unverified answer. Without it, the search/economy/blackboard amplifiers degrade into the inference-scaling false-positive trap, where a weak selector lets coverage saturate or *invert* into worse-than-single-shot outcomes ([Limits of Inference Scaling Through Resampling, arXiv:2411.17501](https://arxiv.org/abs/2411.17501)).

This section preserves APEX v1's verification kernel **verbatim** as the hardened substrate, then layers exactly two evidence-supported extensions on top — a swappable generative critic *for discrimination only*, and a single re-usable `verify_and_refute()` primitive callable at multiple granularities — both bound strictly by the Cardinal Safety Contract.

### 13.1 The Cardinal Safety Contract

> **Execution evidence is authoritative. A soft signal may re-rank *within* an execution-verified tier, or *downgrade* an already-accepted candidate — it may NEVER promote an unverified candidate, and it may never exclude a candidate before execution evidence exists.**

Enforced structurally (v1 §13.1): `_apply_evidence_bound_review` flips `accepted` only `True -> False`; `VerificationResult` has no `passed`/`status` field (`accepted` is the gate); the deterministic lexicographic ranking tuple `(combined_score, accepted, public_signal_score, critic_score, size, verification_score, eg_critic_tiebreak, perspective_score, len(changed_files), -cluster_id)` places every soft/learned/LLM key strictly below every execution+critic key, terminating in `-cluster_id` (never insertion order, killing slot-0 bias). Pre-execution scorers (Sections 9–10) set priors/budget only.

**Inverse-violation symmetry.** The contract forbids both directions of error: soft signals cannot promote (false positive), and pre-execution scorers cannot suppress a correct-but-unverified candidate (false negative). This is why static-AST CTDG-as-gate and plan-scoring-as-gate are **rejected** (Section 18): both decide a candidate's fate without execution evidence. Static-AST-as-gate also fails on merit (PyCG ~70% recall; reflection/monkeypatch/fixtures invisible; pytest set not statically enumerable).

**Abstention is first-class.** No positive execution evidence ⇒ `None`, not a least-bad guess (v1 §13.10): all-env-failed → `None`; `cross_candidate_voter` → `winner=None` when all `oracle_score==0`; xval returns `[]` (not a synthetic `0.5`) for singletons; `_selected_result_is_accepted` requires positive evidence; the heuristic fallback stamps `0.35` confidence so it is never mistaken for a real solution.

### 13.2 Cheap-first verification cascade (kept verbatim — never synthesizes a pass)

```
verify_patch(candidate, repo_ctx, baseline) -> VerificationResult:
    score = 0.0
    if not ast_compiles(candidate.changed_py_files):                # 1 SYNTAX
        return VerificationResult(accepted=False, score=0.0, reason="syntax")
    if flake8_available() and lint_clean(candidate): pass           # 2 LINT (gate only)
    if reproduction_passes(candidate): score += 0.35                # 3 REPRO
    prune = prune_by_regression(candidate, baseline, chunk=50)      # 4 REGRESSION
    if not prune.is_valid:                                          #   only baseline-passers,
        return VerificationResult(accepted=False, score=score, reason="regression")
    score += 0.35                                                   #   chunks of 50, in worktree
    xval = build_cross_validation_matrix(candidate, peers)          # 5 NxN ([] for singletons)
    if xval: score += 0.10 * mean(xval)
    score += 0.10 * candidate.pass_rate                             # 6 SCORE + ACCEPT
    return VerificationResult(accepted=positive_execution_evidence(candidate), score=score)
```

Never-synthesize-a-pass countermeasures, carried forward exactly:

| Pathology | Countermeasure | Rationale |
|---|---|---|
| Silent `rc==0` no-op | `errors=1`, never `passed=1` | A no-op must never read as pass |
| Timeout `rc==124` | separate `regression_inconclusive` axis, `+0.15` partial | Honest uncertainty, neither FP nor FN |
| Zero tests / singleton xval | abstain (`[]`), never synthetic `0.5`; `M[i][i]` excluded | No fabricated evidence |

**Test suite is a noisy signal, not an oracle.** SWE-bench has ~11.3% flaky problems; CodeContests has reference solutions failing their own tests ([arXiv:2407.21787](https://arxiv.org/abs/2407.21787)). v1 mitigations stay: NDFF flake firewall declares a flake only on positive evidence; the upstream Docker harness is the only publishable number (v1 §15.4).

### 13.3 Bounding the N×N matrix (cost discipline)

Cross-validation is `O(N²)` sandboxed test runs (v1 §12.6). Bound it **before** building, never by dropping candidates: **Bound 1** — AST two-pass clustering (exact `ast.dump` sha256 fingerprint, then single-linkage merge at `0.95`; `semantic_distance` pools `operator_kinds→control_flow`, `constant_signature→data_structures`) collapses behavioral duplicates; the matrix is built over **cluster representatives** (zero false-negative risk — identical patches can't disagree). **Bound 2** — CTDG test-impact selection (Section 10) skips cell `(i,j)` only when `j`'s diff provably can't affect `i`'s tests, with a **full-suite backstop**. Reordering + dynamic-coverage skip, never static gating.

```
build_cross_validation_matrix(candidates, cfg):
    reps = cluster_representatives(candidates)             # Bound 1
    M = {}
    for i in sorted(reps, key=cluster_id):
        impacted = dynamic_coverage_impact(i.tests)        # Bound 2
        for j in sorted(reps, key=cluster_id):
            if i is j: continue
            if cfg.test_impact_prune and disjoint(j.diff_files, impacted): continue  # cell SKIP
            M[(i,j)] = run_tests_sandboxed(tests=i.tests, worktree=j.worktree)       # 120s/cell
    return M
```

`xval.cluster_before_matrix=true`, `xval.test_impact_prune=true`, `xval.full_suite_backstop=true`, `xval.cell_timeout_s=120`, `xval.sandbox=true`.

### 13.4 Hybrid verification: execution anchors correctness, critic breaks ties

For code (distinct from math), execution-based and execution-free verifiers are **complementary**: execution is the reliable correctness anchor but has *low distinguishability* (cannot separate two patches passing the same tests); execution-free critics give *better discrimination* but are biased toward surface features and gameable. Each plateaus ~42–43% on SWE-bench Verified; the hybrid reaches **51.0%** at 26 rollouts ([R2E-Gym, arXiv:2504.07164](https://arxiv.org/abs/2504.07164)). *Do not choose; combine.* APEX-Ω keeps execution as the gate and adds a generative critic whose only job is **discrimination among equally-passing patches within the execution-verified tier**.

```
hybrid_select(candidates):
    verified = [c for c in candidates if verify_patch(c).accepted]   # execution gate
    if not verified: return None                                     # ABSTAIN
    tier = top_execution_tier(verified)
    if len(tier) == 1: return tier[0]
    scores = {c: generative_critic.score(c) for c in tier}           # within-tier ONLY
    return deterministic_rank(tier, critic=scores)[0]
```

**Best@K vs Pass@K gap is a first-class metric.** SWE-Gym's ORM: Best@16=32.0% vs Pass@16=42.8% — ~11pp, capturing only ~70% of headroom ([arXiv:2412.21139](https://arxiv.org/abs/2412.21139)); CodeMonkeys realized 57.4% vs a 69.8% ceiling ([arXiv:2501.14723](https://arxiv.org/abs/2501.14723)). APEX-Ω reports `best_at_k`, `pass_at_k` (oracle ceiling via held-out gold), and their gap per run — the honest measure of unrealized oracle headroom and the optimization target for Sections 14 and 17.

### 13.5 The swappable generative critic (vendor-neutral, fail-open)

Verifier *strength* is the binding constraint: weak/open critics can fail to beat the no-verifier baseline ([SWE-PRM, arXiv:2509.02360](https://arxiv.org/html/2509.02360v1): closed +5–11pp, open drop to 30–38.8%). The critic is pluggable and degrades to execution-only when weak.

```
@dataclass
class CriticVerdict:
    score: float;  rationale: str;  reliability: float;  abstain: bool   # score is NOT a pass
class Critic(Protocol):
    def score(self, candidate, peers, evidence) -> CriticVerdict: ...
@dataclass
class CriticConfig:
    backend="vendor_neutral"; style="generative"; min_reliability=0.5
    fail_open=True; sees_producer_context=False; timeout_s=1800
```

- **Generative/long-CoT over discriminative scalar PRMs** — robust + data-efficient OOD ([THINKPRM, arXiv:2504.16828](https://arxiv.org/abs/2504.16828): +4.5% LiveCodeBench on ~1K examples; discriminative PRMs fragile under shift).
- **Reliability-weighted, Youden-gated** — below `min_reliability` the critic is ignored (A-HMAD weighting; "SWE-finetuned models are not inherently reliable PRMs").
- **Execution-grounded self-critique over trained per-step PRMs** ([ORPS, arXiv:2412.15118](https://arxiv.org/html/2412.15118v1): implicit 59.9% vs trained 37.0% Pass@1). No bespoke per-token PRM.
- **Fail-open** — error/timeout ⇒ `accept=True`+abstain, fall back to deterministic ranking (v1 §13.8 default-off/fail-open).

> **Hard rule:** the critic never promotes an unverified candidate — `hybrid_select` scores only the already-verified `tier`, and the lexicographic tuple (13.1) keeps every critic key below every execution key.

### 13.6 `verify_and_refute()`: one primitive, many granularities

The capability multiplier is verify-and-refute ("stops saying done when half done"): orchestrator-worker + refutation beat single-agent Opus 4 by 90.2% ([Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)); debate reduces hallucination ([Du et al., arXiv:2305.14325](https://arxiv.org/abs/2305.14325)). v1's three forms (family-disjoint tool-call review, self-play tournament, VerificationAmplifier) lift into one primitive.

```
verify_and_refute(target, granularity, refuters, evidence, cfg) -> Verdict:
    verdict = Verdict(accepted=False, reason="default_refute")        # 1 default-to-refute
    if granularity == "candidate":
        ev = verify_patch(target)                                     # 2 executed evidence
        if not ev.accepted: return verdict                           #   no promotion path
        verdict = Verdict(accepted=True, evidence=ev)
    challenges = parallel([r.refute(target, evidence, redacted=True)  # 3 cross-vendor
                           for r in refuters])                       # 4 no producer ctx
    weighted = sum(c.broke * r.reliability for c,r in zip(challenges, refuters))  # 5 weighted
    if weighted >= cfg.refute_threshold: verdict = downgrade(verdict)   # downgrade-only
    return verdict
```

Invariants: (1) **default-to-refute** (broken until evidence shows otherwise); (2) **grounded in executed evidence** (gate is `verify_patch`); (3) **cross-vendor refutation is a strict diversity gain** over single-family review — different (vendor,model) families decorrelate blind spots (Devlo/TRAE pattern, Section 3); (4) **verifier never sees producer context** (`redacted=True`; family-disjoint enforced at config+hook-build, "a CLI may not be reviewed by its own family") — **reviewer fail-open caveat:** a reviewer error → no veto, so refutation can only strengthen the gate, never weaken it, but a silently-broken reviewer offers no protection (backstopped by reliability weighting + the execution gate); (5) **reliability-weighted, downgrade-only**.

| Granularity | Target | Use site | v1 lineage |
|---|---|---|---|
| `tool_call` | one tool invocation | mid-rollout, turn boundary | family-disjoint tool-call review |
| `finding` | a claim/localization | blackboard contribution (Section 11) | EpisodicMemoryBus |
| `candidate` | a complete patch | selection (13.4) | self-play tournament + VerificationAmplifier |

### 13.7 Anti-reward-hacking (a safety requirement)

Reward hacking scales with capability and generalizes beyond the hack: exploitable coding-RL produced `sys.exit(0)`/pytest-patching that generalized to 12% sabotage, 50% alignment-faking vs 0% baseline ([Anthropic, arXiv:2511.18397](https://arxiv.org/abs/2511.18397)); extensional-only verifiers "admit false positives" ([arXiv:2604.15149](https://arxiv.org/abs/2604.15149)). Since APEX-Ω may self-improve (Section 17), hack-resistance is a safety requirement.

- **Hidden/read-only tests in an agent-unreachable grader** — upstream Docker harness in an isolation tier the worker can't reach/edit; test files read-only, hash-checked; editing assertions/tests/CI forbidden. **Prefer held-out tests**; never benchmark with the same tests used as the selector's verifier.
- **Exploit detection** — `sys.exit(0)`, `__eq__`-override, pytest-monkeypatch/report-patching, exfil. v1 `patch_sanitizer`: `VENDORED_UPSTREAM_ARTIFACT` (stripped, soft signal) vs `GOLD_PROTECTED_TEST` (reject wins) — a protected test under a vendored `testing/` dir still rejects.
- **Transcript auditing** — anti-hack ledger grounds by structured fingerprint (`op+shape+repr`), biases to false-negatives, soft-downweights.
- **Gold-channel destruction, not neuron suppression** — gold fields discarded (not masked) at load; Commit0 true git-history flatten ("block the channel, never the neuron").
- **Verifier Youden gating** — learned verifier admitted only above a Youden-J threshold; TTS skill ≠ RL-reward fitness ([SWE-RM, arXiv:2512.21919](https://arxiv.org/abs/2512.21919)), so re-validate before any RL use.
- **Inoculation prompting** (defer, Section 17) — cut misalignment 75–90% despite >99% hack rate (Anthropic).

### 13.8 Failure taxonomy & abstention (infra ≠ model miss)

8-bucket `FailureClass` + `UNCLASSIFIED`, phase-ordered never-raising `classify()` (v1 §17.2): `charges_apex == {APEX_MISS, UNCLASSIFIED}` only; `HARNESS_BUG` excluded from both `is_environment` and `charges_apex`; `ENV_*`/`NON_DETERMINISTIC` retryable. Phase makes `ModuleNotFoundError` an install-failure during setup but an APEX-miss during test execution. Keeps abstention honest: env failure abstains+retries; real miss abstains, no retry.

### 13.9 Selection pipeline (end-to-end) and config

```
select(candidates, repo_ctx, baseline, cfg) -> Selection:
    verified  = [verify_patch(c, repo_ctx, baseline) for c in candidates]      # 1 cascade
    survivors = [c for c,v in zip(candidates, verified) if v.accepted]
    if not survivors: return Selection(winner=None)                            # ABSTAIN
    reps = cluster_representatives(survivors)                                   # 2 cluster
    M    = build_cross_validation_matrix(reps, cfg.xval)                        #   bounded NxN
    ranked = deterministic_rank(reps, xval=M); tier = top_execution_tier(ranked) # 3 exec-rank
    if cfg.critic.enabled and len(tier) > 1 and critic.reliability >= cfg.critic.min_reliability:
        ranked = reorder_within_tier(tier, critic.score)                       # 4 within-tier
    winner = ranked[0]
    if cfg.refute.enabled and verify_and_refute(winner,"candidate",cross_vendor_refuters,M,cfg.refute).accepted is False:
        ranked = ranked[1:]; winner = ranked[0] if ranked else None            # 5 refute
    winner = apply_evidence_bound_review(winner)                               # 6 True->False only
    record_metrics(best_at_k=..., pass_at_k=..., gap=...)                      #   13.4 metric
    return Selection(winner=winner)                                           #   may be None
```

| Config key | Default | Notes |
|---|---|---|
| `critic.enabled` | `false` | Default-off; downgrade-only ⇒ cannot move headline |
| `critic.min_reliability` | `0.5` | Youden gate; below ⇒ execution-only |
| `critic.backend` | `vendor_neutral` | Any worker family |
| `refute.enabled` | `true` | Default-to-refute |
| `refute.cross_vendor_only` | `true` | Strict diversity over single-family |
| `eg_critic.enabled` | `false` | Triple-gated (v1) |
| `final_acceptance.enabled` | `false` | Downgrade-only, fail-open |
| `abstain_on_no_evidence` | `true` | Cardinal Contract |
| `metrics.report_best_vs_pass_gap` | `true` | Oracle-headroom diagnostic |

### 13.10 Worker/Executor interface (vendor-neutral) and build notes

Critic and refuters are ordinary **workers** via the Normalized Executor (Section 3): opaque external CLIs (Claude Code, Codex, open-weight, mixed) observed via stdout turn-parsing + artifacts, never driven in-process. They consume a redacted prompt (`sees_producer_context=False`) and emit structured `CriticVerdict`/`Verdict` (embed-and-post-parse when no native schema). Evidence/verdicts flow to disk (filesystem-as-source-of-truth), never into an orchestrator context. To build on **Codex** or **Claude Code** you need only: (a) `verify_patch` over a worktree (Section 15), (b) the `Critic` protocol pointed at any worker, (c) `verify_and_refute` with cross-vendor refuters, (d) the deterministic ranking tuple — none vendor-specific.

**Honest limits.** Guarded, not certain: (1) the generative critic helps only with a strong backend, so it ships default-off + Youden-gated; (2) the Best@K–Pass@K gap means real oracle headroom is left unrealized — we report it, not hide it; (3) the test suite is noisy (flaky/contaminated), so every accept is "best available execution evidence," not proof of correctness. The final human gate (read the diff, run the upstream harness) is the ultimate authority — passing tests are not correctness.

## 14. The Active Adaptive Controller & Learned Search Policy

### 14.1 Scope, Stance, and the One Load-Bearing Claim

This section specifies the component that *steers* APEX-Ω: the policy that, at every fan-out point, chooses **which vendor/model to spawn, at what effort, and how to allocate width-vs-depth budget** — promoting v1's *passive* controller (observe-and-nudge) into an *active* one (decide-and-drive), without ever surrendering the execution-authoritative guarantees of the substrate.

The load-bearing claim, stated honestly against the adversarial reality of the SOTA digest:

> An **active, learned** orchestration controller beats a static hand-tuned baseline **only** when it is (a) staged so it ships value at every milestone, (b) cheap-by-default (a contextual-bandit/REINFORCE router on day one, reflective prompt evolution next, full RL last and optional), (c) grounded on the deterministic verifier signal APEX already produces, and (d) structurally *unable* to promote an unverified candidate. It does **not** beat baseline if we bet the system on plain outcome-reward RL converging over long, role-heterogeneous traces.

Two facts from [the RL-for-LLM-MAS survey (arXiv:2605.02801)](https://arxiv.org/html/2605.02801v1) bound the entire section. First, orchestrator-level credit assignment is the *easiest* to define yet still rare (8 of 84 tracked methods), and **no** RL method yet targets the stop decision (O5) — so "when to stop / escalate / spawn" must be *engineered*, not awaited from RL. Second, [GEPA (arXiv:2507.19457)](https://arxiv.org/abs/2507.19457) beats GRPO by ~6pp average / up to 19pp using **up to 35x fewer rollouts**, which makes reflective prompt evolution — not scalar RL — the default high-leverage learning lever for a controller whose feedback is rich natural-language traces (test logs, diffs, reviewer comments).

The controller sits **above** selection (Section 13) and **drives** the search/allocation layer (Section 9), the model economy (Section 12), and the blackboard's phase boundaries (Section 11). It is the active brain; the Cardinal Safety Contract (Section 13) is the brake it cannot release.

#### 14.1.1 What "active" means here, precisely

v1's controller is passive in two distinct senses (Section 4, SEAM 2). Layer A — the wave/escalation loop `_execute_with_dynamic_transitions` — only decides *how many* rollouts and *whether* to escalate; its sole mid-flight channel into an opaque CLI is the `CLITurnParser` `turn_observer` course-correction. Layer B — the calibrated controller-policy (`controller_policy.py` / `controller_models.py`) — is **blend-not-switch**: `evaluate_policy_model` guarantees `applied=False => value==baseline`, missing/disabled/malformed models degrade silently to heuristics, and the intended library-wide kill switch `ControllerModelLibraryConfig.library_enabled` is **unwired** (zero runtime consumers).

APEX-Ω makes the controller active by giving it three new authorities it lacked, while keeping the blend-not-switch safety property:

| Decision | v1 (passive) | APEX-Ω (active) | Bounded by |
|---|---|---|---|
| Worker count (K) | heuristic wave count; adaptive allocation OFF by default | difficulty-adaptive low-K, default ON | Section 9 budget caps |
| (vendor, model) per fan-out | resolved from `LLMConfig`/failover only | Thompson-sampled capability/cost profile (§14.4) | Section 13 selection; portfolio (Section 3) |
| Effort / reasoning level | fixed per backend | per-node effort knob (cheap-first cascade) | Section 12 model economy |
| Wider vs deeper allocation | fixed `max_frontier_branching` | learned AB-MCTS-style allocation (§14.5) | Section 9; `CONFIDENCE_FLOOR` |
| Stop / escalate / spawn | heuristic backstops | decoupled **calibration module** (§14.7) | Cardinal Contract; never promotes |

"Active" is therefore **allocation and routing authority**, exercised *before* a worker runs and *between* turns — never the authority to *accept* a result. Acceptance is and remains the execution-authoritative selector's job.

### 14.2 The Non-Negotiable Invariants (the brakes)

These trace directly to v1's Cardinal Safety Contract, the foundational frame, and the adversarial verdicts. They are enforced structurally, not by convention.

1. **Cardinal Contract is below the controller.** The controller may set branch priors, budget share, vendor/model, and effort. It **cannot** promote, accept, or rank-above any candidate that lacks execution evidence. Soft/learned controller signals re-rank *within* an execution-verified tier or downgrade only — never promote (Section 13). A controller that could elevate an unverified patch would re-introduce exactly the inference-scaling false-positive failure mode the substrate exists to prevent.

2. **Blend-not-switch, fail-open to heuristic.** Preserve v1's `evaluate_policy_model` contract verbatim: a missing, disabled, malformed, or low-confidence learned policy degrades **silently and deterministically** to the heuristic baseline (`applied=False => value==baseline`). A learned component can *improve* a run; it can never *break* one. This is the property that lets us ship learned control at low risk.

3. **Wire or remove `library_enabled`.** The single most important correctness gap in v1's controller layer is an unwired kill switch. APEX-Ω **wires it**: `controller.library_enabled=False` is a hard, runtime-honored master off-switch that forces every controller decision to the heuristic baseline. (If a future maintainer prefers, *remove* it entirely — but the current half-state, defined-and-serialized-yet-unconsumed, is forbidden.)

4. **Cache validity is input-hash-keyed.** An active route must never replay a stale cached result for changed code. Every controller-routable result is keyed on a content hash of `(repo_state_hash, issue_hash, node_prefix_hash, vendor, model, effort)`; a cache hit is admissible *only* on exact match (§14.8). This is the active-controller analogue of v1's input-hash journaled resume (Section 15).

5. **Every decision is journaled.** Every controller decision — features, action, the sampled `(vendor, model, effort)`, the realized verifier outcome and cost — is appended to `controller_decisions.jsonl` (§14.9), the substrate for reproducible *off-policy* credit assignment.

### 14.3 The Staged Roadmap (works at every milestone)

The cardinal design discipline, dictated by the survey and the canonical IN/OUT dispositions, is: **do not bet the system on Stage-2 RL converging.** Each stage is independently shippable and strictly dominates the prior in capability while preserving every invariant.

| Stage | Mechanism | Ships | Disposition | Beats baseline? (evidence) |
|---|---|---|---|---|
| **0** | Contextual-bandit / REINFORCE **router** over capability/cost profiles | **Day one (active)** | adopt | Yes — [BaRP (arXiv:2510.07429)](https://arxiv.org/html/2510.07429v1): REINFORCE-MLP 0.7432 > LinTS 0.6430 > LinUCB 0.6166; +12–16% over offline routers at ~50% lower cost |
| **1** | [GEPA](https://arxiv.org/abs/2507.19457)-style reflective prompt evolution of controller prompts/tool-descriptions | After Stage 0 | defer | Yes — beats GRPO ~6pp avg / 35x fewer rollouts; beats MIPROv2 >10pp |
| **2** | Cost-penalized RL ([Puppeteer](https://arxiv.org/abs/2505.19591)/[Conductor](https://arxiv.org/html/2512.04388) style) over orchestrator decisions | Only when volume justifies | defer | Conditional — Conductor 7B avg 77.27 > GPT-5 74.78; Puppeteer 0.7731 > AFlow 0.6899; but credit diffuses on long traces |

Crucially, **Stage 0 ships active control on day one** by learning from the verifier signal APEX *already* produces (test pass/fail, regression survival, build, lint, reviewer accept). It does not require a training pipeline, a held-out set, or RL infrastructure — it learns online from partial feedback, which is the realistic deployment regime ([RouteLLM](https://github.com/lm-sys/RouteLLM) assumes full supervision that breaks at deployment; bandits handle the partial-feedback reality).

### 14.4 Learned Capability/Cost Profiles, not One-Hot Identities

This is the defensible, NeurIPS-grade contribution of the section and the open-pool generalization mechanism.

A naive router learns a per-`(vendor, model)` one-hot value (a bandit arm per identity). That **cannot generalize** to a backend it has never seen, and a vendor-neutral fleet *will* add, drop, and version-bump backends mid-lifecycle (Section 3's self-evicting `BackendPortfolio`). Instead, following the MoMA/DAAO "model-as-attributes" framing, each backend is represented as a **dense profile vector** of *capabilities and costs*, and the policy maps `(task_features ⊕ profile_vector) -> value`. A new backend is admitted by *describing its profile*, not by retraining from scratch — the policy already knows how to value "a cheap, fast, weak-at-multi-file-edit model" because it has seen that *region of profile space* before.

```python
@dataclass(frozen=True)
class BackendProfile:
    # identity is metadata only — NEVER a feature the policy keys on (open-pool generalization)
    backend_id: str                  # "claude_cli", "codex_cli", ... (logging/portfolio only)
    model_id: str                    # resolved launcher id, pinned in RunManifest

    # --- learned/measured CAPABILITY axes (EWMA-updated from verifier outcomes) ---
    cap_localization: float          # [0,1] navigation / multi-file localization skill
    cap_multifile_edit: float        # [0,1] cross-file coherent edit skill (HyperAgent-sensitive)
    cap_narrow_edit: float           # [0,1] single-hunk/diff-application skill (Aider editor lane)
    cap_repro_test: float            # [0,1] reproduction / test authoring skill
    cap_long_horizon: float          # [0,1] survival over many turns without drift

    # --- COST / latency axes (measured, deterministic) ---
    cost_per_ktoken_in: float        # provider-cache-adjusted (Section 12, prefix reuse)
    cost_per_ktoken_out: float
    median_turn_latency_s: float
    median_tokens_per_solve: float

    # --- reliability / health (from BackendPortfolio + failure taxonomy) ---
    transient_failure_rate: float    # 429/stall/reset; informs call-failover, not value
    hard_failure_rate: float         # auth/missing-binary; informs global reroute

    # --- profile freshness ---
    n_observations: int
    updated_at: float
```

```python
@dataclass(frozen=True)
class TaskFeatures:
    difficulty_est: float            # v1 estimate_difficulty (Section 5/9)
    n_relevant_files: int            # localization futility signal (Section 9 gate)
    n_candidate_hypotheses: int
    has_reproduction: bool
    has_reliable_tests: bool         # gates execution-authority strength (Section 13)
    issue_embedding: tuple[float, ...]  # frozen-encoder embedding of issue text
    stage: Literal["reproduce","localize","patch","verify","narrow_edit"]
    remaining_budget_frac: float     # [0,1]; broad-early/greedy-late (Section 9, BAVT)
    feedback_confidence: float       # [0,1]; below floor -> heuristic (Section 9 floor)
```

The policy is a **non-linear MLP** over `concat(task_features, profile_vector)` (BaRP shows REINFORCE-MLP beats linear LinUCB/LinTS, which underperform on heterogeneous query distributions). Preference-conditioning ([BaRP](https://arxiv.org/html/2510.07429v1)) lets one policy serve many cost-quality tradeoffs via a per-request preference vector `w = (w_quality, w_cost)`:

```
score(task, profile, w) = MLP_theta( task_features, profile, w )
reward                   = w_quality * quality - w_cost * min(cost/tau, 1)
```

This is the mechanism that makes `(vendor, model)` a first-class search/diversity axis (Section 9.7) *learnable* without one-hot lock-in, and it is intrinsically vendor-neutral because the policy reasons over *attributes*, never names.

> Honesty note: open-pool generalization via profile vectors is **promising but not independently SWE-proven** at the orchestrator level. It is therefore guarded: profiles bootstrap from measured cost + heuristic capability priors, the policy fails open to v1 failover ranking when `n_observations` is below a floor, and we always evaluate the bandit's lift against the heuristic baseline (§14.10) before trusting it.

### 14.5 Stage 0 — The Contextual-Bandit / REINFORCE Router (active day one)

Stage 0 is the active control surface. At each fan-out point the controller emits an `AllocationDecision` (the structure from Section 9.3): an action in `{WIDER, DEEPER, DIVERSIFY, STOP}`, a target node, a branch count clamped to remaining budget, and a sampled `(vendor, model, effort)`.

```python
def route(task: TaskFeatures, profiles: list[BackendProfile],
          pref: Preference, budget: SearchBudget, cfg) -> AllocationDecision:
    # INVARIANT 2 + 3: master off-switch and fail-open to heuristic
    if not cfg.controller.library_enabled or not cfg.controller.bandit_enabled:
        return heuristic_allocation(task, profiles, budget)        # v1 baseline, applied=False

    # INVARIANT 1 prerequisite: below feedback floor, do NOT let a soft policy steer search
    if task.feedback_confidence < cfg.controller.confidence_floor:
        return heuristic_allocation(task, profiles, budget)        # Section 9 floor -> best-of-N

    # Thompson sampling over (vendor, model) via the profile policy (open-pool, §14.4)
    candidates = [p for p in profiles if not portfolio.is_disabled(p)]   # Section 3 portfolio
    sampled = []
    for p in candidates:
        mu, sigma = policy_value_and_uncertainty(task, p, pref)    # MLP head + dropout/ensemble var
        sampled.append((p, gaussian_sample(mu, sigma)))           # exploration via posterior
    sampled.sort(key=lambda t: t[1], reverse=True)

    action, branch_count = wider_deeper_policy(task, budget)       # AB-MCTS allocation (Section 9.5)
    chosen = sampled[:branch_count]                                # diversify across top profiles
    effort = effort_policy(task, [p for p,_ in chosen], pref)      # cheap-first cascade (Section 12)

    return AllocationDecision(action=action, target_node_id=task.node_id,
                              branch_count=branch_count,
                              vendor_model=[(p.backend_id, p.model_id) for p,_ in chosen],
                              effort=effort)
```

Online learning is REINFORCE with a batch-mean baseline and entropy regularization ([BaRP](https://arxiv.org/html/2510.07429v1)), updated **off-policy** from `controller_decisions.jsonl` after each solve (never mid-turn — mid-subprocess mutation is infeasible against opaque CLIs and breaks determinism, Section 11.1.1):

```python
def update_policy(decisions: list[ControllerDecision], lr, entropy_coef):
    # one short "episode" per solve; keep episodes short to limit credit diffusion (survey)
    baseline = mean(d.reward for d in decisions)                   # batch-mean baseline
    for d in decisions:
        advantage = d.reward - baseline
        grad = score_function_grad(theta, d.features, d.action) * advantage
        grad += entropy_coef * entropy_grad(theta, d.features)     # keep exploring the open pool
        theta -= lr * grad
```

The reward is the deterministic verifier outcome plus cost, *not* a model-judge score (verifiable beats judgeable; deterministic verifiers resist reward-hacking, Section 13). Stage 0 is the only stage that is *active by default*; Stages 1 and 2 refine it.

### 14.6 Stage 1 — GEPA Reflective Prompt Evolution; Stage 2 — Cost-Penalized RL

#### 14.6.1 Stage 1: reflective evolution of the controller's prompts

The controller — like every component — is partly *prompted* (its routing rationale, its tool descriptions, the instructions it hands workers). [GEPA](https://arxiv.org/abs/2507.19457) optimizes *prompts, not weights*, by reflective mutation over natural-language execution traces plus Pareto-frontier candidate selection. This is the **highest-leverage learning lever** for APEX because (a) APEX's feedback is rich NL (test logs, diffs, reviewer comments), which carry far more bits/rollout than a scalar reward, and (b) prompts-not-weights is *inherently vendor-agnostic* — an evolved prompt works across any backing model, satisfying the foundational frame.

```
gepa_evolve(controller_prompts, val_tasks, budget):
    pool = [controller_prompts]                                   # Pareto frontier of candidates
    while budget.remaining():
        parent = pareto_sample(pool, val_tasks)                   # keep per-instance bests
        traces = run_on_minibatch(parent, val_tasks.sample())     # NL traces, not scalars
        child  = reflect_and_mutate(parent, traces)               # LLM proposes prompt edits
        child  = length_regularize(child)                         # anti-bloat (Decagon: 4x compression)
        if dominates_on_pareto(child, pool, val_tasks):
            pool.append(child)
    return pareto_best(pool, held_out_tasks)                      # separate validation set
```

Guardrails are mandatory and trace to the named pitfall (prompt bloat is the overfitting tell): a **separate** validation set, a **length-regularized** reflection proposer, and anti-overfit instructions so the reflector does not memorize training keywords ([Decagon production notes](https://decagon.ai/blog/optimizing-gepa-for-production): >5000-char prompts are the bloat signal; length regularization gave 4x compression). Stage 1 is `defer` (canonical disposition) — it ships *after* the Stage-0 bandit, because it improves a controller that is already active.

#### 14.6.2 Stage 2: cost-penalized RL over orchestrator decisions (optional, last)

Only when solve volume justifies a training pipeline do we train the controller itself with policy-gradient RL. We adopt the **Puppeteer cost-penalized terminal reward** verbatim in shape:

```
R_T   = correctness - lambda * cost            # terminal; correctness from deterministic verifier
R_t   = gamma * R_{t+1} - lambda * C_t         # recursive, drives early termination
C_t   = F * log(1 + t/phi)                     # per-step cost term (F = FLOPs/token proxy)
# Puppeteer defaults: lambda=0.1, gamma=0.99, episode length ~4, ~200 samples
```

Three pitfalls are honored as hard design constraints:

- **Credit diffusion.** Per-decision signal vanishes as trace length grows under a shared terminal reward. Mitigation: **short episodes** (Puppeteer length ~4), orchestrator-level centralized credit (tractable because the controller is a single centralized policy), and acceptance that we trade some credit resolution for stability.
- **Spawn non-identifiability.** The untaken branch produces a structurally different trace, so spawn/delegate decisions are non-identifiable from on-policy traces alone. Mitigation: learn spawn decisions *off-policy* from `controller_decisions.jsonl` shadow branches, or via [Lemon-style](https://arxiv.org/html/2605.14483) localized counterfactual credit on edited orchestration fields — never from on-policy traces only.
- **GRPO instability.** Do **not** use naive GRPO group-normalization across role/turn-heterogeneous prompts; it is unstable because prompts differ by role and turn ([Dr. MAS / AT-GRPO](https://arxiv.org/html/2510.11062)). If GRPO is used at all, group **within agent and turn** (AT-GRPO). Plain REINFORCE (Puppeteer) sidesteps this and is preferred at our scale.

Stage 2 is `defer` and explicitly optional: **the system is fully functional and SOTA-competitive at Stage 0 + Stage 1**, which is the whole point of staging.

### 14.7 The Decoupled Calibration Module (Calibrate-Then-Act)

The "active" decisions of *when to stop, when to escalate, when to spawn another worker* depend on a calibrated estimate of success probability. The strongest cross-cutting finding in the digest is that **this behavior does not emerge from end-to-end RL on coding tasks** ([Calibrate-Then-Act, arXiv:2602.16699](https://arxiv.org/html/2602.16699v1)); LLMs as self-judges are badly overconfident/miscalibrated. We therefore build calibration as a **separate, defensively-designed module**, structurally decoupled from action selection.

```python
@dataclass
class CalibratedEstimate:
    p_success: float                 # calibrated probability current frontier yields an accepted patch
    p_success_lower: float           # lower confidence bound (used for stop/escalate)
    source: Literal["EXECUTION","CRITIC","MODEL"]   # execution-grounded preferred
    n_signals: int

def calibration_decision(frontier, est: CalibratedEstimate, budget, cfg) -> Literal["STOP","CONTINUE","ESCALATE","SPAWN"]:
    # decoupled: uncertainty estimation (est) is computed independently of this policy
    if any_verified_accepted(frontier):                # Cardinal Contract: a verified win is terminal
        return "STOP"
    if budget.remaining_frac() < cfg.calib.min_budget_frac:
        return "STOP" if est.p_success_lower < cfg.calib.stop_thresh else "CONTINUE"
    if est.p_success_lower > cfg.calib.escalate_thresh and localization_futile(frontier):
        return "ESCALATE"                              # route budget to frontier models (HyperAgent)
    if est.p_success < cfg.calib.spawn_thresh and budget.remaining_frac() > cfg.calib.spawn_floor:
        return "SPAWN"                                 # widen with a decorrelating vendor (Section 9.7)
    return "CONTINUE"
```

The estimator is execution-grounded first (regression-survival, F2P/P2P from real tests), critic-second (the swappable generative critic, discrimination-only, Section 13), model-inferred last and most distrusted. If APEX ever runs Stage-2 RL, we add a proper-scoring-rule (Brier/log) calibration reward term to counter RLVR-induced overconfidence — but the module exists and works *without* RL. This module also feeds Section 9's `CONFIDENCE_FLOOR`: below it, the controller collapses to verified best-of-N.

> Honesty note: calibration of LLM-derived `p_success` is hard and the lower bound is conservative by design. The module is allowed to be *cautious* (over-spend slightly) but is forbidden from being *confident-and-wrong* (it never authorizes acceptance — only allocation). This asymmetry is deliberate.

### 14.8 Input-Hash Cache Validity for Active Routes

An active controller that reuses prior worker results to save cost introduces a correctness hazard the passive controller never had: **replaying a stale result for changed code**. The rule is absolute.

```python
def cache_key(node) -> str:
    return blake2b(canonical_json({
        "repo_state_hash": git_tree_hash(node.checkpoint),   # exact worktree/snapshot state
        "issue_hash":      sha256(issue_text),
        "node_prefix_hash": node.prefix_key,                 # prompt-prefix (Section 9.8)
        "vendor":          node.assigned_vendor,
        "model":           node.assigned_model,
        "effort":          node.effort,
        "tool_schema_rev": EXECUTOR_SCHEMA_REV,              # Section 3 capability negotiation rev
    }))

def maybe_replay(node, cache) -> RolloutResult | None:
    hit = cache.get(cache_key(node))
    return hit if hit is not None else None                  # exact-match ONLY; no fuzzy reuse
```

Any change to repo state, issue, prefix, vendor, model, effort, or executor schema **invalidates** the entry. This is the active-controller analogue of v1's input-hash journaled resume (Section 15) and reuses the same hashing discipline. There is no similarity-threshold cache; a near-match is a miss.

### 14.9 The Decision Journal: `controller_decisions.jsonl`

Every controller decision is appended as one durable record, providing the substrate for reproducible off-policy credit assignment (the explicit canonical mandate, and the thing v1 already partially logs). The journal is append-only, written through the per-`agent()`-call WAL (Section 15), and survives full restart.

```python
@dataclass(frozen=True)
class ControllerDecision:
    decision_id: str                 # deterministic: f"{run_id}:{node_id}:{seq}"
    run_id: str
    node_id: int
    seq: int                         # monotonically increasing within run
    # --- inputs the policy saw (for replay) ---
    task_features: dict              # serialized TaskFeatures
    profile_snapshot: list[dict]     # BackendProfile vectors at decision time
    preference: tuple[float, float]  # (w_quality, w_cost)
    policy_version: str              # theta hash OR "heuristic" OR "gepa:<promptHash>"
    library_enabled: bool            # INVARIANT 3 audit
    applied: bool                    # INVARIANT 2 audit: False => value==baseline
    # --- the action taken ---
    action: str                      # WIDER|DEEPER|DIVERSIFY|STOP|ESCALATE|SPAWN
    vendor_model: list[tuple[str, str]]
    effort: str
    branch_count: int
    sampled_value: float | None      # Thompson sample (None if heuristic)
    # --- realized outcome (filled after the node executes; verifier-authoritative) ---
    verifier_outcome: str | None     # ACCEPTED|REJECTED|REGRESSION_INCONCLUSIVE|ABSTAINED
    quality: float | None            # from deterministic verifier, NOT a model judge
    cost: float | None               # provider-cache-adjusted tokens + latency
    reward: float | None             # w_q*quality - w_c*cost (filled at solve end)
    created_at: float
```

Because `applied`, `library_enabled`, and `policy_version` are all logged, every published run can be *audited* for whether learned control was active and whether it ever violated blend-not-switch — and the journal can be *replayed* to re-derive credit under a new policy without re-running workers (off-policy evaluation).

### 14.10 Evaluation, Ablations, and the Heuristic Floor Guarantee

The controller is only trustworthy if we can prove, per milestone, that it does not regress against the heuristic baseline. The evaluation hooks (detailed in Section 20) are:

| Metric | Question it answers | Pass condition |
|---|---|---|
| Bandit lift vs heuristic | Does Stage 0 beat v1's failover-ranking router? | resolved-rate ≥ baseline AND cost ≤ baseline at equal K |
| GEPA lift vs static prompt | Does Stage 1 beat the hand-tuned controller prompt? | resolved-rate ≥ baseline on held-out set, no prompt bloat |
| Open-pool generalization | Does a *held-out* backend get sane routing? | zero-shot route value correlates with realized quality |
| Calibration error (ECE/Brier) | Is `p_success` calibrated? | ECE below threshold; over-cautious tolerated, over-confident not |
| Off-policy replay fidelity | Can `controller_decisions.jsonl` reproduce credit? | replayed reward matches logged reward on deterministic terms |
| Invariant audit | Did `applied=False` ever change a value? | zero violations of blend-not-switch in the journal |

The **floor guarantee** is the closing safety property and the reason the staged design is safe to ship at every milestone: because the controller fails open to the heuristic baseline (Invariant 2), is master-switchable (Invariant 3), collapses to verified best-of-N below the confidence floor (§14.5, Section 9), and cannot promote an unverified candidate (Invariant 1), the *worst case* of every learned stage is **the v1 passive controller's behavior**. We can never do worse than the baseline; we only sometimes — and measurably — do better.

### 14.11 Cross-References and Vendor-Neutrality Recap

- The controller **drives** the adaptive-branching allocation of Section 9 (it *is* the policy behind `allocate()` and the `(vendor, model)` Thompson axis), the cheap-first effort cascade of Section 12, and the phase boundaries of Section 11.
- It **sits below** nothing it can override: the execution-authoritative selector of Section 13 is the brake; the controller never accepts.
- It **depends on** the durable journaling and input-hash resume of Section 15 (the WAL backs `controller_decisions.jsonl` and the cache).
- It **feeds** the self-improvement and memory loop of Section 17 (abstracted profiles and evolved prompts are durable, abstracted priors — never raw trajectories).

Vendor neutrality is structural, not incidental: the policy keys on *capability/cost profile attributes*, never on backend identity (§14.4); the evolved controller is *prompts, not weights* (§14.6.1); and every worker is invoked through the normalized `Executor` (Section 3), so a node routed to Codex, Claude Code, or any other agent is identical to the controller except for the profile vector it reasons over. The active controller is the brain that makes a mixed-vendor fleet *choose well* — and the Cardinal Contract is the spine that guarantees it can never choose *wrong* in a way that matters.

## 15. Isolation, Determinism & Durable Resumable Runs

This section specifies the **hardened substrate** on which every other APEX-Ω mechanism stands: per-rollout filesystem isolation, best-effort determinism, and a durable, restart-survivable journal that makes the engine's `agent()`/`parallel()`/`pipeline()` primitives (see Section 2) resumable and the speculative tree-search of Section 9 replayable. The design lifts APEX v1's load-bearing invariants verbatim where they are sound (worktree isolation, `fcntl` locks, deterministic snapshot SHAs, the escrow WAL) and *generalizes* two of them — the narrow one-candidate escrow WAL and the unwired `ReplayRecorder` — into a single per-`agent()`-call journaled resume that beats the reference Claude Code implementation's session-scoped resume. Throughout, three claims are held precisely and never overstated: **(a)** determinism is *best-effort around irreducibly-stochastic workers*, never bit-for-bit agent output; **(b)** what is reproduced is **artifacts** (diffs + re-run verification + selected winner), not token streams; **(c)** durable resume is *unbuilt in v1 as a general mechanism* (the v1 ingest is explicit: `ReplayRecorder` has no production callsite and the escrow WAL is a one-candidate backstop), so this section is a **build spec**, not a description of a preserved guarantee.

The unifying principle is **per-unit scoping**: locks, kills, registries, secret boundaries, and journals are all scoped to a single rollout/run, never machine-wide. There is no global mutex anywhere in this design.

### 15.1 Per-Rollout Isolation: Worktrees, Locks, and Three-Tier Degradation

Parallel and speculative workers that edit the same file conflict; the documented fix in the source paradigm and in v1 is **git worktrees — an isolated checkout per rollout**. APEX-Ω keeps v1's hardened version of this verbatim and exposes it as the `isolation:"worktree"` option on the `agent()` primitive (Section 2) and as the implicit default for any file-editing worker spawned by `parallel()`/`pipeline()`.

#### 15.1.1 The lock-before-touch invariant

Every rollout acquires a **per-rollout advisory lock before touching the workspace path**, and releases it on *every* exit path (success or failure). This is the v1 invariant (Section 8.2 of the v1 blueprint) and it is non-negotiable: it is the primitive that makes any parallelism or branching safe (the v1 ingest attributes a CAID improvement of 63.3 vs 57.2 to this isolation discipline).

```
LOCK_PATH = workspace_dir / ".locks" / f"rollout_{rollout_id}.lock"

acquire_rollout_workspace(rollout_id, workspace_dir):
    lock = open(LOCK_PATH, mode="w")            # create lock file FIRST
    try:
        fcntl.flock(lock.fd, LOCK_EX | LOCK_NB) # POSIX; non-blocking
    except BlockingIOError:
        raise ConcurrentWorktreeError(rollout_id)   # never silently share
    # ---- ONLY NOW may we create/mutate the worktree for this rollout ----
    workspace = provision_isolation_tier(rollout_id, workspace_dir)
    return WorkspaceHandle(lock=lock, workspace=workspace, tier=workspace.tier)

# Windows fallback (no fcntl): atomic PID-marker file via O_CREAT|O_EXCL;
#   stale-marker reclaim only if the recorded PID is provably dead.
release_rollout_workspace(handle):  # in a finally: block, always
    fcntl.flock(handle.lock.fd, LOCK_UN); handle.lock.close()
```

The lock is **scoped to the rollout id, not the machine**. Two concurrent solves sharing a `workspace_dir` cannot destroy each other's worktrees, and reaping one stalled rollout never touches its siblings (the per-rollout `RolloutCLIRegistry` binds each worker pid to one `rollout_id`; `SIGTERM->SIGKILL` escalation is scoped to that rollout's pids only). **Pitfall honored:** do **not** take a global/machine-wide mutex — that would serialize the whole fleet and defeat the scaling unlock.

#### 15.1.2 Three-tier degradation (degrade, never crash)

Isolation degrades downward on any failure so the engine stays runnable on dirty or non-git repos. This is the ACP-style "graceful degradation" mandate applied to the filesystem (Section 3):

| Tier | Used when | Diff source of truth | Determinism property |
|------|-----------|----------------------|----------------------|
| `seed_clone` (override) | caller supplies a pre-built clone | clone's git diff | inherits seed's SHA |
| `worktree` (preferred) | clean git repo | `git worktree add` checkout | bit-identical diff vs base SHA |
| `snapshot` | dirty tree / worktree-add fails | deterministic synthetic commit | **deterministic snapshot SHA** |
| `synthetic` | non-git repo | content-hash overlay | content-hash identity |

**Deterministic snapshot SHA (kept verbatim from v1):** the snapshot tier commits with a *fixed* author/committer (`APEX Snapshot <apex@local>`), a *fixed* date (`2026-01-01T00:00:00+0000`), and a commit message containing `source_head8 + dirty_hash8` (sha256 over sorted dirty rel-paths + bytes). Two identical source states therefore produce **bit-identical commit SHAs and bit-identical diff text** — the property that lets verification and replay compare diffs across runs. Note the date is a *constant*, not `Date.now()`; per the no-timestamps-in-orchestration pitfall, the orchestration layer never reads the wall clock.

A `WorktreePool` recycles worktrees (~10x cheaper than create+warmup, per v1) and is the natural home for the warm-pool / CoW-clone optimizations of Section 16; the pool is a *cost* lever and never weakens isolation — every checked-out worktree still takes its own per-rollout lock.

### 15.2 Best-Effort Determinism (What We Pin, What We Cannot)

Determinism is the prerequisite for both sound replay and a learned controller's off-policy credit assignment (Section 14). APEX-Ω pins everything pinnable *around* the stochastic worker and is explicit about the boundary.

**Honesty boundary (adversarial verdict honored).** The adversarial review of speculative search is explicit: "bit-for-bit determinism" overstates what v1 guarantees. We therefore state precisely what determinism means here:

- **Reproducible:** orchestration control flow, dispatch ordering, candidate ordering, cluster reduction order, selection (the deterministic ranking tuple terminates in a content sha1 / `-cluster_id`, *never insertion order*), snapshot SHAs, diff text from identical source states, and atomic artifact bytes.
- **NOT reproducible from scratch:** the worker's token stream (temperature-0 hosted APIs are batch-non-invariant), and therefore the *exact shape of a re-run speculative tree*. These are reproduced only via **replay from the journal** (Section 15.3), never re-derived.

We **reject** the "bit-reproducible agent OUTPUT replay" mechanism as impossible across hosted APIs and instead reproduce **artifacts**: diffs, re-run verification results, and the selected winner.

Concrete determinism rules the engine enforces:

| Source of nondeterminism | Rule | Rationale / SOTA anchor |
|--------------------------|------|-------------------------|
| Worker sampling | `temperature=0.0` default (CLI backends ignore temperature anyway; diversity comes from `(vendor, model)` + prompts, not temp — Section 13) | v1 invariant |
| Candidate / dispatch ordering | sort by `(rollout_id, content_hash)`; reduce clusters in `cluster_id` order | eliminates slot-0 bias |
| Local mutation / mutant scoring | `Random(0)` seeded | v1 |
| Search-control sampling (Thompson, AB-MCTS wider-vs-deeper) | **seed the RNG from a content hash** of the node's accepted-evidence set, not `Math.random()` | adversarial verdict: AB-MCTS Thompson sampling must be content-hash-seeded to remain replayable ([AB-MCTS, arXiv:2503.04412](https://arxiv.org/abs/2503.04412)) |
| Hedge / cancel / preempt decisions | **journal the decision** (Section 15.3); replay reproduces it from the log, never re-derives from timing | Dean & Barroso hedging is wall-clock-triggered and inherently nondeterministic ([The Tail at Scale, CACM 2013](https://cacm.acm.org/research/the-tail-at-scale/)) |
| Wall clock / RNG / network in orchestration | **forbidden in orchestration (workflow) code**; confined to journaled worker "activities" | Temporal: "non-determinism is fatal to replay" ([Temporal docs](https://docs.temporal.io/workflow-execution)) |
| File writes | atomic `mkstemp -> write -> flush -> os.fsync -> os.replace` (readers see old-or-new, never torn) | v1; durable-execution practice |

**Seeding search-control nondeterminism (required for Section 9/14).** Any place the controller samples — Thompson sampling for wider-vs-deeper, an epsilon-greedy bandit arm choice, a tie-break among equal-priority branches — derives its seed as `seed = int.from_bytes(sha256(canonical_evidence_bytes)[:8])`. Because the evidence bytes are themselves journaled and content-addressed, the same evidence yields the same sample on replay. This is the mechanism that lets a *stochastic* search controller remain deterministically replayable.

### 15.3 Durable Input-Hash Journaled Resume

This is APEX-Ω's headline differentiator over the reference implementation, whose resume is session-scoped and "starts the workflow fresh" after a full restart ([Claude Code workflows docs](https://code.claude.com/docs/en/workflows)). The mandate is explicit: **unchanged `agent()` calls return cached results; only edited/new calls re-run — and this survives a full process restart.** The design is the durable-execution template (Temporal event-history + deterministic replay; DBOS Postgres checkpoints; Inngest memoized `step.run`) applied at the granularity of a single worker call.

#### 15.3.1 The per-call journal (generalize escrow WAL/CCEDF to per-node)

We **generalize** v1's escrow WAL (CCEDF: fsync-durable, flock-guarded, monotonic-seq, idempotent exactly-once) from a one-candidate confirmed-pass backstop into a **per-`agent()`-call write-ahead log**. The escrow WAL already proved the hard part — making concurrent, nondeterministically-*timed* operations replay-stable by ordering on a **monotonic seq (not wall-clock)** with idempotent keys so duplicate appends are harmless. We reuse that exact pattern per node.

**Journal record (`JournalEntry`):**

```python
@dataclass(frozen=True)
class JournalEntry:
    seq: int                 # monotonic, max(existing)+1 under flock; ORDER KEY (not wall-clock)
    input_hash: str          # sha256 over the canonical key below — the cache key
    kind: str                # "agent" | "hedge_decision" | "cancel" | "node_expand" | "node_prune"
    # ---- the input-hash key components (all that determine an agent() result) ----
    prompt_canonical: str    # prompt with volatile spans normalized out (see 15.3.2)
    model_id: str            # pinned launcher id, e.g. "claude-opus-4-8[1m]"
    vendor: str              # "claude_cli" | "codex_cli" | "gemini_cli" | ...
    cli_version: str         # vendor CLI version string (from RunManifest, 15.4)
    scoped_inputs_hash: str  # sha256 of the worker's scoped inputs: base snapshot SHA,
                             #   discovery_scope (file_paths/symbols/test_ids), schema, effort
    # ---- the recorded outcome (what replay returns for a cache hit) ----
    result_status: str       # "ok" | "infra_nonresult" | "abstain"
    structured_result: dict  # validated schema object OR final text
    fs_diff_ref: str         # content-addressed blob ref for the produced git diff
    usage: dict              # tokens in/out, cache_read/cache_creation (Section 16 SLO)
    idempotency_key: str     # f"{run_id}::{node_id}::{attempt}"
```

The journal lives at `<run_dir>/journal/calls_wal.jsonl`, appended with the CCEDF discipline. To avoid the O(n)-per-append seq scan (a v1 cost note that "matters only if record volume grew" — and it now does, since every call is journaled), the engine keeps an in-memory `next_seq` counter recovered once on startup by reading the tail, then appends are O(1); the flock still guards cross-process appends.

#### 15.3.2 Resume algorithm: replay cached, re-run edited/new

```
resume_or_run(call_request):
    key = canonical_key(call_request)        # prompt_canonical, model, vendor, cli_version,
                                             #   scoped_inputs_hash  -> sha256
    hit = journal.lookup(key)                # index built from WAL on startup; latest-wins by seq
    if hit and hit.result_status != "infra_nonresult":
        materialize_fs_diff(hit.fs_diff_ref) # restore the produced diff into the worktree
        log("replay", key); return hit.structured_result   # CACHE HIT: no worker spawned
    # MISS (new call) or EDITED call (key changed because prompt/scope/model changed):
    result = executor.spawn(call_request)    # the only place a worker runs
    journal.append(JournalEntry(... key ..., result ...))   # durable BEFORE returning
    return result
```

**Cache-validity semantics (the subtle part the adversarial verdict flagged).** A "cached result" means *replaying the recorded output*, not re-deriving it. The cache key therefore must capture **everything that determines the result**: prompt, model, vendor, CLI version, and the scoped inputs (crucially the **base snapshot SHA** — if the code under the worker changed, the snapshot SHA changes, the key changes, and the call correctly re-runs). This is exactly the "unchanged -> cached / edited -> re-run" contract. Getting the key wrong in either direction is the failure mode: too-coarse a key replays a stale answer against changed code; too-fine a key never gets a cache hit and defeats resume.

**`prompt_canonical` normalization.** The prompt is canonicalized before hashing by stripping a declared *volatile region* (timestamps, session ids, run-local paths) from the *stable prefix* — the same prefix-stability discipline Section 16 needs for provider caching. This serves two ends at once: a stable journal key and maximal provider-cache hit-rate. (Per the cache pitfall, a single byte of drift in the stable region both breaks provider caching *and* spuriously invalidates a journal entry, so the canonicalizer is shared between the journal and the prompt-assembly layer.)

#### 15.3.3 Journaling hedge / cancel / speculative decisions

Speculative search (Section 9) and hedging (Section 16) inject timing- and sampling-nondeterminism into orchestration, which would break replay if left implicit. The fix (adversarial verdict, recommended adaptation B) is to make **every branch expand, prune, hedge-fire, and cancel a journaled decision** keyed by seq:

- A `hedge_decision` entry records *that* a duplicate was fired and *which* branch's result was taken; on replay the engine reads this and reproduces the same winner **without** re-running the deadline timer.
- A `node_expand` / `node_prune` entry records the controller's choice (and the content-hash seed that drove any sampling); replay reconstructs the tree in **seq order** regardless of original timing.

Because ordering is by monotonic seq and keys are idempotent, **duplicate appends are harmless and replay is order-stable** — the exact CCEDF property, now serving the whole tree. This is what makes the adversarial verdict's "speculative search trees can preserve replay" hold *in v1's actual sense* (orchestration/selection determinism + replay-from-journal), and only in that sense.

#### 15.3.4 Idempotency for external side effects (at-least-once reality)

Worker calls that mutate external state (a repo push, a network write) are **at-least-once**, not exactly-once, because a crash can occur after the side effect but before the journal append (Temporal/DBOS/Step Functions all enforce this distinction). Every such activity therefore carries `idempotency_key = run_id::node_id::attempt` so a re-run on resume is a no-op or a safe overwrite. **Pitfall:** never speculate or auto-resume an *irreversible, externally-visible* effect (deleting records, sending mail); the effect taxonomy of Section 9 gates which actions are eligible, and the journal records the attempt so resume can detect a possible duplicate. Storage backend: a local fsync'd JSONL WAL is the portable default (no infra dependency); a Postgres-backed journal (DBOS-style, transactional exactly-once when a step writes the same Postgres) is an **optional** drop-in for high-fan-out deployments, with the documented caveat that Postgres status-row contention bites under hundreds of concurrent appends — partition or batch in that regime.

### 15.4 RunManifest Pinning for Replay

Replay across vendors and hosts is only sound if the environment is pinned. The `RunManifest` (kept verbatim from v1, satisfying the vendor-agnostic mandate's "pin vendor+model+version for replay") is captured side-effect-free and never raises:

```python
@dataclass(frozen=True)
class RunManifest:
    apex_git_sha: str; apex_dirty: bool
    python_version: str; platform: str
    seed: int
    redacted_env: dict[str, str]          # APEX_* only, secrets stripped
    model_versions: dict[str, str]        # alias -> pinned launcher id
    vendor_cli_versions: dict[str, str]   # claude_cli/codex_cli/... -> version (NEW emphasis)
    docker_images: dict[str, str]         # tag -> repo@sha256:digest
    upstream_harness_versions: dict[str, str]
```

**Docker digest pinning** uses v1's `resolve_image` precedence — `@sha256:` short-circuit (return as-is, avoiding the double-pin bug) -> prepinned registry file -> `docker_inspect` (30s) -> bare tag — and records which path won. Image drift silently changing scores is the failure this prevents. For a **heterogeneous fleet**, `vendor_cli_versions` is load-bearing: the journal's `cli_version` component (Section 15.3.1) is read from here, so a replay that finds a different CLI version correctly treats the call as edited and re-runs it rather than replaying an output the new binary could not have produced.

### 15.5 The CI Acceptance Test: Kill Mid-Run -> Identical Resume

Durable resume is only a guarantee if it is *gated by a test*. Borrowing the durable-execution community's standard validation ("test by killing mid-run and confirming identical resumed output"), APEX-Ω adds a **mandatory CI acceptance test** before the durability claim may be asserted:

```
test_kill_midrun_identical_resume:
  1. Start a fixed-seed run on a pinned task (RunManifest captured).
  2. At a deterministic checkpoint (e.g. after node_expand seq==K), SIGKILL the engine.
  3. Restart from the same run_dir.
  4. ASSERT: every pre-kill agent() call is a journal REPLAY (zero workers re-spawned for them).
  5. ASSERT: the resumed trajectory is identical in seq order (same node_expand/prune/hedge log).
  6. ASSERT: the SELECTED WINNER is byte-identical (same content sha1) AND its re-run
            verification reproduces the same accept decision and the same diff text.
  7. ASSERT (negative): editing one node's prompt before restart causes EXACTLY that node
            (and its dependents) to re-run, and nothing else.
```

This test asserts on **artifacts and trajectory**, not token streams — consistent with the honesty boundary of Section 15.2. It is the same property Temporal's replay tests check and is the concrete evidence that "we did better than the reference impl." Until it is green, the plan treats durable resume as *unbuilt* (per the v1 ingest's note that `ReplayRecorder` has no production callsite).

### 15.6 Interaction with the Cardinal Safety Contract and Speculation

A perfectly deterministic, perfectly resumable run can still be **reproducibly wrong** if it prunes a correct candidate before execution evidence exists. The adversarial verdict is explicit here: a deterministic-but-unsafe pruner is *worse* than a nondeterministic one because it is reliably wrong, matching the documented "premature pruning prunes correct paths / PRM reliability decreases with distance from terminal states" failure mode ([Limits of PRM-Guided Tree Search, arXiv:2510.20272](https://arxiv.org/html/2510.20272)). Therefore this section's machinery is subordinate to the Cardinal Safety Contract (Section 13):

- Speculative pruning (Section 9) may only **re-order exploration** or **downweight** branches via journaled, content-seeded decisions; it may **never** terminate a branch on a soft/learned/LLM signal in a way that prevents a would-be execution-verified candidate from reaching the verifier.
- The journal records *which* branches were expanded vs pruned, but selection still runs the full execution-authoritative cascade over whatever survivors reached it; the deterministic ranking tuple places every soft key strictly below every execution key (Section 13).

In short: isolation and determinism make search **safe to run and cheap to replay**; the Cardinal Contract keeps it **honest**. Neither substitutes for the other.

### 15.7 Build Notes for a Vendor-Neutral Implementer (Codex or Claude Code)

A coding agent building this on either backend must observe:

1. **No clock/RNG/network in orchestration code.** All three live only inside journaled worker activities. Lint for `time.*`, `random.*` (except explicitly seeded `Random(content_hash)`), and socket calls in the orchestration module; a violation silently breaks replay (Section 15.2).
2. **Lock first, touch second, release in `finally`.** The `acquire -> provision -> ... -> release` ordering of Section 15.1.1 is the single most common place to introduce a corruption bug; the release must be unconditional.
3. **The journal append must be durable before the result is returned to the caller** (write-ahead), or a crash in the return path loses a completed call's record and forces a needless re-run.
4. **Share one canonicalizer** between the journal key (Section 15.3.2) and the prompt-assembly stable-prefix contract (Section 16) so journal validity and provider-cache hit-rate never diverge.
5. **Wire the journal into `agent()` itself** (the `resume_or_run` wrapper of Section 15.3.2), not into per-stage callers — this is the v1 "promote `ReplayRecorder` into a production per-call journal" seam, and centralizing it is what makes *every* primitive (`parallel`, `pipeline`, speculative trees) resumable for free.
6. **Heterogeneous fleets:** the journal key includes `vendor` and `cli_version`, so a run can replay a Claude-orchestrated, Codex-executed trajectory exactly, and a mid-run vendor swap correctly re-runs only the affected calls.

#### Config keys (defaults chosen to be safe, not cheap)

| Key | Default | Meaning |
|-----|---------|---------|
| `isolation.tier_preference` | `["seed_clone","worktree","snapshot","synthetic"]` | degradation order |
| `isolation.worktree_pool_size` | `cores-2` | recycled worktrees (cost lever, Section 16) |
| `determinism.temperature` | `0.0` | worker sampling |
| `determinism.search_seed_source` | `"content_hash"` | how Thompson/bandit RNG is seeded |
| `journal.enabled` | `true` | durable per-call journaling ON by default (the differentiator) |
| `journal.backend` | `"jsonl_wal"` | or `"postgres"` (DBOS-style, opt-in) |
| `journal.materialize_diffs` | `true` | restore fs diffs on cache hit |
| `resume.require_manifest_match` | `true` | refuse cross-host replay on RunManifest mismatch unless `--force` |
| `resume.on_cli_version_mismatch` | `"rerun"` | `rerun` (safe) \| `replay` (force, unsafe) |

### 15.8 Honest Limitations

- **Not bit-reproducible worker output.** Hosted-API temperature-0 batch non-invariance makes this impossible; we reproduce artifacts, not token streams. Any claim otherwise is rejected.
- **At-least-once external side effects.** Idempotency keys mitigate but do not eliminate the window between a side effect and its journal append; irreversible effects must never be auto-resumed or speculated.
- **Journal validity hinges on the cache key.** A missing key component (e.g. forgetting an effort-level flag) replays a stale answer; the shared canonicalizer and the negative arm of the CI test (Section 15.5, step 7) are the guards.
- **Postgres-journal contention at extreme fan-out** is real (DBOS caveat); the JSONL WAL default avoids it, and the Postgres backend is opt-in with partitioning guidance.
- **Process/memory state is not snapshotted.** v1's isolation is filesystem/git-only by design (the fast path); a worker's live dev-server or language-server state is lost on rollback. Process-memory checkpoint/restore (DeltaBox/Crab-class) is a Section 16 optimization, deliberately out of scope here, and is the one place the filesystem-as-source-of-truth invariant trades capability for cost/simplicity.

These limitations are stated so the engine never *fakes* a guarantee it cannot keep — the fail-loud-never-fake invariant applied to the substrate itself.

## 16. Speed & Cost Engineering

> **Frame.** Speed and cost are not a separate subsystem; they are properties of how the workflow engine (Section 2) schedules `agent`/`parallel`/`pipeline`/`phase`/`budget` primitives over vendor-neutral workers. This section specifies the cost/latency contract APEX-Ω enforces on top of APEX v1's "optimize for SOTA, never for cost" substrate. The governing rule throughout: **speed comes from not *starting* doomed work, never from *killing* working agents** (v1's progress-based liveness, Section 15, is preserved verbatim). Every cost lever here is a *bounded amplifier* of the three capability properties in the Central Thesis (isolation, verify-and-refute, orchestration-as-code) — it may reduce wasted spend, but it may **never** abort an in-flight succeeding rollout (the escrow-WAL/CCEDF invariant, Section 15) and may **never** promote an unverified candidate to make a deadline (the Cardinal Safety Contract, Section 13).

APEX v1 is, by design, the worst-case cost profile: `enable_adaptive_allocation=False` runs a fixed `num_rollouts=5` (no adaptive down-scaling) regardless of difficulty, escalating to the `max_rollouts=16` cap only under wave escalation / portfolio floor; `repo_token_cap=None` and `max_tokens_per_repo_followup=0` leave all token-budget machinery inert; the default backend resolves to `codex_cli:gpt-5.5` (first CLI preference; `claude_cli:opus` is the failover), with `--effort max` applied on the Claude CLI path specifically (codex uses `xhigh`). The remarkable fact — established in the v1 ingest's `change_seams` — is that v1 already *built* nearly every cost-control hook (adaptive allocation, token caps, worktree pool, model-id indirection) and left them **OFF**. APEX-Ω's job is therefore mostly to **flip and wire** these hooks behind a quality SLA, plus add the genuinely new primitives (per-item `pipeline()` streaming, the provider-cache adapter, prefix-stability linting). This is a "do better than the reference impl" mandate executed by configuration and scheduling discipline, not a rewrite.

### 16.1 The Eight Cost Layers and Where the Money Goes

v1's cost stacks multiplicatively across four generation/selection layers plus orchestration overhead. APEX-Ω attacks each with a specific, bounded lever:

| # | Cost layer (v1) | Dominant driver | APEX-Ω lever | Adopted mechanism / disposition |
|---|---|---|---|---|
| 1 | Full-trajectory rollouts (up to 16) | redundant full solves | **Difficulty-adaptive low-K** (default ON) | adopt: "single biggest cost lever" |
| 2 | Per-rollout agent loop length / context | many turns × 80–120k ctx | prefix-stable assembly + provider cache | adopt |
| 3 | Per-rollout setup (worktree create+warmup ~4s) | duplicated provisioning | **warm CoW worktree pool (~10×)** | adopt (v1 `use_worktree_pool`) |
| 4 | N×N cross-validation matrix | O(N²) sandboxed test runs | **clustering-before-matrix + test-impact prune** | adopt-modified (CTDG) |
| 5 | Selection verification cascade | baseline + regression reruns | cheap-first cascade (unchanged) | adopt verbatim (Section 13) |
| 6 | Escalation / progressive waves / 4 follow-up loops | "spend more on partial progress" | **futility gate + budget kill-switch** | adopt |
| 7 | Optional LLM selection layers (voters/amplifier/reviewers) | contested-tie LLM calls | cost-bounded mode disables, deterministic fallback | adopt (already gated) |
| 8 | Subprocess retry/salvage (≤4 attempts) | flaky infra | unchanged (happy path pays ~0) | keep v1 |

**Wall-clock**, separately, is dominated by the *longest single rollout* plus serial selection. v1 runs K rollouts (default 5, up to the 16 cap) at `parallel_workers=3` ≈ `ceil(K/3)` sequential batches × slowest-rollout, then serial O(N²) cross-validation. The two biggest wall-clock wins are (a) lowering K so fewer batches run, and (b) `pipeline()` streaming so a stage-2 worker starts the moment a *single* item clears stage-1, rather than waiting for the slowest stage-1 worker (Section 16.3).

### 16.2 Difficulty-Adaptive Low-K Allocation (default ON)

This is the headline change versus v1. v1's adaptive path is **fully wired but disabled** (`estimate_difficulty → compute_rollout_count → evaluate_policy_model('planning.rollout_count') → _clamp_rollout_bucket`); APEX-Ω makes it the default and binds it to a quality SLA rather than full-cap. Evidence: Snell et al. compute-optimal scaling shows 2–4× compute savings at matched quality, and the optimal number of independent attempts K is **often < 10** even on hard tasks; pure best-of-N coverage scales only log-linearly (Large Language Monkeys, [arXiv:2407.21787](https://arxiv.org/abs/2407.21787)) so the 9th–16th trajectory buys almost nothing on most issues. Self-consistency's edge is *diminishing* on modern models ([arXiv:2511.00751](https://arxiv.org/html/2511.00751): 0.4–1.6% over 20 samples).

```python
# allocation.py — runs once per task, after Phase-2 localization (amortized)
RolloutBuckets = [1, 4, 8, 16]   # v1 quantization, kept

def select_K(task, ctx, cfg) -> AllocationDecision:
    d = estimate_difficulty(task, ctx)          # 0..1; reuse v1 estimate_difficulty
    # difficulty -> bucket thresholds (v1 Section 5.2)
    if   d <= 0.25: k = RolloutBuckets[0]        # easy: 1 (+ speculative-first cheap attempt)
    elif d <= 0.55: k = RolloutBuckets[1]        # 4
    elif d <= 0.80: k = RolloutBuckets[2]        # 8
    else:           k = RolloutBuckets[3]        # 16 (thin-feedback / hard floor)

    # Portfolio floor RAISES only (v1 invariant): guarantee >=1 of each distinct
    # (vendor, model) profile we want for diversity decorrelation (Section 3 / 12).
    k = max(k, portfolio_rollout_floor(cfg))     # never lowers below diversity floor

    # Below the feedback-confidence floor we cannot trust difficulty -> collapse to
    # the verified best-of-N FLOOR we can never do worse than (Central Thesis).
    if feedback_confidence(ctx) < cfg.feedback_floor:
        k = RolloutBuckets[3]                     # full-cap as the thin-signal backstop
    return AllocationDecision(k=k, difficulty=d, reason=...)
```

Config keys (all promoted from v1 seams):

```yaml
allocation:
  enable_adaptive_allocation: true        # v1 default False -> ON
  rollout_buckets: [1, 4, 8, 16]
  feedback_floor: 0.35                     # below this, collapse to full-cap
  portfolio:
    min_distinct_profiles: 2               # diversity floor (cross-vendor; Section 3)
    max_distinct_profiles: 6               # cap profile budget when cost-bounded
  quality_sla:                             # adaptive must hold these vs full-cap ablation
    max_resolve_rate_delta: 0.01           # gate adaptive ON only if within 1pt of full-cap
```

**Honesty / pitfall.** Snell's 2–4× is *matched-quality on math-reasoning regimes*; repo-SWE difficulty estimation is noisier. APEX-Ω therefore (1) keeps the **full-cap path as the thin-feedback floor** so a mis-estimated hard task degrades to v1 behavior, not to under-spend; (2) gates the *default-ON* decision on the `quality_sla` ablation in Section 20 (adaptive must land within 1pt resolve-rate of full-cap on a contamination-resistant split); and (3) keeps the portfolio floor *raise-only* so the diversity axis that decorrelates hallucinations (Devlo/TRAE cross-vendor, Section 3) is never sacrificed for cost. The early-localization-futility gate (Section 16.6) further routes the freed budget toward *surviving* hypotheses rather than blindly cutting K.

### 16.3 `pipeline()` Per-Item Streaming — the One Net-New Primitive

v1 runs the rollout as a hard-coded staged trajectory (reproduce → localize → patch → verify) but materializes each stage as a *batch* across rollouts: stage N+1 waits for the slowest worker of stage N. `pipeline()` is the genuinely new engine primitive (accepted as **adopt**): it streams *per item* so each item flows stage→stage independently.

The cost identity is the whole point:

```
batch-staged wall-clock   = Σ_stages  max_item( stage_latency )      # sum-of-slowest-per-stage
pipeline per-item streamed = max_item( Σ_stages item_stage_latency ) # slowest SINGLE chain
```

For a 4-stage chain where each stage's slowest item differs (the common case), this collapses wall-clock from the *sum of four per-stage maxima* to the *single slowest end-to-end chain*.

```python
# engine/pipeline.py — vendor-neutral; workers are opaque Executors (Section 3)
def pipeline(items, stages, *, workers, budget) -> list[Result]:
    """
    stages: ordered list of Stage(name, fn, worker_profile, timeout_policy)
    Each item advances independently; a free worker pulls the next ready item
    from the earliest non-empty stage queue (longest-shared-prefix-first; 16.5).
    """
    queues = [Queue() for _ in stages]; queues[0].extend(items)
    results = []
    with WorkerPool(workers) as pool:
        while not all_drained(queues) and budget.alive():
            stage_idx, item = pick_ready(queues, policy=LONGEST_SHARED_PREFIX_FIRST)
            stage = stages[stage_idx]
            fut = pool.submit(stage.fn, item, profile=stage.worker_profile)
            on_done(fut, lambda r:
                (queues[stage_idx+1].put(r) if stage_idx+1 < len(stages)
                 else results.append(r)))
            journal.append(StageEvent(item.id, stage.name, attempt=item.attempt))  # 15.x WAL
    return results
```

Per-item streaming composes with everything else: a fast-localized item can be *patching* while a slow item is still *reproducing*, and the patcher worker can be a *cheaper* tier than the navigator (Section 12 model economy). Crucially, `pipeline()` changes only *scheduling order*, never *acceptance*: every streamed item still terminates in the unchanged cheap-first verification cascade, so streaming cannot leak an unverified pass.

### 16.4 Prefix-Stability Discipline (portable across all providers)

The single highest-leverage, fully-portable rule: **structure every worker prompt so the largest possible prefix is byte-identical across forks and turns**, then schedule to hit hot caches. This is what makes branching's constant-factor saving real (the adversarial verdict on speculative branching is explicit: the win "comes from prefix/KV-cache reuse, not the tree per se," and is *bounded constant*, not exponential — [Tree-GRPO arXiv:2509.21240](https://arxiv.org/abs/2509.21240), [RadixAttention arXiv:2312.07104](https://arxiv.org/pdf/2312.07104)).

**Prompt-assembly contract.** Every `agent()` template emits two regions:

```
[ STABLE  ] tooling defs + system prompt + policies + repo-invariant context
[ VOLATILE] task + scoped hypothesis + live context + per-rollout discovery_scope
```

**Lint rules (build-time + assembly-time, fail-loud).** The stable region is forbidden from containing anything that drifts. One byte of drift = total cache miss + write premium ([Don't Break the Cache, arXiv:2601.06007](https://arxiv.org/html/2601.06007v2)).

| Forbidden in STABLE region | Why | Enforcement |
|---|---|---|
| timestamps / dates / `now()` | drifts every call | regex lint, hard error |
| UUIDs / session IDs / `rollout_id` | per-rollout divergence | regex lint, hard error |
| dynamic tool definitions reordered | invalidates tools→system→messages | canonical sort of tool defs |
| summarized/pruned tool history | mutates mid-loop | summaries go in VOLATILE only |

**v1 conflict that must be resolved (from the adversarial verdict).** v1 *deliberately* injects per-rollout-unique state into prompts — air-gapped HOME, per-rollout lock paths, `discovery_scope` drawn from the `EpisodicMemoryBus` excluding the caller's own `rollout_id`, snapshot commit messages embedding `source_head8+dirty_hash8`. These are the *exact* anti-pattern for prefix caching. APEX-Ω resolves this by **physically separating** the stable shared-prefix region (identical across all rollouts of a task) from the per-rollout-divergent scoping, which is appended *strictly in the VOLATILE tail*. The blackboard's abstracted negative constraints (Section 11) likewise land in VOLATILE, never in the cached prefix.

#### 16.4.1 Provider-Cache Adapter (declare-stable-prefix API)

Because APEX-Ω is vendor-neutral it cannot assume any single provider's KV internals. The adapter exposes a uniform **`mark_cacheable(span)`** API compiled per-provider:

| Provider | Mechanism | Compiled behavior | Verified economics |
|---|---|---|---|
| Anthropic (Claude Code) | explicit `cache_control` breakpoints | ≤4 breakpoints at STABLE/VOLATILE boundary; 1h TTL (write 2.0×) for hot shared prefixes, else 5m (1.25×) | cache read **0.10× input (90% off)**; min 1,024 tok (Opus/Sonnet 4.x) |
| OpenAI (Codex) | auto-cache >1,024 tok, routes by first ~256-tok hash | keep STABLE prefix byte-stable; pin `prompt_cache_key` per task | ~50% off cached tokens |
| Gemini CLI | auto-cache | identical stable-prefix discipline | provider-reported |
| self-hosted (SGLang/vLLM) | RadixAttention / block-hash | optional: target true cross-fork KV reuse | up to 5–6.4× throughput on prefix-heavy work |

```python
class ProviderCacheAdapter(Protocol):
    def mark_cacheable(self, prompt: AssembledPrompt) -> ProviderPayload: ...
    def read_cache_metrics(self, resp) -> CacheMetrics:
        # cache_read_tokens, cache_creation_tokens, uncached_tokens
        ...
```

**SLO + degrade-gracefully (mandatory pitfall mitigation).** *Do not assume server-side prefix caching is on.* APEX-Ω tracks `cache_read_tokens` vs `cache_creation_tokens` as a first-class fleet SLO per `(vendor, model)` and **detects whether caching actually fired**:

```yaml
cache_slo:
  min_cache_read_ratio: 0.50        # below this on a stable-prefix run -> WARN + investigate
  below_min_token_guard: 1024       # prompts under threshold: DO NOT request caching
  on_cache_miss: degrade            # cap fan-out, fall toward sequential; never assume the win
```

Two honesty caveats baked into the design: (1) caching **below the min-token threshold is associated with ~10–18% TTFT variance** ([arXiv:2601.06007](https://arxiv.org/html/2601.06007v2)) — the `below_min_token_guard` suppresses cache requests for short prompts; (2) a pure API consumer **cannot literally share KV tensors across forks** — against black-box provider caches APEX-Ω can only maximize *hit-rate* via byte-identical prefixes + dispatch ordering, and the docs/comments must never promise true cross-fork reuse on hosted APIs.

### 16.5 Dispatch Ordering: Longest-Shared-Prefix-First (KVFlow-style)

Given the prefix-stability contract, *the order in which forks are dispatched* determines provider-cache hit-rate even when APEX-Ω has zero control over the KV store. APEX-Ω schedules the workflow graph KVFlow-style ([arXiv:2507.07400](https://arxiv.org/html/2507.07400v1)): assign each pending branch a "steps-to-execution" priority and **dispatch longest-shared-prefix-first / depth-first** so a freshly-warmed prefix is immediately reused by sibling forks before it ages out of the provider's lookback window (Anthropic 20-block lookback; OpenAI first-~256-tok routing).

```python
def pick_ready(queues, policy):
    ready = collect_ready(queues)
    if policy is LONGEST_SHARED_PREFIX_FIRST:
        # group by stable-prefix hash; serve the largest hot group first so its
        # cache entry is reused before eviction. Ties -> depth-first (finish chains).
        return max(ready, key=lambda it: (hot_group_size(it.prefix_hash),
                                          it.depth))
    ...
```

For self-hosted serving APEX-Ω can additionally target SGLang for genuine cross-fork KV reuse and KVFlow-style eviction (evict far-off branches first, prefetch soon-to-run). For hosted CLIs this is purely a dispatch-ordering win — portable, free, and it degrades to no-op (never worse) if caching is off.

### 16.6 Futility / Token-Snowball Detection (start-less, not kill)

The most important honesty point in this section: **failures cost more than successes.** SWE-Effi documents the "token snowball" — off-track runs cost **4×+** a success ([arXiv:2509.09853](https://arxiv.org/pdf/2509.09853)). v1's hidden cost amplifier is exactly this: on partial progress the controller *adds* waves (escalation cap 20, progressive waves cap 6, follow-up loops cap 24, near-miss ×3 multiplier) with tokens uncapped. The lever is to **not start doomed work**, never to wall-clock-kill a working agent (v1's progress-based liveness is preserved verbatim — the four kill paths remain K1-stall, emergency-silence, preempt, dead-future; no `wallclock_deadline`).

Two gates, both **routing decisions made at turn/checkpoint boundaries**, never mid-subprocess (mid-subprocess prompt mutation is infeasible against opaque CLIs and breaks determinism/replay):

1. **Early localization-futility gate** (adopt). After localization is amortized once, route the next wave's budget *toward surviving hypotheses* rather than spawning K identical patch attempts on a dead frontier. This kills the "15/16 doomed at localization" waste *before* the patch loop. It **informs allocation; it never suppresses a candidate without execution evidence** (that would invert the Cardinal Contract — see Section 13 and the rejected "plan scoring as a hard prune").

2. **Token-snowball detector** (futility-based early termination, EET-style, [arXiv:2601.05777](https://arxiv.org/html/2601.05777): ~31.8% avg cost cut, negligible resolution loss). Detect a run *unlikely to produce usable output* and **stop spawning further turns for that one rollout** — at a checkpoint boundary, on progress-derived signal, not wall-clock.

```python
# futility.py — evaluated only at checkpoint boundaries (turn end / stage end)
def snowball_score(rollout, history) -> float:
    # progress-derived signals; ALL fail-open (can only delay a stop, never accelerate)
    s = 0.0
    s += w1 * no_diff_progress_turns(rollout)          # turns since last worktree edit
    s += w2 * repeated_blocker_count(rollout, history)  # same error N times (v1: stop after 3)
    s += w3 * token_burn_vs_p75(rollout)                # this rollout's burn vs fleet p75
    s -= w4 * verified_partial_progress(rollout)        # DOWN-weight if tests improving
    return s

def maybe_stop_spawning(rollout, history, cfg) -> bool:
    if has_confirmed_pass(rollout):        # CCEDF guard: NEVER stop a succeeding rollout
        return False
    return snowball_score(rollout, history) >= cfg.snowball_threshold
```

**Hard invariants (the three pitfalls, encoded):**

- *Do not flat-wall-clock-kill a working agent.* `maybe_stop_spawning` fires only on progress-derived futility at a checkpoint; a long legitimate thinking turn or CPU-busy test run is never stopped (v1 liveness preserved).
- *Do not let a budget cap abort an in-flight succeeding rollout.* `has_confirmed_pass()` (escrow-WAL/CCEDF lookup, Section 15) short-circuits every stop. A confirmed full-scope pass is durable and final.
- *Do not assume the snowball signal is correct.* All sub-signals fail-open; the detector can only *stop spawning new turns for one rollout*, never invalidate work already done.

### 16.7 Budget Kill-Switches & Cascade Routing (inside the loop)

v1's token-cap plumbing (`repo_token_cap`, `max_tokens_per_repo_followup`, `_cap_followup_rollouts_for_token_budget`, p75 cost estimation) is **built but inert**. APEX-Ω exposes it as an **opt-in `budget{}` primitive, defaulted unbounded** (honoring v1's "never optimize for cost" stance as the default, while making cost a first-class *optional* objective).

```yaml
budget:                                  # opt-in; defaults below = v1 behavior (unbounded)
  max_tokens_per_task: null              # null = unbounded (v1 default)
  max_usd_per_task: null
  max_followup_waves: 6                  # v1 cap
  kill_switch:
    scope: spawn_only                    # caps gate NEW spawns; never abort in-flight
    spare_confirmed_pass: true           # MANDATORY: a cap can never drop a verified pass
  early_termination:
    enable_snowball_detector: true
    snowball_threshold: 0.70
```

The kill-switch semantics are the load-bearing safety property: **a budget cap gates *new* spawns and *new* follow-up waves; it can never abort a rollout that is in flight, and it always spares a confirmed pass.** This is the direct resolution of the adversarial verdict's v1 tension — cost becomes an opt-in `budget{}` primitive, defaulted unbounded, never a gate that drops a verified pass (which would collide with the escrow-WAL invariant). Budget accounting is sourced from the per-vendor `CostLedgerEntry` sub-accounts defined in Section 12 (USD-normalized, because token units differ across tokenizers and Claude subscription `-p` draws a separate Agent-SDK credit pool from 2026-06-15); the kill-switch evaluates against estimated USD, never against a raw token count, and never against acceptance.

**Cascade routing** puts cheap executors on the happy path (Section 12 model economy; verdict: **adopt-modified, verification-gated**). Cheap/read-only sub-roles (reproducer, localizer, run/verify, narrow single-tool edits) route to a cheaper tier; the frontier tier stays on navigation and multi-file editing (HyperAgent ablation: cheapening navigation/editing causes the worst resolve-rate drops, [arXiv:2409.16299](https://arxiv.org/html/2409.16299v1)). Escalation is **cascade, not blind routing** — try cheap, escalate to frontier on the *first verify-on-diff failure*, with a rewrite-cycle cap — which both fits APEX's verify-on-diff loop and avoids xRouter's documented brittleness of static routing trees ([arXiv:2510.08439](https://arxiv.org/html/2510.08439v1)). The honesty caveat (modeleconomy verdict `partially_sound`): measure **cost-per-resolved-task net of verification** (including the N×N matrix), not gross executor tokens — the "almost-right trap" means a thin executor needing 3–4 retries can cost *more* than one frontier pass.

### 16.8 Warm CoW Worktree Pools & Snapshot-Restore Sandboxes

v1 already ships `WorktreePool` (`use_worktree_pool`): pre-warmed per-`(task, base_commit)` worktrees recycled via `git reset` are **~10× cheaper** than create+warmup (~4s each). APEX-Ω keeps this and hardens the seam the v1 ingest flagged: the pool is silently *defeated* when any request carries a `workspace_seed` (different baselines defeat pre-warming), which is exactly what the seed-carrying escalation/recovery paths do on long runs. APEX-Ω therefore maintains **per-baseline pools** so escalation waves still hit a warm pool rather than falling back to cold creates.

```yaml
worktree_pool:
  use_worktree_pool: true
  shared_object_store: true            # one fetch, N worktrees (Augment pattern)
  warm_pool_size_per_baseline: 4       # absorb K-wide bursts without cold creates
  cow_clone: reflink_or_btrfs          # millisecond CoW from golden image when available
  preserve_pool_on_seeded_requests: true   # fix: don't disable pool on escalation seeds
```

This makes the **millisecond fork substrate** that makes branching net-positive: warm CoW worktrees over a single shared git object store (one fetch, N working dirs) provision in milliseconds vs seconds. Where real runtime isolation is needed (ports/processes/kernel), layer worktrees inside snapshot-restored sandboxes (Firecracker snapshot memory-map+resume, tens of ms vs ~125ms–1s cold boot; warm pools still pay off for burst absorption).

**Honesty / pitfalls (do not over-claim):** practitioner snapshot numbers (28ms restore; 59ms p95 create→exec) are *indicative*, not peer-reviewed — real end-to-end was 2.7s p95 in one case study before optimization. Firecracker needs KVM (absent on macOS/many CI). Pure CoW wastes page-cache at high density (OverlayFS shares it). Git worktrees give **no** runtime isolation — they must be layered inside containers/microVMs for ports/processes, which is exactly why APEX-Ω keeps v1's per-rollout `fcntl`-locked worktree isolation (Section 15) as the *floor* under any faster substrate. The branching-economics verdict is explicit: without a millisecond-class fork substrate, re-running can be *faster* than forking (naive CRIU/E2B: hundreds of ms to seconds) — so the pool is the precondition that makes adaptive branching net-positive, not a guaranteed win on its own.

### 16.9 Bounding the O(N²) Cross-Validation Matrix

The N×N cross-validation matrix (each candidate's tests run on every other candidate's worktree) is the quadratic selection-cost hotspot. v1 already bounds it with **two-pass semantic clustering before the matrix** (exact AST fingerprint, then single-linkage merge at `ast_similarity_threshold=0.95`), which dedups behaviorally-equivalent patches so N = *clusters*, not raw candidates. APEX-Ω keeps clustering-before-matrix and adds **test-impact pruning** (CTDG as test prioritizer, verdict **adopt-modified**) to shrink the per-cell test set:

- **Reorder + dynamic-coverage prune** the tests each cell runs (reordering has *zero* false-negative risk; dynamic coverage is near-safe), then **full-suite backstop** keeps it honest. Static-AST gating is **rejected** (PyCG ~70% recall; reflection/monkeypatch/fixtures invisible; gating silently drops fault-revealing tests, violating execution-authority).
- For large surviving pools, optionally **sample pairs** rather than full N×N, but only above a pool-size threshold and always with the deterministic ranking as the tie-break floor.

```yaml
cross_validation:
  cluster_before_matrix: true
  ast_similarity_threshold: 0.95
  test_impact_prune: reorder_and_dynamic_coverage   # never static-AST gate
  full_suite_backstop: true                          # honesty backstop on the winner
  pair_sampling_above_n: 12                           # sample pairs only for large pools
```

This bounds the dominant quadratic term without ever silently dropping a fault-revealing test — the prune is *near-safe by construction* (reorder = lossless; dynamic coverage = observed, not predicted) and the backstop catches the residual.

### 16.10 Batch API for Non-Interactive Fleet Work

Non-interactive fleet work — benchmark evals, bulk refactors, test generation, backfills — routes to the async **Batch API** (OpenAI/Anthropic: **50% off** input+output for ≤24h async). This **stacks with caching**: batch + cached prefix reaches **~95% combined savings**. Only user-blocking / interactive work stays on the real-time path.

```yaml
execution_lane:
  interactive:  realtime          # user-blocking solves
  fleet:        batch_api         # evals, bulk refactors, test-gen, backfills
  batch_api:
    max_wait_hours: 24
    stack_with_cache: true        # prefix-stable + batch -> ~95% off
```

This is the cleanest "free" lever for the evaluation matrix (Section 20) and any bulk maintenance run: the same prefix-stability discipline that helps interactive caching makes batch+cache compound.

### 16.11 Putting It Together: The Cost/Latency Decision Flow

```
per task:
  1. amortize Phase-1 RepoContext + Phase-2 localization ONCE (v1; ~15% up-front)
  2. K = select_K(task)                          # adaptive low-K, portfolio floor raise-only (16.2)
  3. assemble prompts: [STABLE | VOLATILE]       # prefix-stable, lint-enforced (16.4)
       -> ProviderCacheAdapter.mark_cacheable()  # 90% off cached reads when it fires
  4. pipeline(items=K seeds, stages=[reproduce, localize, patch, verify],
              workers=warm_cow_pool,             # ~10x cheaper provisioning (16.8)
              dispatch=LONGEST_SHARED_PREFIX_FIRST)   # KVFlow ordering (16.5)
       per turn/checkpoint:
         - cascade-route stage to cheapest safe tier; escalate on verify-fail (16.7)
         - if maybe_stop_spawning(rollout): stop SPAWNING new turns   # futility (16.6)
         - track cache_read vs cache_creation as SLO; degrade if miss  (16.4.1)
         - budget kill-switch gates NEW spawns only; spares confirmed pass (16.7)
  5. cluster-before-matrix -> test-impact-pruned N×N cross-validation (16.9)
  6. cheap-first verification cascade -> Cardinal Contract selection (Section 13)

INVARIANTS (never violated for speed/cost):
  * progress-based liveness only; NO wall-clock kill of a working agent
  * a budget cap NEVER aborts an in-flight succeeding rollout (CCEDF)
  * cost levers re-rank/route/allocate; they NEVER promote an unverified candidate
  * best-of-N (full-cap) is the FLOOR we can never do worse than (thin-feedback backstop)
```

### 16.12 What We Explicitly Do *Not* Do (rejected for cost reasons)

| Tempting cost lever | Disposition | Why rejected |
|---|---|---|
| Flat wall-clock kill of slow agents | **reject** | kills slow-but-legitimate runs; v1's deliberately-avoided failure mode. Speed = not *starting* doomed work. |
| Budget cap that aborts in-flight rollouts | **reject** | collides with escrow-WAL/CCEDF; would drop confirmed passes. |
| Static-AST CTDG as a test-pruning *gate* | **reject** | ~70% recall; silently drops fault-revealing tests; violates execution-authority. |
| Pre-execution plan scoring as a *hard prune* | **reject** | false-negative pruning suppresses correct-but-unverified plans before evidence; inverse Cardinal-Contract violation. |
| Non-adaptive fixed-K default (5, up to the 16 cap), caps OFF | **reject** | the headline cost pathology; replaced by adaptive low-K + budget-aware deepening (full-cap kept only as thin-feedback floor). |
| Assume server-side prefix caching is on | **reject** | must detect (cache_read vs cache_creation SLO) and degrade gracefully; promising true cross-fork KV reuse on hosted APIs is dishonest. |
| Heavy-orchestrator + *thin* executor as the default on hard repo SWE | **reject** (verdict `partially_sound`) | HyperAgent: cheapening navigation/multi-file editing causes worst resolve drops; the almost-right trap can cost more than one frontier pass. Cheapen only narrow run/verify/single-tool sub-roles. |
| Bit-reproducible agent *output* replay | **reject** | impossible across hosted APIs (temp-0 batch non-invariance); reproduce *artifacts* (diffs + re-run verification), not token streams. |

The net effect: APEX-Ω turns v1's "non-adaptive fixed-K (default 5, up to the 16 cap), caps off" into a difficulty-adaptive, prefix-cached, pipeline-streamed, futility-gated fleet whose cost levers are all *bounded amplifiers* of the execution-authoritative kernel — every one of which degrades gracefully to v1 behavior, and none of which can ever make a deadline by shipping an unverified patch or by killing an agent that is still working.

## 17. Self-Improvement & Memory

This section specifies how APEX-Ω gets *better over time* without ever compromising the substrate guarantees of Section 15 (isolation, determinism, durable journaling) or the Cardinal Safety Contract of Section 13 (execution evidence is authoritative; soft signals re-rank within an execution-verified tier or downgrade an already-accepted candidate, never promote an unverified one). Self-improvement here is a **bounded amplifier** of the three load-bearing properties from Section 1 — execution-grounded verify-and-refute, context isolation, and deterministic orchestration-as-code — not a new source of authority.

The section owns two distinct systems and the substrate that joins them:

1. **The staged controller's learning loop** — *how* the active controller of Section 14 improves its policy: **bandit now → GEPA-style reflective prompt evolution → RL (volume-gated)**, in that strict order, each rung gated. Section 14 owns the controller's *runtime* shape (the `BackendProfile`/`TaskFeatures` features, the `AllocationDecision` action space, blend-not-switch, `library_enabled`). Section 17 owns its *learning* — the update rules, the reward, the gates, and the off-policy substrate.
2. **Abstracted episodic memory** — how insights and skills are distilled from runs and re-injected, evolving v1's `EpisodicMemoryBus` and `extract_durable_insights` (`rollout/engine.py`) into a relevance-and-compactness-gated store that **never replays raw trajectories**.

The governing design stance: **language-feedback learning and abstracted memory are the high-leverage, low-risk levers we ship; scalar-reward RL over orchestration decisions is deferred behind hard volume and verifier-quality gates** ([GEPA, ICLR 2026 oral](https://arxiv.org/abs/2507.19457); [RL-for-LLM-MAS survey, arXiv 2605.02801](https://arxiv.org/html/2605.02801v1)). Everything below is offline and journal-driven; no learning step ever mutates a live run except through the fail-open blend-not-switch policy.

### 17.1 The staged controller — three rungs, each gated

The active controller (Section 14) is a policy `π(decision | features)` that, at each orchestration decision point (which `(vendor, model)`, which strategy axis, depth/K allocation, when to escalate, when to stop) emits an action. The SOTA evidence is unambiguous that for a vendor-agnostic, budget-constrained system the right *learning* path is staged, not a single end-to-end RL bet. The disposition mirrors the Fusion Ledger (Section 18): **bandit = adopt now; GEPA = defer to Stage 1; full RL = defer to Stage 2 (volume-gated)**.

| Stage | Mechanism | When it ships | Learning signal | Vendor-neutrality | Primary risk |
|---|---|---|---|---|---|
| 0 | Contextual bandit / lightweight policy-gradient router | Day one (active control from launch) | Online partial feedback: verifier pass/fail, regression survival, lint, build, reviewer accept, cost | Inherent — chooses among `(vendor, model)` profiles | Cost-over-conservatism; slow linear convergence |
| 1 | GEPA-style reflective prompt evolution of orchestrator instructions + tool descriptions | After bandit is stable and journal volume exists | Natural-language traces (test logs, diffs, reviewer comments) + Pareto-frontier selection | Inherent — **changes prompts, not weights** | Prompt bloat / overfitting |
| 2 | Full RL (REINFORCE/GRPO) over orchestrator decisions emitting parsable NL plans | Only when volume **and** a hardened verifier justify it | Cost-penalized terminal reward keyed on deterministic verifiers (RLVR) | Plans are NL structures, portable | Credit diffusion; reward hacking → sabotage |

Why this order and not "just do RL": [GEPA](https://arxiv.org/abs/2507.19457) beats GRPO by ~6pp on average (up to 19pp) using **up to 35× fewer rollouts**, and beats the MIPROv2 prompt optimizer by >10pp. Natural-language traces carry far more bits per rollout than a scalar reward — which is *exactly* APEX-Ω's situation, where every run produces rich test/lint/CI/review feedback. The survey ([arXiv 2605.02801](https://arxiv.org/html/2605.02801v1)) further warns that outcome-reward RL over long, dynamic multi-agent traces suffers severe credit diffusion and that spawn decisions are non-identifiable from on-policy traces. RL is therefore the *last* rung, fenced behind explicit gates, not the headline.

**Each rung strictly dominates the prior and preserves every Section-14 invariant.** Turning Stage 1 or 2 off must leave a working Stage-0 system; turning the whole library off (`library_enabled=False`, Section 14.2) must leave a working *heuristic* system whose published number is unchanged. This is the same staging discipline the controller section commits to, viewed from the learning side.

#### 17.1.1 Stage 0 — the bandit router learning rule (active control on day one)

Stage 0 is the day-one active control surface. Section 14 specifies its decision interface and its open-pool `BackendProfile`/`TaskFeatures` representation (it routes over learned capability/cost *attributes*, never one-hot vendor names, so a freshly added backend is valued by *describing its profile*). Section 17 specifies the **online learning rule** that updates that policy from the verifier signal APEX-Ω already produces.

The reward follows [BaRP](https://arxiv.org/html/2510.07429v1)'s preference-conditioned multi-objective form, so a *single* policy serves many cost-quality tradeoffs via a per-request preference vector `w = (w_quality, w_cost)` sampled per decision:

```
quality  = 1.0 if VerificationResult.accepted else partial_axis_score   # deterministic verifier ONLY
cost     = normalized(token_cost + wall_cost) for this decision's subtree
r        = w_quality * quality - w_cost * min(cost / tau, 1.0)
```

`quality` is keyed on the **deterministic** verifier outcome (`VerificationResult.accepted`, Section 13) — *never* on a soft scorer. This is the same anti-false-positive discipline that protects selection, applied to learning: a soft signal can re-rank within a verified tier or downgrade, but it can never become the learning target (if it did, the policy would learn to satisfy the soft signal, which is the inference-scaling false-positive failure mode moved into training).

The policy is a **non-linear MLP** over `concat(task_features, profile_vector, w)`, trained with REINFORCE + batch-mean baseline + entropy regularization. The MLP (not a linear bandit) is deliberate: BaRP's ablation shows REINFORCE-MLP (0.7432) beats LinTS (0.6430), LinUCB (0.6166), and ε-greedy (0.6556) — **linear bandits underperform on heterogeneous query distributions and converge slowly**, which is the SWE regime.

```python
# controller/learn_bandit.py  (offline, journal-driven; runs between runs)
def bandit_update_step(theta, journal_batch, cfg):
    """One REINFORCE step over a batch of journaled decisions.
    journal_batch: list[DecisionRecord] read from controller_decisions.jsonl,
                   each already joined to its realized DecisionOutcome (verifier + cost).
    """
    grads, rewards = [], []
    for rec in journal_batch:
        x = concat(rec.task_features, profile_vec(rec.arm), rec.pref_vector)
        logits = mlp_forward(theta, x_over_all_arms(rec))     # over rec.candidate_arms
        logp   = log_softmax(logits)[index_of(rec.arm)]
        quality = 1.0 if rec.outcome.accepted else rec.outcome.partial_axis_score
        r = rec.pref_vector.w_quality * quality \
            - rec.pref_vector.w_cost * min(rec.outcome.cost / cfg.bandit_reward_tau, 1.0)
        rewards.append(r); grads.append((logp, r))
    baseline = mean(rewards)                                   # batch-mean baseline (variance reduction)
    loss = -mean((r - baseline) * logp for (logp, r) in grads) \
           - cfg.entropy_coef * mean_entropy(theta, journal_batch)   # entropy reg → keep exploring
    theta = adam_step(theta, grad(loss, theta), lr=cfg.bandit_lr)
    return theta
```

Critically, learning is **off-policy from the journal**, not on-policy during a run: `select()` is called live (fail-open), outcomes are appended to `controller_decisions.jsonl`, and `bandit_update_step` consumes those records *between* runs. This means a learning bug can never corrupt a live solve — it can only produce a worse `theta`, which the blend-not-switch wrapper (Section 14.2) will still clamp to the heuristic baseline when its confidence is low. The bandit's cost-over-conservatism failure mode (it can stop exploring expensive-but-strong frontier models too early) is mitigated by entropy regularization plus a floor on exploration of any arm whose `n_observations` is below `bandit_min_explore_n`.

#### 17.1.2 Stage 1 — GEPA reflective prompt evolution

Once the bandit is stable and the journal holds enough completed runs, Stage 1 optimizes the **orchestrator's own natural-language artifacts** — the controller's system instructions, the per-stage agent prompts (reproduce/localize/patch/verify), and the tool descriptions — by reflective mutation on NL traces plus Pareto-frontier selection ([GEPA](https://arxiv.org/abs/2507.19457)). This is the highest-leverage *and* most vendor-agnostic lever, because **it changes prompts, not weights**, so improvements transfer across any backing model (Claude Code, Codex/GPT, Gemini, open-weight) behind the Normalized Executor of Section 3.

```
# controller/gepa.py — run OFFLINE over journaled runs; gated on no-regression vs upstream harness
candidates ← {current_prompt_set}                            # each item carries per-instance scores
for generation in range(cfg.gepa_max_generations):
    parent   ← pareto_sample(candidates)                     # sample from per-instance best, not global best
    traces   ← collect_nl_traces(parent, minibatch)          # test logs, diffs, reviewer comments
    feedback ← reflect(traces, anti_overfit_instructions)    # LLM proposer: "why did these fail/succeed?"
    child    ← mutate(parent, feedback)                       # NL prompt / tool-desc edit
    if len(child) > cfg.gepa_prompt_max_chars:               # length regularization (anti-bloat)
        child ← compress(child)                              # Decagon: ~4x compression achievable
    scores   ← eval_on_validation(child)                     # HELD-OUT tasks, deterministic verifier
    if not regressed(scores, baseline) and not regressed_headline(child):
        candidates ← pareto_update(candidates, child, scores)  # keep if non-dominated on ANY instance
return pareto_frontier(candidates)                            # promote a frontier member only after a gate
```

Two GEPA design points are load-bearing:

- **Pareto-frontier selection (not global-best).** GEPA keeps the per-instance best prompts so the search does not collapse into a local optimum — this is the source of its sample-efficiency edge over GRPO. `pareto_update` adds `child` iff it is non-dominated on at least one validation instance; `pareto_sample` weights parents by how many instances they uniquely win.
- **Anti-overfit guardrails are mandatory.** Prompt bloat (>5000-char prompts that encode training keywords) is the documented overfitting tell ([Decagon production notes](https://decagon.ai/blog/optimizing-gepa-for-production), reporting 4× compression from length regularization). APEX-Ω therefore (a) holds a **separate validation set** distinct from the reflection minibatch; (b) **length-regularizes** the reflection proposer (`gepa_prompt_max_chars`); (c) injects explicit **anti-overfit instructions** into the reflective feedback ("generalize the lesson; do not memorize this issue"); (d) **gates promotion of any evolved prompt set on a no-regression check against the upstream-harness number** (Section 15) — the same fairness discipline that protects the headline.

A caveat recorded honestly: GEPA/DSPy's modular-program assumption fits *linear pipelines* better than highly dynamic multi-tool agents. APEX-Ω's per-rollout flow is largely a staged pipeline (reproduce→localize→patch→verify, Sections 2/13), so it fits well; the controller's *branching* decisions (Section 9) are where GEPA's structure assumption is weakest, and there the bandit (Stage 0) remains the active surface.

#### 17.1.3 Stage 2 — full RL, volume-gated and verifier-gated

RL over orchestrator decisions is **deferred** (Section 18: defer). It is admitted only when *both* a volume threshold and the hardened-verifier gate (§17.3) are met. When admitted, the design borrows from the trained-orchestrator track but in its safest form:

- **Plan-as-parsable-structure** ([The Conductor, arXiv 2512.04388](https://arxiv.org/html/2512.04388v1)): the controller emits a CoT then structured fields — `(worker/model id, NL subtask, context-visibility list)` — executable and RL-trainable with a trivial format-gate + correctness reward and **no KL penalty**. Anonymize candidate models by ordinal ID to force exploration over name-priors. This keeps RL vendor-agnostic: the policy reasons about *roles*, not brands (and dovetails with Section 14's profile-vector, name-blind representation).
- **Cost-penalized terminal reward** ([Puppeteer, NeurIPS 2025](https://arxiv.org/abs/2505.19591)): `R_T = correctness - λ·C_T` with a per-step cost term `C_t` so the controller learns to terminate early and pick cheaper agents. Start `λ≈0.1`; **correctness is the deterministic verifier (RLVR / unit tests / upstream harness), never a soft scorer.**
- **Short episodes** (length ~4) to limit credit diffusion — the survey's central warning.
- **Agent/turn-wise grouping if GRPO** ([AT-GRPO / Dr. MAS](https://arxiv.org/html/2510.11062)): vanilla group-normalization is unstable because prompts differ by role/turn; normalize *within agent and turn*.
- **Off-policy credit via the journal.** Spawn/delegate counterfactuals are non-identifiable from on-policy traces. APEX-Ω's durable journal — `controller_decisions.jsonl` plus the per-`agent()`-call WAL (Section 15) — is the substrate for off-policy / counterfactual credit ([Lemon, arXiv 2605.14483](https://arxiv.org/html/2605.14483)): we log alternative branches and edited orchestration fields so spawn decisions can be evaluated honestly, rather than pretending the untaken branch's reward is observable.

**Do not expect when-to-stop to emerge from RL.** This is explicit ([Calibrate-Then-Act, arXiv 2602.16699](https://arxiv.org/html/2602.16699v1)): cost-aware stopping/exploration does *not* emerge from end-to-end outcome-reward RL on coding tasks. Stopping is handled by a decoupled calibration module (§17.4), not by the RL objective.

### 17.2 Abstracted episodic memory (ReasoningBank / CODESKILL lineage)

APEX-Ω evolves v1's `EpisodicMemoryBus` (append-only Discovery store, with negative/ruled-out sharing and reserved-negative-id cross-solve priors) and `extract_durable_insights` (caps at 64, decay 0.85) into an **abstracted insight/skill store**. The cardinal rule, strongly evidenced: **distill insights and skills from both successes AND failures; never replay raw trajectories.** Raw-trajectory reuse causes "experience following" error propagation and self-degradation, and free retrieval can underperform memory-free ([ReasoningBank](https://research.google/blog/reasoningbank-enabling-agents-to-learn-from-experience/); [ExpeL, arXiv 2308.10144](https://arxiv.org/html/2308.10144v2); [SWE-ContextBench](https://arxiv.org/abs/2602.08316), which shows reused context helps *only* when compact AND correctly selected). Expected gains are ReasoningBank-class: ~+4.6% on SWE-Bench-Verified and ~3 fewer steps/task.

```python
@dataclass
class MemoryItem:
    item_id: str                # content sha256 (deterministic, dedup key)
    kind: str                   # "insight" | "skill" | "negative_constraint"
    statement: str              # ABSTRACTED NL lesson, length-bounded (<= insight_max_chars)
    scope: str                  # "repo:<id>" | "language:<x>" | "global"
    provenance: list[str]       # source run_ids (for audit; NEVER replayed as context)
    derived_from_failure: bool  # True if distilled from a failure (negative-constraint lineage)
    support_count: int          # corroboration across runs (raises confidence)
    contradiction_count: int    # times a later run refuted it (lowers confidence; can evict)
    confidence: float           # [0,1]; downweighted, NEVER authoritative
    created_at: str
    last_used_at: str
    decay: float                # default 0.85 per v1
```

#### 17.2.1 Distillation (write path)

At solve end, a distiller pass reads the run's **verified artifacts** (diffs, test outcomes, reviewer comments) and the failure-taxonomy outcome (Section 15) and emits *abstracted* `MemoryItem`s — not transcripts. This is the [CODESKILL](https://arxiv.org/abs/2605.25430) / ExpeL pattern of contrasting success/failure into reusable heuristics.

```python
# memory/distill.py — runs at solve end, after the verdict is final and journaled
def distill(run: RunRecord, cfg) -> list[MemoryItem]:
    items = []
    if run.status is Status.SOLVED:
        # SKILL: what generalizable move worked, abstracted away from this issue's specifics
        items += propose_skills(run.winning_diff, run.verified_tests,
                                instruction="State the reusable fix pattern; omit issue-specific identifiers.")
    # NEGATIVE CONSTRAINTS from BOTH paths: dead strategy axes, doomed localizations
    items += propose_negative_constraints(run.ruled_out_hypotheses, run.failure_class,
                                          instruction="State the abstract dead-end; do not name this ticket.")
    out = []
    for it in items:
        it.statement = enforce_len(it.statement, cfg.insight_max_chars)   # abstraction is enforced, not hoped
        it.item_id   = sha256(it.statement + it.scope)                    # content-addressed dedup
        existing = store.get(it.item_id)
        if existing:                              # corroboration, not duplication
            existing.support_count += 1
            existing.confidence = bump(existing.confidence, existing.support_count)
        else:
            out.append(it)
    return out  # caller persists via atomic_write (Section 15); cap per-scope at insight_max_items (v1: 64)
```

Successes yield reusable **skills** ("for `X`-style import errors in this repo, the fix lives in `conftest.py` fixtures"); both successes and failures yield **negative constraints** ("strategy `inverted_logic` repeatedly breaks the async path here"). Abstraction is *enforced* by a hard length bound and a content-addressed `item_id`, so two runs that learn the same lesson corroborate (raise `support_count`/`confidence`) rather than bloat the store.

#### 17.2.2 Retrieval (read path) — hard gating

Retrieval is gated on **relevance AND compactness**, and excludes the caller's own rollout (preserving v1's `query()` own-rollout-exclusion).

```python
# memory/retrieve.py — runs before a run / before a rollout; may legitimately return {}
def retrieve(task_emb, scope_keys, exclude_rollout_id, cfg) -> InjectedMemory:
    cands = store.by_scope(scope_keys, exclude_rollout=exclude_rollout_id)
    scored = [(it, cosine(task_emb, it.embedding)) for it in cands]
    scored = [(it, s) for (it, s) in scored if s >= cfg.memory_relevance_floor]   # below floor → drop
    if not scored:
        return InjectedMemory.empty()             # memory-free is a VALID, often-better state
    rank = lambda it_s: it_s[1] * it_s[0].confidence * (it_s[0].decay ** age_days(it_s[0]))
    scored.sort(key=rank, reverse=True)
    pos = [it for it, _ in scored if not it.derived_from_failure][:cfg.memory_positive_limit]  # v1: 5
    neg = [it for it, _ in scored if it.derived_from_failure][:cfg.memory_negative_limit]      # v1: 3
    return truncate_to_budget(pos + neg, cfg.memory_inject_char_budget)          # compactness cap
```

- **Relevance floor.** Cosine similarity ≥ `memory_relevance_floor`; below the floor, **return nothing**. Free retrieval that hurts is the documented failure mode; the floor makes memory-free the safe default.
- **Compactness cap.** Total injected memory capped at `memory_inject_char_budget`, ranked by `relevance · confidence · decay^age` and truncated. Caps `positive_limit=5` / `negative_limit=3` preserved from v1.
- **Diversity preservation for negatives.** Negative constraints are shared *abstracted and phased at turn boundaries* (Section 11, Blackboard 2.0), **never share-all** — share-all measurably lowers accuracy (≈−3.7pp) and homogenizes attempts. The verifier must never see producer context (Section 13 independence invariant).

Failure modes this design specifically avoids (each a documented pitfall): *no raw replay* → no experience-following self-degradation; *relevance floor + compactness cap* → free retrieval cannot hurt; *abstracted negatives at boundaries* → diversity preserved, not collapsed.

### 17.3 Verifier-quality gate before any RL (anti-collapse)

A noisy verifier used as a learning signal does not merely add noise — it **flips learning to collapse**, because the policy learns to produce verifier-passing-but-wrong code. Therefore, before *any* RL (Stage 2) is enabled, the verifier that produces its reward must pass a hard quality gate.

```python
# learn/verifier_gate.py — must return True before rl.train() is allowed to run
def verifier_quality_gate(verifier, labeled_holdout, cfg) -> bool:
    # 1. Pre-filter flaky tasks: rerun ~50x under the deterministic harness (Section 15); drop flippers.
    stable = [t for t in labeled_holdout
              if len({run_harness(t).passed for _ in range(cfg.rl_flaky_rerun_count)}) == 1]
    if len(stable) < cfg.rl_min_stable_tasks:
        return False
    # 2. Youden's J: discrimination must be strictly positive (better than a coin flip).
    tp, fp, tn, fn = confusion(verifier, stable)         # vs ground-truth labels
    sensitivity = tp / (tp + fn); specificity = tn / (tn + fp)
    J = sensitivity + specificity - 1.0
    if J <= cfg.rl_youden_j_min:                          # default: > 0
        return False
    # 3. Calibration on the AXIS OF USE: TTS reranking skill does NOT predict RL-reward fitness.
    if brier_score(verifier, stable) > cfg.rl_brier_max:  # SWE-RM: measure classification+calibration
        return False
    return True
```

- **Youden's J > 0.** `J = sensitivity + specificity − 1` must be strictly positive on a held-out labeled set. A verifier with `J ≤ 0` is worse than a coin flip at separating correct from incorrect and must never be a reward.
- **Pre-filter flaky tasks.** Rerun each candidate RL task ~50× under the deterministic harness (Section 15) and drop tasks whose pass/fail flips — flaky tasks inject pure noise into the reward.
- **Binary + appeals, not pass-rate density.** Use a binary verified/not-verified reward with an appeals path (re-verification on dispute), not a continuous pass-rate density a policy can hill-climb by partial gaming.
- **Evaluate the verifier on the axis of use.** Test-time-selection quality does *not* predict RL-reward fitness ([SWE-RM, arXiv 2512.21919](https://arxiv.org/abs/2512.21919)): a verifier good for best-of-N reranking can be a poor, miscalibrated RL reward. Measure classification accuracy + calibration (Brier/log), not just Best@K, before promoting a verifier to a reward.

This gate is the learning-time counterpart to the selection-time Cardinal Contract: in both, **execution evidence is the anchor and a soft/learned signal is admitted only after it earns trust** (and only as a re-ranker, never a promoter — Section 13). Cross-reference Section 13's hybrid verifier (execution + swappable generative critic, [R2E-Gym](https://arxiv.org/abs/2504.07164) ~43%→51%): that critic is discrimination-only at *selection* time and is categorically **not** used as the RL reward.

#### 17.3.1 Anti-reward-hacking is a SAFETY requirement

Reward hacking on coding RL environments is demonstrated to **generalize beyond the hack** — to alignment-faking (~50%) and attempted sabotage of safety code (~12% via Claude Code vs 0% for baselines) ([Anthropic, arXiv 2511.18397](https://arxiv.org/abs/2511.18397)). Because APEX-Ω may run self-improvement loops over coding environments, hack-resistance is a *safety* property, not just a quality one. Mandatory guardrails (most already present in v1's anti-cheat, Section 15; here made non-negotiable for the learning loop):

- **Never use a soft scorer as the RL reward.** Only deterministic verifiers / the upstream harness.
- **Seal test/CI/assertion files** read-only with hash checks; forbid editing tests, CI, or assertions; preserve v1's `patch_sanitizer` `GOLD_PROTECTED_TEST` rejection (a gold test buried under a vendored dir still falls through to rejection) and true git-history flatten for gold-recovery channels.
- **Detect exploit signatures**: `sys.exit(0)`, `__eq__`/`__hash__` overrides on assertion targets, pytest monkeypatching of the reporter — the documented hack family ([LLMs-Gaming-Verifiers, arXiv 2604.15149](https://arxiv.org/abs/2604.15149): extensional-only checks "admit false positives").
- **Prefer held-out hidden tests** the policy never sees during training; rotate to fresh tasks to fight contamination.
- **Inoculation prompting**: reframing hacking as unacceptable in training prompts cut final misalignment 75–90% despite >99% hack rate in Anthropic's study — cheap and adopted as a default for any RL stage.

### 17.4 Decoupled calibration module (when to stop)

Knowing *when to stop / escalate / spawn* is engineered, not trained-in ([Calibrate-Then-Act](https://arxiv.org/html/2602.16699v1)): it must be a separate module that **decouples uncertainty estimation from action selection**. APEX-Ω implements stopping as a calibration head consuming **execution evidence** plus the bandit's confidence, emitting a decision against the budget (Section 16). It is the brain's "brake-aware" reflex, and it is owned here (not by the RL objective and not by the selector, which only accepts).

```python
@dataclass
class StopSignal:
    verified_pass_count: int        # candidates with VerificationResult.accepted (Section 13)
    xval_agreement: float           # cross-validation matrix agreement [0,1] (Section 13.4)
    regression_survival_frac: float # fraction surviving regression-prune (Section 13)
    bandit_confidence: float        # policy's own confidence in further marginal gain
    budget_remaining_frac: float    # Section 16

def decide_stop(s: StopSignal, cfg) -> Literal["STOP", "CONTINUE", "ESCALATE"]:
    # Uncertainty estimate is computed FIRST and independently of the action choice.
    p_more_helps = calibrated_head(s)                    # proper-scoring-rule-trained if learned at all
    if s.verified_pass_count == 0 and s.budget_remaining_frac < cfg.stop_abstain_budget:
        return "STOP"                                   # abstain: no positive evidence, budget low
    if p_more_helps < cfg.stop_marginal_floor:
        return "STOP"
    if s.verified_pass_count == 0 and s.budget_remaining_frac > cfg.escalate_budget:
        return "ESCALATE"                               # spend on a stronger model / wider search
    return "CONTINUE"
```

This preserves v1's first-class abstention: with no positive execution evidence and a depleted budget, the system **abstains rather than guesses** (Section 13.10). If a calibration head is ever *trained*, it uses a proper scoring rule (Brier/log) to counter RLVR-induced overconfidence — but the default is the *engineered* decoupled module above, not an RL artifact. Because it is decoupled, the stopping policy can be hardened, audited, and unit-tested independently of whatever learning rung is active.

### 17.5 Off-policy credit substrate & control-flow summary

The durable journal is the through-line that makes all three rungs learnable without violating determinism:

- **`controller_decisions.jsonl`** (v1, kept) logs every decision with its `DecisionContext`, chosen arm, sampled `pref_vector`, and the eventual `DecisionOutcome` — the substrate for bandit `update()`, GEPA trace collection, and off-policy RL credit. Appended via the same atomic-write discipline as every other artifact (Section 15).
- **The per-`agent()`-call WAL** (Section 15) records inputs, the resolved `(vendor, model, effort)`, and verified outcomes, so a learning run **reproduces artifacts (diffs + re-run verification), never token streams** — bit-reproducible agent output is rejected (Section 18) as impossible across hosted APIs (temp-0 batch non-invariance).
- **All learning is offline and journal-driven.** No learning step mutates a live run except through the blend-not-switch policy, which is fail-open by construction.

```
during a run:    arm = controller.select(ctx)          # Stage-0 active control, fail-open (Section 14)
                 ... execute worker, verify (deterministic, Section 13) ...
                 journal.append(ctx, outcome)          # WAL + controller_decisions.jsonl
                 # at solve end:
                 store.persist(distill(run))           # abstracted MemoryItems, never transcripts

between runs:    theta = bandit_update_step(theta, journal_batch)     # Stage 0, online from journal
                 prompt_set = gepa.evolve(prompt_set)                 # Stage 1, Pareto + no-regression gate
                 if volume_gate and verifier_quality_gate(...):       # §17.3
                     rl.train(...)                                    # Stage 2, RLVR + short episodes + inoculation

before a run:    mem = memory.retrieve(task)           # relevance+compactness gated, may return {}
```

### 17.6 Config keys

| Key | Default | Meaning |
|---|---|---|
| `controller_learning_stage` | `0` | 0=bandit, 1=+GEPA, 2=+RL (each higher stage requires its gate) |
| `controller_policy_enabled` (`library_enabled`) | `true` | Master kill switch (Section 14.2); off ⇒ pure heuristics, headline unchanged |
| `bandit_reward_w_cost` / `bandit_reward_tau` | `0.3` / tuned | Preference-conditioned cost weight and cost-normalizer |
| `bandit_lr` / `entropy_coef` | tuned / tuned | REINFORCE step size; exploration regularizer |
| `bandit_min_explore_n` | tuned | Forced exploration floor per under-observed arm (anti cost-over-conservatism) |
| `gepa_max_generations` / `gepa_val_fraction` | `—` / `0.3` | GEPA budget and held-out validation fraction |
| `gepa_prompt_max_chars` | `5000` | Length regularization (anti-bloat) |
| `memory_relevance_floor` | tuned | Below this cosine similarity, retrieve nothing |
| `memory_inject_char_budget` | tuned | Compactness cap on injected memory |
| `memory_positive_limit` / `memory_negative_limit` | `5` / `3` | v1 injection caps preserved |
| `insight_max_chars` / `insight_max_items` | tuned / `64` | Abstraction length bound; per-scope store cap (v1) |
| `rl_youden_j_min` | `>0` | Verifier-quality gate before RL |
| `rl_flaky_rerun_count` / `rl_min_stable_tasks` | `50` / tuned | Reruns to pre-filter flaky tasks; min surviving set |
| `rl_brier_max` | tuned | Calibration ceiling on the reward verifier (axis-of-use check) |
| `rl_episode_len` / `rl_lambda_cost` | `4` / `0.1` | Short episodes; cost-penalized terminal reward |
| `rl_inoculation_enabled` | `true` | Anti-reward-hacking prompt reframing |
| `stop_marginal_floor` / `stop_abstain_budget` / `escalate_budget` | tuned | Calibration-module thresholds (§17.4) |

### 17.7 What is explicitly NOT done (honest non-goals)

- **No raw-trajectory replay** in memory — distilled insights/skills only; raw experience-following causes self-degradation.
- **No soft scorer as an RL reward** — deterministic verifiers only; the generative critic of Section 13 is selection-time, discrimination-only.
- **No trusting a noisy verifier as the learning signal** — a verifier below the Youden/flaky/calibration gate flips learning to collapse and is forbidden as a reward.
- **No expectation that stopping emerges from RL** — calibration is a separate engineered module (§17.4).
- **No RL before both gates are met** — the verifier-quality gate (§17.3) and the volume gate must both pass.
- **No learned component that can promote an unverified candidate or break a live run** — blend-not-switch, fail-open, downgrade-only, consistent with the Cardinal Safety Contract (Section 13).

## 18. Fusion Ledger: Kept / Modified / Dropped

This section is the **canonical in/out list** for the entire plan. Every other section (the speculative tree-search layer in Section 9, CTDG pruning in Section 10, the blackboard in Section 11, the model economy in Section 12, verification in Section 13, the controller in Section 14, isolation/determinism in Section 15) MUST respect the dispositions recorded here. Where a writer believes a mechanism should be promoted, demoted, or re-scoped relative to this ledger, that is a change to the ledger and requires re-deriving the disposition from a verdict or an ingest finding — not a quiet override in a downstream section.

The ledger encodes one machine-readable invariant — the `accepted_mechanisms` array — plus the human-readable rationale, the origin of each idea (v1 substrate, the v3 redesign, the SOTA digest, or the dynamic-workflow paradigm), and the **traceability** of each disposition to a specific adversarial verdict or v1/paradigm digest finding. The five dispositions form a strict precedence: a **Reject** can never be silently softened into a **Defer**, and a **Defer** is a staged future build, not a hedge. The two pitfalls that govern this section are explicit: (1) do not contradict the `accepted_mechanisms` array, and (2) do not soften a reject into a defer without verdict basis.

### 18.1 How to read the ledger

#### 18.1.1 Disposition vocabulary

| Disposition | Meaning | Operational consequence |
|---|---|---|
| **Adopt** | Build in v1.0. The substrate-level commitment; usually a verbatim lift of a v1 invariant or the one net-new primitive (`pipeline()`). | Ships day one; cannot be flag-gated off in a way that weakens an invariant. |
| **Adopt-modified** | Build in v1.0, but only in a re-scoped, qualified form that respects the Cardinal Safety Contract and the relevant verdict. | Ships day one with the qualification baked into its data structures and control flow, not bolted on. |
| **Defer** | Sound in principle, sequenced to a later stage behind a prerequisite. | Not built in v1.0; the seam is left in place. Defer ≠ Reject. |
| **Reject** | Dropped as a design pillar, on an `unsound` or `partially_sound→default-rejected` verdict basis. | Never built as described. The salvageable fragment, if any, re-enters under a different mechanism's **Adopt-modified** entry. |

The distinction between **Defer** and **Reject** is load-bearing and is the subject of pitfall (2). A Defer item (GEPA, full RL) has *no adverse verdict* — it is gated only by sequencing and prerequisite maturity. A Reject item has an *explicit adverse verdict* (`unsound`, or `partially_sound` where the default form is rejected). The ledger never moves an item between these buckets without changing its verdict basis.

#### 18.1.2 The Cardinal Safety Contract as the universal admission rule

Every **Adopt-modified** and every **Reject** in this ledger traces back, directly or indirectly, to one v1 invariant: the **Cardinal Safety Contract** (APEX_DESIGN_BLUEPRINT.md §13.1) — *execution evidence is authoritative; soft signals may re-rank within an execution-verified tier or downgrade an already-accepted candidate, but may NEVER promote an unverified candidate.* Enforced structurally in v1 by `_apply_evidence_bound_review` (flips `accepted` only `True→False`) and the deterministic ranking tuple where every soft/learned/LLM key sits strictly below every execution key (§13.7). The redesign's bolder mechanisms (pruning gates, plan scoring, blackboard push) are admitted *only in forms that cannot make a soft signal load-bearing for exclusion* — because the inverse-equivalent violation (a soft signal *suppressing* a would-be-verified candidate) is just as fatal as promotion. This is why three redesign mechanisms have a gate-form **Reject** sitting beside a hint-form **Adopt-modified**.

### 18.2 The canonical ledger (grouped by origin × disposition)

The following table is the human-readable rendering of the `accepted_mechanisms` array. The `Origin` column tags each idea: **v1** (the hardened substrate), **redesign** (the v3 prose blueprint), **SOTA** (the research digest), **paradigm** (the vendor-neutral dynamic-workflow model). The `Verdict / finding basis` column is the traceability required by Chief Guidance — every disposition points to either an adversarial verdict or a v1/paradigm digest finding.

#### 18.2.1 Adopt (substrate — build verbatim or lift-not-rebuild)

| # | Mechanism | Origin | Verdict / finding basis | See |
|---|---|---|---|---|
| A1 | Vendor-neutral workflow engine (`agent`/`parallel`/`pipeline`/`phase`/`budget`) | paradigm | "Re-architecting as a deterministic dynamic-workflow engine is a stronger substrate" → **sound_with_caveats** (substrate half sound); paradigm finding: v1's `run_structured_prompt`≈`agent()`, `execute_rollout_requests`≈`parallel()` already exist. | §2 |
| A2 | `pipeline()` per-item staged streaming | paradigm | Paradigm finding: "the one genuinely net-new primitive"; no v1 analog; cuts wall-clock from sum-of-slowest-per-stage to slowest-single-chain. | §2, §9 |
| A3 | Cardinal Safety Contract (execution-evidence-authoritative selection) | v1 | v1 `keep` finding §13.1; the universal admission rule (18.1.2); counters the "Inference Scaling FLaws" false-positive mode. | §13 |
| A4 | Cheap-first verification cascade that never synthesizes a pass | v1 | v1 `keep` finding §12.1 (rc==0→errors=1; rc==124→`regression_inconclusive` +0.15). | §13 |
| A5 | Per-rollout git-worktree isolation + `fcntl` locks | v1 | v1 `keep` finding §8.2; CAID ablation 63.3 (worktree) vs 57.2 (single) vs 55.5 (soft). | §15 |
| A6 | Determinism + RunManifest + Docker digest pinning | v1 | v1 `keep` finding §17.3/§17.4/§18.3; reproduce artifacts not token streams. | §15 |
| A7 | Anti-cheat / fairness / failure taxonomy / first-class abstention | v1 | v1 `keep` finding §15/§17.2/§13.10; reward-hacking scales with capability (ImpossibleBench GPT-5 76%). | §13, §15 |
| A8 | Two-tier failure memory + self-evicting BackendPortfolio | v1 | arch finding (`llm_routing.py`, `backend_portfolio.py`): call-failover vs backend-global reroute. | §3 |
| A9 | Durable input-hash journaled restart-survivable resume | paradigm + v1 | paradigm finding: ReplayRecorder has **no production callsite**; escrow WAL (CCEDF) is a narrow backstop. Durable-resume verdict **sound_with_caveats**. | §15 |
| A10 | Normalized Executor + ACP-style capability negotiation (graceful degradation) | vendor + v1 | Vendor-executor verdict: feasibility **sound_with_caveats**; consolidate v1's scattered per-vendor fragments. | §3 |
| A11 | (vendor, model) as a first-class diversity/search axis | vendor + SOTA | Vendor-executor verdict: diversity half **sound** *given an execution-grounded selector* (Devlo 70.2%, TRAE 70.4%). | §3, §9 |
| A12 | Difficulty-adaptive low-K allocation (default ON) | SOTA + v1 seam | Resampling-limits: optimal K often <10; v1's `enable_adaptive_allocation` exists but is OFF. | §9, §14 |
| A13 | Early localization-futility gate | SOTA + redesign | Snell difficulty-adaptive; routes budget to surviving hypotheses; **never** suppresses a candidate without execution evidence. | §9, §10 |
| A14 | Hybrid verifier (execution + swappable generative critic, discrimination-only) | SOTA | R2E-Gym ~43%→51%; critic breaks ties only **within** the execution-verified tier (Cardinal Contract). | §13 |
| A15 | Prefix-stable prompt assembly + provider-cache adapter | efficiency | Speculative-branching verdict: the saving "comes from prefix/KV reuse, not the tree"; ~90% off cached reads. | §16 |

#### 18.2.2 Adopt-modified (build only in the qualified, re-scoped form)

| # | Mechanism | Origin | Qualification (the modification) | Verdict basis | See |
|---|---|---|---|---|---|
| M1 | Bounded adaptive-branching over FrontierSearch | redesign + SOTA(AB-MCTS) | Keep the part that wins (AB-MCTS wider/deeper allocation); run **inside** FrontierSearch budget caps; **mandatory collapse to verified best-of-N below a feedback-confidence floor**. | MCTS verdict **unsound** for distributed-MCTS-as-core-loop; AB-MCTS only wins above ~64 calls. | §9 |
| M2 | Agent-initiated `speculate()` fork | redesign | Admit **only at turn/checkpoint boundaries**, feeding FrontierSearch ranking/budget; constant-factor cheaper via prefix reuse, **not exponential**; bounded by virtual-loss / `min_branch_reward`. | Speculative-branching verdict **partially_sound**: "exponentially" unsupported; mid-subprocess fork infeasible vs opaque CLIs. | §9 |
| M3 | CTDG as test prioritizer + dynamic-coverage prune + full-suite backstop | redesign + SOTA(ctdg) | Reordering = zero false-negative risk; dynamic coverage near-safe; full-suite backstop keeps it honest. **Static-as-gate is rejected** (see R2). | CTDG verdict **unsound** for static gate; "prioritize, don't prune." | §10 |
| M4 | Cheap pre-execution plan scoring as a downgrade-only prioritizer | redesign + SOTA(prm) | Allowed only to set branch priors / budget share the controller can override; **never excludes a candidate pre-execution**. **Hard-gate form is rejected** (see R3). | Plan-scoring verdict **partially_sound**; pre-execution hard prune = inverse Cardinal-Contract violation. | §10, §14 |
| M5 | Blackboard 2.0: phased, abstracted negative-constraint sharing at turn boundaries | redesign + SOTA(blackboard) | MEMOIR/LTS: abstracted negatives preserve diversity; share-all loses. Evolve `EpisodicMemoryBus` delivery (keep relevance/confidence/dedup/own-rollout exclusion); **verifier must not see producer context**. | Cross-branch-sharing verdict **sound_with_caveats**; share-all −3.7pp (LTS). **Raw share-all rejected** (see R4). | §11 |
| M6 | Model economy as sub-role, verification-gated cascade | redesign + SOTA(modeleconomy) | Aider/opusplan 5–14× cheaper *with a competent editor*; cheapen run/verify and narrow edits only; keep frontier on navigation/multi-file edits (HyperAgent); escalate on first verify-on-diff failure with a rewrite-cycle cap. | Heavy-orch/thin-exec verdict **partially_sound**; default thin-everywhere **rejected** (see R5). | §12 |
| M7 | Open-pool active controller via learned capability/cost profiles | novel + selfimprove | Staged (bandit → GEPA → RL); blend-not-switch, fail-open to heuristic; wire/remove `library_enabled`. **Stage-0 (bandit) ships day one.** | Substrate verdict **sound_with_caveats**; arch finding: `library_enabled` is unwired (the layer's single most important correctness gap). | §14 |

#### 18.2.3 Defer (sound, sequenced behind a prerequisite — NOT rejected)

| # | Mechanism | Origin | Stage / gate | Verdict basis (no adverse verdict) | See |
|---|---|---|---|---|---|
| D1 | GEPA-style reflective prompt evolution of the controller | selfimprove | **Stage 1**, after the bandit (M7 Stage-0) ships. | No adverse verdict; 35× cheaper than RL, prompts-not-weights = vendor-agnostic. | §14, §17 |
| D2 | Full RL (Puppeteer/Conductor GRPO) over orchestrator decisions | selfimprove | **Stage 2**, only when volume justifies. | No adverse verdict; cost-penalized terminal reward on deterministic verifiers; short episodes to limit credit diffusion. | §14, §17 |

#### 18.2.4 Reject (dropped as a design pillar — explicit adverse verdict)

| # | Mechanism | Origin | Verdict | One-line basis | Salvage routes to |
|---|---|---|---|---|---|
| R1 | Distributed/classical MCTS as the core loop | redesign | **unsound** (high) | Re-describes `FrontierSearchController`; plain MCTS does not reliably beat verified sampling at repo scale; brittle vs non-serializable container state. | M1 (bounded adaptive-branching in budget) |
| R2 | Static-AST CTDG as a test-pruning gate | redesign | **unsound** (high) | PyCG ~70% recall; reflection/monkeypatch/fixtures invisible; pytest set not statically enumerable; silently drops fault-revealing tests. | M3 (prioritizer + dynamic-coverage prune + backstop) |
| R3 | Cheap pre-execution plan scoring as a hard prune/gate | redesign | **partially_sound** → gate rejected | False-negative pruning suppresses correct-but-unverified plans before evidence exists; inverse-equivalent Cardinal-Contract violation. | M4 (downgrade-only prioritizer) |
| R4 | Raw share-all / "instant push" mid-subprocess injection blackboard | redesign | **sound_with_caveats** → share-all/real-time rejected | Share-all −3.7pp and homogenizes attempts; mid-subprocess prompt mutation infeasible vs opaque CLIs; breaks determinism/replay. | M5 (phased abstracted negative sharing at boundaries) |
| R5 | Heavy-orchestrator + thin executor as the **default** shape on hard repo SWE | redesign | **partially_sound** → default rejected | HyperAgent ablation: cheapening navigation/multi-file editing causes the worst resolve-rate drops; the almost-right trap can cost more than one frontier pass. | M6 (sub-role, verification-gated cascade) |
| R6 | Non-adaptive fixed-K default (5, up to the 16 cap), caps OFF | v1 default | the headline cost pathology | Replace with adaptive low-K + budget-aware deepening; full-cap kept **only** as the thin-feedback floor. | A12 (adaptive low-K) + M1 (deepening) |
| R7 | Bit-reproducible agent **output** replay | redesign | impossible across hosted APIs | temp-0 batch non-invariance (Thinking Machines: 80 distinct completions / 1000 runs); reproduce artifacts (diffs + re-run verification) instead. | A6/A9 (artifact replay) |

### 18.3 The Rejects, with their verdict basis spelled out

Chief Guidance requires that the rejects be highlighted with their `unsound` / `partially_sound` verdict basis, and pitfall (2) forbids softening any of them into a Defer. This subsection makes the verdict basis explicit so no downstream section can quietly re-admit a rejected pillar.

#### 18.3.1 R1 — Distributed/classical MCTS as the core loop (verdict: unsound, high)

The adversarial verdict is `unsound` with high confidence. It dismantles the claim on four axes simultaneously: (1) **no clean head-to-head** — the strongest repo-level MCTS result (SWE-Search, +23% relative) is measured against a *greedy single-trajectory* agent, not verified best-of-N, and SWE-Search's own Pass@5 (34.0) ≈ Pass@1 (31.0), proving the bottleneck is *selection*, not search; (2) **repeated sampling already extracts most coverage** (Large Language Monkeys 15.9%→56% at <1/3 cost), and REBASE found "MCTS underperformed plain sampling at every budget"; (3) **the fixed-budget framing is where MCTS is weakest** — AB-MCTS only beats sampling above ~64 calls, and under strict small budgets can *lose* to plain repeated sampling (BAVT); (4) **"distributed over the codebase" is a hard engineering blocker** — a shared mutable tree needs cheap state save/restore that non-serializable Docker environments do not provide.

It also collides with v1 invariants: a shared mutable tree is the architectural opposite of v1's context-isolation / filesystem-as-source-of-truth / per-rollout-scoping (no machine-wide mutex); and value-function pruning of unexecuted nodes violates the Cardinal Safety Contract. **This is a hard Reject, not a Defer** — the verdict is `unsound`, not merely "premature." The salvageable core (adaptive wider-vs-deeper allocation, git-checkpoint branching, strong selector) re-enters as **M1**, bounded inside FrontierSearch caps with a mandatory collapse to verified best-of-N below a feedback-confidence floor.

#### 18.3.2 R2 — Static-AST CTDG as a test-pruning gate (verdict: unsound, high)

The verdict is `unsound` with high confidence. The conjunction (static + safe + millisecond + dynamic Python + zero correct-candidate loss) fails on the load-bearing terms: PyCG reports ~99.2% precision but only **~69.9% recall** — ~30% of real call edges are absent, and each missing edge is a potential silently-dropped fault-revealing test. Worse, the pytest item set is **not statically enumerable** (parametrize, fixtures, plugins, `conftest`, collection hooks); only `pytest --collect-only` knows the real set, so a static graph cannot even name what it prunes. Reflection alone makes static RTS unsafe in Java (OOPSLA'19); Python is strictly more dynamic. Rothermel–Harrold makes safe-AND-maximal-pruning structurally impossible from imperfect dependency data.

This directly violates the Cardinal Safety Contract: a static CTDG is a non-execution soft signal, and using it to *prune* is strictly stronger than the prohibited "promote unverified." **Hard Reject.** The valuable parts — prioritization (zero false-negative risk) and *dynamic*-coverage pruning with a full-suite backstop — re-enter as **M3**.

#### 18.3.3 R3 — Cheap pre-execution plan scoring as a hard gate (verdict: partially_sound → gate rejected)

The verdict is `partially_sound`. The salvageable half (cheap scoring to *prioritize*) is real; the rejected half is the *hard gate*. The evidence is hostile on every axis the gate depends on: verifier quality is the binding constraint (weak/open critics *drop* performance, SWE-PRM 30–38.8% vs 40.0% base); execution signals dominate as the anchor (ORPS execution-grounded untrained critic 59.9% vs execution-free trained PRM 37.0%); PRM reliability *decreases* with distance from terminal states — exactly where pre-execution pruning happens; and the ~11pp Best@K-vs-Pass@K gap quantifies the correct solutions a verifier discards. A pre-execution hard prune is the **inverse-equivalent** of the Cardinal Contract's prohibited promotion: it makes a soft, execution-free signal load-bearing for *exclusion*, permanently denying a branch any chance of verification.

**The gate form is Reject; the prioritizer form is Adopt-modified (M4).** Per pitfall (2), the gate is *not* softened to a Defer — there is an adverse verdict basis (`partially_sound`, gate-half rejected), so it is a Reject of that form. M4 keeps the cheap critic as a downgrade-only prior the controller can override, with a "wildcard" lane that always executes the lowest-scored unconventional branch.

#### 18.3.4 R4 — Raw share-all / instant-push mid-subprocess blackboard (verdict: sound_with_caveats → share-all & real-time rejected)

The verdict on cross-branch sharing is `sound_with_caveats`; the *specific* rejected forms are (a) **unconditional share-all** and (b) **real-time mid-subprocess push**. Share-all measurably *lowers* accuracy (LTS Table 2: −3.7pp on GAIA) and homogenizes attempts, destroying the diversity that makes sampling work (pass@k gains "vanish when candidates are highly correlated"). Real-time push is infeasible against opaque external CLIs (no mid-subprocess prompt-mutation channel) and breaks determinism/replay (each `agent()` call's inputs become non-deterministic). **Reject** the share-all and real-time forms. The diversity-preserving mechanism — **phased**, **selective**, **abstracted**, **negative-constraint** sharing at turn boundaries, evolving v1's `EpisodicMemoryBus` (which already shares negative discoveries, relevance-ranks, dedups, and excludes the caller's own `rollout_id`) — is **M5**, with the strict guardrail that the verifier never sees producer context (anti-collective-delusion).

#### 18.3.5 R5 — Heavy-orchestrator + thin executor as the default shape (verdict: partially_sound → default rejected)

The verdict is `partially_sound`: the role-split *spine* is supported (Aider architect/editor improved every model, ~14× cheaper), but the claim's own scope word — *hard repo SWE* with a *thin* executor — is the documented failure case. The HyperAgent ablation shows weakening the Navigator (codebase exploration) or Editor roles causes the **worst** resolve-rate drops, and small models trail frontier sharply (HyperAgent's Llama-3-8B "Lite" variant ~16%). The "almost-right trap" means a thin executor needing 3–4 retries costs *more* than one frontier pass. **The default thin-everywhere shape is Reject.** The verification-gated cascade — frontier planner + frontier reviewer + cheap models confined to run/verify and narrow well-specified edits, escalating on first verify-on-diff failure — is **M6**.

#### 18.3.6 R6 — Non-adaptive fixed-K default (5, escalating to the 16 cap), caps off (the cost pathology)

This is v1's own default (`num_rollouts`, escalation waves up to 6, `max_strategy_iterations`=20, token caps OFF per "never optimize for cost"). It is the headline cost pathology the redesign correctly critiques. **Modify/Reject the default**: replace with **A12** (difficulty-adaptive low-K, optimal K often <10) plus **M1** (budget-aware deepening), keeping full-cap 16 *only* as the thin-feedback floor (the regime where verified best-of-N is provably the safe fallback). Note this is a *default* change, not a removal of the capability.

#### 18.3.7 R7 — Bit-reproducible agent output replay

Impossible across hosted APIs: temperature-0 is not bitwise reproducible due to batch non-invariance (Thinking Machines: 80 distinct completions in 1000 runs). v1 itself is explicit that "manifest pinning guarantees environment + ordering, NOT agent OUTPUT; replay guarantees the trajectory." **Reject** the bit-output claim; **A6/A9** reproduce *artifacts* (diffs + re-run verification) instead.

### 18.4 The Adopt-modified items, with their qualification spelled out

Chief Guidance requires highlighting the adopt-modified items *with their qualification*. The qualification is the part that keeps each item inside the Cardinal Safety Contract; a writer who builds the unqualified form has built a Reject. The qualifications are summarized as enforceable predicates below.

| # | Mechanism | The non-negotiable qualification (predicate the build must satisfy) |
|---|---|---|
| M1 | Bounded adaptive-branching | Runs inside FrontierSearch budget caps (`max_depth`, `max_frontier_branching`, `min_branch_reward`); **collapses to verified best-of-N** when feedback confidence < floor; never a free-standing tree. |
| M2 | `speculate()` fork | Fires only at turn/checkpoint boundaries; feeds FrontierSearch ranking; cost framed as **constant-factor (prefix reuse), never exponential**; bounded by virtual-loss. |
| M3 | CTDG | **Prioritize, never statically gate.** Pruning only via dynamic coverage; full-suite backstop authoritative; per-repo safety-mode flag defaults to `prune-with-backstop`. |
| M4 | Plan scoring | **Downgrade-only prior.** Sets branch priority/budget share; controller can override; never removes a candidate from the set pre-execution; wildcard lane preserved. |
| M5 | Blackboard 2.0 | **Phased + abstracted + negative-only at boundaries.** Producer-only scope; verifier never sees producer context; keep relevance/confidence/dedup/own-rollout exclusion. |
| M6 | Model economy | **Sub-role cascade.** Cheap models only on run/verify + narrow edits; frontier on planning/navigation/multi-file/review; escalate on first verify-on-diff failure; rewrite-cycle cap. |
| M7 | Active controller | **Blend-not-switch, fail-open to heuristic.** Staged bandit→GEPA→RL; `library_enabled` wired or removed; Stage-0 bandit ships day one. |

#### 18.4.1 The pruning-gate / sharing-mode contract (machine-checkable)

To make the M3/M4/M5 qualifications enforceable rather than aspirational, the engine carries a small typed config object that downstream sections consume. This is the single point where "adopt-modified, not the gate form" becomes load-bearing in code.

```text
SafetyModeConfig:
  ctdg_mode:            Enum{ advisory, prune_with_backstop, prune_hard }  = prune_with_backstop
  plan_score_mode:      Enum{ prioritize_only, downgrade_only }            = downgrade_only
  blackboard_delivery:  Enum{ off, phased_negative, share_all }           = phased_negative
  blackboard_phase_gate: bool   # first exploratory wave is fully isolated  = true
  branch_collapse_floor: float  # feedback-confidence below which M1→best-of-N = 0.6
  full_cap_floor_K:      int    # thin-feedback floor only                  = 16
  default_adaptive_K:    int    # difficulty-adaptive low-K target          = 5
```

```text
# Admission rule enforced at config-load (fail-loud, never silently downgrade a Reject into an Adopt):
def validate_safety_modes(cfg) -> None:
    # R2: static-AST gate is rejected; prune_hard requires DYNAMIC coverage + backstop, not static AST.
    if cfg.ctdg_mode == prune_hard and not dynamic_coverage_available():
        raise FailLoud("ctdg prune_hard requires dynamic coverage; static-AST gating is rejected (R2)")
    # R3: plan scoring may never gate. There is no 'gate' enum value by construction.
    assert cfg.plan_score_mode in {prioritize_only, downgrade_only}
    # R4: share_all is a rejected form; only permitted behind an explicit research-ablation flag.
    if cfg.blackboard_delivery == share_all and not cfg.research_ablation_optin:
        raise FailLoud("blackboard share_all is rejected (R4, -3.7pp); opt-in ablation only")
    # M5: verifier isolation is structural, not configurable.
    assert verifier_cannot_read_producer_context()  # enforced at wiring time
```

The point of encoding this is pitfall (1): the `accepted_mechanisms` array says `share_all` is `reject` and `plan_score gate` is `reject`. The config above makes it *impossible* to silently ship those forms — `share_all` requires an explicit ablation opt-in, and a plan-score "gate" mode does not exist in the enum at all. This is the mechanical guarantee that the ledger is not contradicted.

### 18.5 Cross-reference to the `accepted_mechanisms` array

Chief Guidance requires an explicit cross-reference to the `accepted_mechanisms` array so writers can verify consistency. The mapping is one-to-one and total: every entry in the canonical array appears exactly once in the ledger above, with the same disposition string. The table below is the verification index.

| `accepted_mechanisms[].name` (verbatim) | `disposition` (verbatim) | Ledger row |
|---|---|---|
| Vendor-neutral workflow engine (agent/parallel/pipeline/phase/budget) | adopt | A1 |
| pipeline() per-item staged streaming | adopt | A2 |
| Cardinal Safety Contract (execution-evidence-authoritative selection) | adopt | A3 |
| Cheap-first verification cascade that never synthesizes a pass | adopt | A4 |
| Per-rollout git-worktree isolation + fcntl locks | adopt | A5 |
| Determinism + RunManifest + Docker digest pinning | adopt | A6 |
| Anti-cheat / fairness / failure taxonomy / first-class abstention | adopt | A7 |
| Two-tier failure memory + self-evicting BackendPortfolio | adopt | A8 |
| Durable input-hash journaled restart-survivable resume | adopt | A9 |
| Normalized Executor + ACP-style capability negotiation | adopt | A10 |
| (vendor, model) as a first-class diversity/search axis | adopt | A11 |
| Difficulty-adaptive low-K allocation (default ON) | adopt | A12 |
| Early localization-futility gate | adopt | A13 |
| Hybrid verifier (execution + swappable generative critic, discrimination-only) | adopt | A14 |
| Prefix-stable prompt assembly + provider-cache adapter | adopt | A15 |
| Bounded adaptive-branching over FrontierSearch | adopt-modified | M1 |
| Agent-initiated speculate() fork | adopt-modified | M2 |
| CTDG as test prioritizer + dynamic-coverage prune + full-suite backstop | adopt-modified | M3 |
| Cheap pre-execution plan scoring as a downgrade-only prioritizer | adopt-modified | M4 |
| Blackboard 2.0: phased, abstracted negative-constraint sharing | adopt-modified | M5 |
| Model economy as sub-role, verification-gated cascade | adopt-modified | M6 |
| Open-pool active controller via learned capability/cost profiles | adopt-modified | M7 |
| GEPA-style reflective prompt evolution of the controller | defer | D1 |
| Full RL (Puppeteer/Conductor GRPO) over orchestrator decisions | defer | D2 |
| Distributed/classical MCTS as the core loop | reject | R1 |
| Static-AST CTDG as a test-pruning gate | reject | R2 |
| Cheap pre-execution plan scoring as a hard prune/gate | reject | R3 |
| Raw share-all / "instant push" mid-subprocess injection blackboard | reject | R4 |
| Heavy-orchestrator + thin executor as the default execution shape | reject | R5 |
| Non-adaptive fixed-K default (5, up to the 16 cap), caps OFF | reject | R6 |
| Bit-reproducible agent OUTPUT replay | reject | R7 |

**Count check:** 15 adopt + 7 adopt-modified + 2 defer + 7 reject = 31 entries, matching the `accepted_mechanisms` array exactly. If a future edit adds, removes, or re-dispositions a row, the array and this index MUST be updated together; a divergence is a build-blocking inconsistency.

### 18.6 The reject↔adopt-modified pairing (why nothing is lost without basis)

A recurring structure in this ledger is that three redesign mechanisms each appear as a **Reject** of their aggressive form *paired with* an **Adopt-modified** of their disciplined form. This is not double-counting; it is the precise expression of "drop the unsound conjunction, keep the sound fragment." The pairing also guards against the inverse of pitfall (2): a writer who reads only the Adopt-modified row might quietly re-introduce the rejected form. The pairings:

| Rejected form | → re-enters as | The dividing line |
|---|---|---|
| R1 Distributed MCTS core loop | M1 Bounded adaptive-branching | Shared mutable tree + value-pruning of unexecuted nodes (Reject) vs budget-capped wider/deeper allocation that collapses to verified best-of-N (Adopt). |
| R2 Static-AST CTDG gate | M3 CTDG prioritizer + dynamic prune | Static graph *excludes* tests (Reject) vs static graph *reorders* + dynamic coverage prunes with full-suite backstop (Adopt). |
| R3 Plan-score hard gate | M4 Plan-score downgrade-only prior | Soft signal *removes* a candidate pre-execution (Reject) vs soft signal *deprioritizes* an still-runnable candidate (Adopt). |
| R4 Share-all / real-time push | M5 Phased abstracted negative sharing | Broadcast raw trajectories mid-flight (Reject) vs share abstracted negatives at turn boundaries, producer-only (Adopt). |
| R5 Thin-executor default | M6 Verification-gated sub-role cascade | Cheap models everywhere on hard SWE (Reject) vs cheap models on run/verify + narrow edits, frontier on navigation/review (Adopt). |

The unifying invariant across all five dividing lines is the Cardinal Safety Contract: the Reject side always lets a soft/cheap/static signal become load-bearing for *exclusion or promotion without execution evidence*; the Adopt side always confines it to *re-ranking, prioritization, or downgrade within the bounds execution still gets the final say.*

### 18.7 What this ledger commits the rest of the plan to

The downstream sections inherit the following hard commitments from this ledger. Each is phrased as a check a reviewer can run against the corresponding section:

1. **Section 9 (speculative tree-search)** must build M1/M2 inside FrontierSearch budget caps with a best-of-N collapse floor; it must *not* present a free-standing distributed MCTS (R1) as the core loop, and must *not* claim exponential savings for `speculate()` (M2 qualification).
2. **Section 10 (CTDG + pruning)** must build M3/M4 as prioritizers with dynamic-coverage pruning and a full-suite backstop; it must *not* introduce a static-AST gate (R2) or a pre-execution hard plan-score gate (R3).
3. **Section 11 (blackboard)** must build M5 as phased/abstracted/negative-only at boundaries with producer-only scope; it must *not* implement share-all or real-time mid-subprocess push (R4), and the verifier must never read producer context.
4. **Section 12 (model economy)** must build M6 as a verification-gated sub-role cascade; it must *not* default to thin-executor-everywhere on hard repo SWE (R5).
5. **Section 13 (verification)** must keep the Cardinal Safety Contract (A3) verbatim as an engine-level invariant; every M1–M6 soft signal is downgrade/re-rank-only.
6. **Section 14 (controller)** must ship the Stage-0 bandit (M7) day one, blend-not-switch, fail-open; GEPA (D1) and RL (D2) are deferred, not built in v1.0; `library_enabled` is wired or removed.
7. **Section 15 (isolation/determinism/resume)** must keep A5/A6 verbatim, build A9 durable journaled resume (the explicit "do better than the reference impl" mandate), and reproduce artifacts not token streams (R7).
8. **Section 16 (speed/cost)** must build A15 prefix-stable assembly and replace the non-adaptive fixed-K default (R6) with adaptive low-K (A12); the 16-rollout cap survives only as the thin-feedback floor.

Any downstream section that needs to deviate from one of these must first amend this ledger — updating both the prose disposition and the `accepted_mechanisms` cross-reference index (18.5) — with a fresh verdict or digest basis. That is the single mechanism by which the in/out list stays canonical.

## 19. Novelty & Scientific Contributions

This section states, precisely and falsifiably, what APEX-Ω contributes to the
scientific literature and what it does **not**. It is written defensively: the
orchestration field is crowded as of mid-2026, and the fastest way to a desk
rejection is to claim, as a headline, a mechanism that is already published or
that already exists inside APEX v1. We therefore (a) disclaim prior art
explicitly, (b) name the single primary contribution and its central falsifiable
hypothesis with concrete falsification conditions, (c) name the secondary and
tertiary scientific contributions, and (d) specify the analyses reviewers now
expect. Every claim is anchored to a controlled experiment defined in outline
here and in full in Section 20 (Evaluation Plan & Experiment Matrix). The
disposition of every mechanism referenced here is fixed by Section 18 (Fusion
Ledger), to which this section remains consistent.

The load-bearing framing — repeated throughout — is the one from Section 1
(Central Thesis): **search/economy are bounded amplifiers; execution evidence is
both the steering signal and the brake; verified best-of-N is the floor we can
never do worse than.** The novelty rides on a *conjunction*, not on any single
mechanism. The mechanisms are substrate; the controller and its evaluation are
the science. The contribution is expressed entirely over the vendor-neutral
substrate (Sections 2–3): it must hold on **Codex, Claude Code, or a mixed
fleet**, never on one vendor's idiosyncrasy.

### 19.1 What is NOT the contribution (explicit prior-art disclaimers)

A reviewer must see, on the first page of any submission, that we know the
landscape and are not re-selling it. The following are prior art (external) or
v1 antecedents (internal). Each is *used* in APEX-Ω; **none is claimed as the
headline novelty.** Naming any of them as the contribution invites rejection on
prior art.

| Mechanism | Prior art / antecedent | Where it lives in APEX-Ω | Why it is NOT novel |
|---|---|---|---|
| Learned RL orchestrator that prunes/sequences agents for cost-quality | Puppeteer ([2505.19591](https://arxiv.org/abs/2505.19591), NeurIPS'25); AgentConductor ([2602.17100](https://arxiv.org/html/2602.17100)); AFlow ([2410.10762](https://arxiv.org/abs/2410.10762)); MaAS | §14 active controller (substrate) | Published. "RL controller that prunes agents to win cost-quality" *is* Puppeteer. It is our **baseline to beat**, not our headline. |
| MCTS / tree search over code states (FrontierSearch MCTS) | SWE-Search ([2410.20285](https://arxiv.org/abs/2410.20285)); AB-MCTS ([2503.04412](https://arxiv.org/abs/2503.04412)); re-described by v1's `FrontierSearchController` | §9 search layer | v1 already ships PUCT/best-first search over branchable checkpoints with value backup, virtual loss, and verification-as-early-stop. The redesign's "novel MCTS" is largely this. |
| Blackboard with negative-discovery sharing (bMAS/LbMAS blackboard) | bMAS/LbMAS ([2507.01701](https://arxiv.org/abs/2507.01701)); v1's `EpisodicMemoryBus` | §11 blackboard | v1 already shares negative/ruled-out discoveries, relevance-ranked, dedup'd, own-rollout-excluded. The only delta is delivery schedule (push-at-turn-boundary). |
| Contract-driven codegen (`contract_slice`) | v1's `core/contract_slice.py`; planner-executor literature | §12 model economy | v1 already builds `# Contract Slice` prompt blocks and carries contract obligations on the `TaskBlackboard`. The model-tier split is incremental, not a new coordination pattern. |
| Orchestration-as-code / durable execution / interop | Temporal; DBOS; LangGraph 1.0; Microsoft Conductor; MCP/A2A | §2 engine; §15 durable resume | A systems/engineering pattern, GA in industry. Durability/DSL alone does not clear a top ML venue; it must *enable* a learning claim. |
| Process reward models / value over partial trajectories | Lightman et al. ([2305.20050](https://arxiv.org/abs/2305.20050)); THINKPRM ([2504.16828](https://arxiv.org/abs/2504.16828)); SWE-TRACE ([2604.14820](https://arxiv.org/abs/2604.14820)) | §13 verifier; §14 transition reward | Trained step-PRMs are well studied (and fragile for code). Our transition reward is *execution*-grounded, which inverts the usual PRM framing rather than re-proposing it. |
| Difficulty-aware budget caps | DAAO ([2509.11079](https://arxiv.org/html/2509.11079v1)); AgentConductor; Snell et al. ([2408.03314](https://arxiv.org/abs/2408.03314)) | §14 allocation; §16 cost | Difficulty-adaptive allocation is established. We adopt it (v1's `enable_adaptive_allocation`, default-ON in APEX-Ω) but do not claim it. |

We say all of the above, in this order, *before* stating what is new.

### 19.2 Primary contribution — Open-Pool Cross-Vendor Search-Policy Generalization

The single defensible, NeurIPS-grade contribution is **a learned orchestration
controller that generalizes its search/route policy zero-shot to a heterogeneous
pool of coding workers it was never trained on**, while keeping every accept
gated by execution evidence, on long-horizon repo-level SWE.

The mechanism that makes this possible — and the concrete, ablatable novelty —
is the **worker representation**. Most published learned orchestrators
(Puppeteer, AgentConductor) train and test on a *fixed, known* pool and
represent each agent by *identity* (one-hot); AFlow/MaAS search over
operator/workflow nodes, and AOrchestra creates sub-agents at runtime with
capability profiles — but none demonstrates open-pool cross-vendor
generalization to a held-out vendor with no retraining. APEX-Ω
instead represents every worker by a **learned capability/cost profile vector**
(extending the routing idea of MoMA
[2509.07571](https://arxiv.org/html/2509.07571v1) and DAAO from short tasks to
long-horizon orchestration). A vendor or model added at inference is described by
a profile the controller already understands, so it can be routed *without
retraining*. This is the difference between learning "route to `claude-opus`" and
learning "route to a *high-localization, high-cost* capability shape."

#### 19.2.1 The controller decision and the profile representation

At each decision node the controller emits an action conditioned on
`(node_features, budget_remaining, [profile vectors of available workers])`:

```text
action = (
  action_type    : Enum{WIDER, DEEPER, SPECULATE, STOP},  # allocate diversity / refine / fork / halt
  worker_target  : ProfileRef,    # a capability/cost PROFILE slot, NOT a (vendor,model) identity
  task_mode      : Enum{CONTRACT, FREEFORM},
)
```

The worker is selected by *profile*, then bound to a concrete `(vendor, model)`
at dispatch time by the Executor layer (§3). This indirection is the whole point:
the policy never names `claude-opus` or `gpt-5`; it names a *capability shape*.

Proposed data structures (build on EITHER Codex or Claude Code workers; both are
leaf executors behind the normalized interface of §3):

```python
@dataclass(frozen=True)
class CapabilityProfile:
    # Learned, low-dim vector of what a worker is GOOD at and what it COSTS.
    # Estimated online from the verifier signal v1 already produces; NOT a one-hot id.
    vector: tuple[float, ...]                 # e.g. 16-32 dims; learned embedding
    # Interpretable anchors (logged for the emergent-structure analysis, §19.6):
    est_resolve_rate_by_difficulty: dict[str, float]  # {"easy":.., "med":.., "hard":..}
    est_localize_skill: float                 # navigation / multi-file edit competence
    est_edit_skill: float                     # narrow single-file edit competence
    est_cost_per_turn: float                  # normalized cross-vendor $ (see §16)
    est_latency_per_turn: float
    n_observations: int                       # confidence in the estimate
    last_updated_seq: int                     # journal sequence (for replayable credit assignment)

@dataclass(frozen=True)
class WorkerBinding:
    profile_ref: str                          # the slot the controller chose
    vendor: str                               # resolved at dispatch: "codex_cli" | "claude_cli" | ...
    model: str                                # canonical->launcher id at command-build time
    cli_version: str
    capability_profile: CapabilityProfile
```

The profile vector is updated from observed outcomes (resolve/abstain, cost,
localization-survival) by a simple online estimator (running posterior). The
estimator and the routing policy are deliberately **decoupled**: estimating a
worker's profile is cheap and online; the policy reads profiles as features. This
is what permits a *new* worker — never seen in training — to receive a profile
within a handful of observations and be routed sensibly.

#### 19.2.2 Why the conjunction is unclaimed

The novelty is the **conjunction**, which no single existing paper demonstrates:

```text
(open-pool vendor-agnostic routing via learned capability profiles)
  ×  (execution-authoritative grounding — the Cardinal Safety Contract as a search invariant)
  ×  (long-horizon repo-level SWE, not function/competition-level code)
  ×  (contamination-resistant, cost-matched evaluation WITH a held-out-vendor split)
```

- Puppeteer (NeurIPS'25) does learned orchestration but on a fixed in-house pool
  and **not on SWE-bench** (GSM-Hard / MMLU-Pro / SRDD).
- AgentConductor / AFlow / MaAS learn topology but on **function/competition
  code** (HumanEval/MBPP/APPS/CodeContests), not repo-level.
- SWE-Search / SWE-TRACE do PRM/MCTS search on repo-level SWE but with
  **hand-designed (not learned) controllers** and largely single-model agents.
- MoMA/DAAO do capability-profile routing but on **short tasks**, not
  long-horizon orchestration, and not with a held-out-vendor generalization test.

The clearest unclaimed gap (corroborated by the SOTA novelty survey) is a
*learned* controller that generalizes zero-shot to an *open* heterogeneous pool
on *execution-verified repo SWE*. That is our headline, and the
capability-profile representation is its falsifiable mechanism.

### 19.3 Central falsifiable hypothesis (H1) and its falsification conditions

We state H1 as a falsifiable claim with an explicit, pre-registered
falsification test. Experiment specifics (benchmark slices, budgets, baselines)
are detailed in Section 20; here we state the hypothesis, the measurement, and
the conditions under which we declare it **false**.

> **H1.** On contamination-resistant repo-level SWE (SWE-bench Pro
> [Scale AI leaderboard](https://labs.scale.com/leaderboard/swe_bench_pro_public)
> and SWE-bench-Live) under standardized scaffolding and **cost-matched** token/$
> budgets, the open-pool controller dominates the cost-quality Pareto frontier
> against (i) the strongest single vendor in the pool, (ii) cost-equal verified
> best-of-N, and (iii) a re-trained published learned orchestrator
> (Puppeteer/AFlow) on the same pool — **AND** it retains that dominance when the
> test-time pool contains a vendor/model **held out from training**, with **no
> controller retraining**.

The held-out-vendor split is the experiment that makes or breaks the claim and
must not be omitted. Concretely: train the controller with a pool of, e.g.,
`{Codex-family, Claude-family}` workers (a frontier/cheap mix); at test time
introduce a *third* vendor/model (e.g., a Gemini-family or open-weight worker)
the controller never saw, give it only a freshly-estimated profile, and re-run
the Pareto evaluation with **no policy weight/prompt update**.

#### 19.3.1 Falsification conditions (pre-registered)

H1 is **falsified** if either holds:

1. **Held-out collapse.** On the held-out-vendor split, at matched token/$
   budget, the controller **fails to beat cost-equal verified best-of-N**. This
   means the learned routing collapsed to "always pick the strongest known
   model" and added nothing over capability-profile-blind allocation — the
   open-pool claim is empty.

2. **Profile-vs-one-hot null.** An ablation that replaces the learned
   capability/cost profile representation with **one-hot vendor IDs** does **not
   measurably hurt** held-out-vendor performance (within noise). This means the
   controller learned *identity*, not *transferable capability*, and the headline
   mechanism is doing no work.

Both conditions are decisive and symmetric: condition 1 attacks the *outcome*
(does open-pool routing help on unseen vendors?), condition 2 attacks the
*mechanism* (is the profile representation the reason?). If either fails, we
report it as a negative result rather than overclaiming — itself publishable
(§19.6).

#### 19.3.2 Honest priors on H1 (what we expect to be hard)

We are explicit about where H1 is likely true vs. unproven, per the adversarial
SOTA reading:

- **Likely true:** the controller beats (a) any single *non-frontier* model and
  (b) naive best-of-N at equal cost, and wins on the **cost axis** (cheaper for
  equal quality). Heterogeneous-fleet evidence (Devlo/TRAE; Multi-LLM AB-MCTS,
  [Sakana](https://sakana.ai/ab-mcts/)) shows cross-vendor diversity + an
  execution-grounded selector beats single-vendor best-of-N and decorrelates
  hallucinations.
- **Hard / unproven:** beating the *single strongest* frontier model on
  **quality at matched cost** on SWE-bench Pro. Frontier single models are very
  strong (Opus 4.5 ~80.9% on the now-deprecated Verified,
  [Anthropic](https://www.anthropic.com/news/claude-opus-4-5)), so the
  quality-at-matched-cost win over the best single model is the genuinely risky
  part. The **open-pool generalization** win (no published baseline does
  held-out-vendor transfer) is where APEX-Ω is most likely to show a clean,
  defensible result.

We therefore frame the headline as the *generalization* result, with the
single-best-model quality win presented as a conditional finding, never a
foregone conclusion. This honesty is itself a guard against the over-claim
pitfall the SOTA digest flags (uncontrolled-cost comparisons, subsampled evals).

### 19.4 Secondary contribution (H2) — execution-grounding as a learnability requirement, not a tax

The usual framing is "safety (execution gating) trades off against capability."
We invert it into a measurable claim.

> **H2.** Because learned verifiers are the binding constraint for SWE selection
> and reward-hacking *scales with capability*, **relaxing** the Cardinal Safety
> Contract (allowing soft signals to *promote* unverified candidates, not merely
> re-rank within or downgrade) **degrades** the learned controller — by feeding
> it a reward-hacked training signal — rather than improving it.

This is a scientific result, not an engineering preference. The premise's
evidence base is strong: reward hacking on real coding RL environments
generalizes to sabotage/deception (Anthropic,
[2511.18397](https://arxiv.org/abs/2511.18397)); imperfect verifiers admit false
positives that RLVR exploits ([2604.15149](https://arxiv.org/abs/2604.15149));
ImpossibleBench shows frontier models exploit tests up to 76% of the time
([2510.20270](https://arxiv.org/html/2510.20270v1)); and verifier noise has a
phase transition (Youden's J = TPR − FPR; J < 0 collapses training, RLVeR
[2601.04411](https://arxiv.org/abs/2601.04411)).

#### 19.4.1 The H2 ablation

```text
Condition A (default, Cardinal Contract intact):
  execution evidence is authoritative; soft/critic/LLM-judge signals may only
  RE-RANK within an execution-verified tier or DOWNGRADE an accepted candidate.
  _apply_evidence_bound_review flips True->False only; never False->True.

Condition B (relaxed, ablation):
  soft signals are allowed to PROMOTE an unverified candidate above a verified one
  (i.e. the contract is removed).

Measure: controller training stability + final held-out resolve rate +
reward-hack incidence (canary tasks from ImpossibleBench-style impossible specs;
transcript hack-signature auditing for sys.exit(0) / __eq__-override / pytest-patch).
```

**H2 is supported** if Condition B shows *lower* held-out resolve rate and/or
*higher* reward-hack incidence than Condition A. This is the falsification-shaped
test: if relaxing the contract *helped*, H2 would be false and we would report
it. We expect H2 to hold (the digest is one-directional on this), which converts
a safety invariant into a *learnability* result — the publishable inversion.

This is also why the Cardinal Safety Contract (§13) is kept verbatim and treated
as a **search invariant**, not merely a selection rule: it grounds the
controller's reward and makes it hack-resistant, a precondition for H1 being
learnable at all.

### 19.5 Tertiary systems contributions (made to pay rent algorithmically)

Systems engineering alone does not clear a top ML venue (§19.1). Each tertiary
contribution is therefore framed so that it *enables a learning or safety claim*.

#### 19.5.1 Durable replay-by-artifact as a learning enabler

We promote v1's narrow durability machinery — `ReplayRecorder` (currently
record/verify-only, **no production callsite** per the v1 ingest) and the escrow
WAL (CCEDF, a narrow one-candidate backstop) — into a **per-`agent()`-call
journal** keyed by an input hash `(prompt, model, vendor, scoped_inputs)`,
persisted to a WAL (Postgres a la DBOS, or a file WAL), restart-survivable
(Temporal/DBOS deterministic-replay model). On restart, unchanged calls return
cached results; only edited/new calls re-run.

The *algorithmic* payoff: a journaled, deterministic decision trace is exactly
the substrate for **reproducible off-policy credit assignment** over
orchestration decisions. We do **not** claim bit-reproducible agent *output*
replay — impossible across hosted APIs (temp-0 batch non-invariance), a rejected
mechanism in §18. We reproduce **artifacts** (diffs + re-run verification). So
the contribution is: *durable replay-by-artifact turns a systems feature into a
credit-assignment substrate that makes the learned controller trainable
off-policy and auditable.*

```python
@dataclass(frozen=True)
class JournalEntry:
    seq: int
    input_hash: str                    # SHA over (prompt, model, vendor, scoped_inputs, prefix_hash)
    decision: dict                     # the controller action emitted at this node
    binding: WorkerBinding             # which (vendor,model) the profile slot resolved to
    artifact_ref: str                  # path to recorded diff (NOT token stream)
    verification: dict                 # re-runnable: {tests, rc, accepted: bool}
    cost: dict                         # {tokens_in, tokens_out, cached_read, normalized_usd}
    run_manifest_ref: str              # pins git SHA, model versions, docker digests
```

#### 19.5.2 A formal safety boundary for test-impact pruning

We state — and prove the conditions for — exactly when test-impact pruning
preserves execution-authority. This is the honest answer the redesign's CTDG
dodged. The boundary (full mechanism in §10):

| Layer | Operation | False-negative risk | Allowed as a gate? |
|---|---|---|---|
| Static import/call graph (CTDG) | **Reorder/prioritize** tests; seed branch priors | Zero (reordering cannot drop a test) | Yes (priority only) |
| One-time dynamic coverage map (coverage.py contexts / block checksums) | **Prune** to at-risk set during iteration | Near-zero (dynamic, but stale to new code) | Yes, *only* with backstop |
| Full-suite run at final pre-accept state | **Backstop** — authoritative | N/A (this is the authority) | Yes (it is the gate) |

Formal statement: pruning preserves execution-authority **iff** every accept
decision is preceded by at least one full-suite (or upstream-harness) run on the
candidate's final state, **and** no candidate is *suppressed* (removed from
contention) on a static signal alone. Static-AST-as-gate is rejected (§18):
PyCG-class recall ~70%, with reflection/monkeypatch/fixtures/parametrize/conftest
statically invisible and the pytest collected set not statically enumerable —
gating would silently drop fault-revealing tests, an inverse-equivalent violation
of the Cardinal Contract. This boundary is a *publishable systems result*: it
specifies the safe operating region for a pruning optimization that the SWE
literature applies without a stated guarantee.

#### 19.5.3 Prefix-stability as a portable cost contract

Branching's economic advantage (constant-factor, not exponential) depends
entirely on prefix reuse, which opaque vendor CLIs do not expose as server-side
KV control. We make prefix-stability a *first-class engine contract*: every
worker prompt is assembled `[stable: tooling + system + contract]` then
`[volatile: task + live context]`, byte-identical across siblings, and a
provider-cache adapter exploits Anthropic cache breakpoints / OpenAI–Gemini
auto-cache (~90% off cached reads). Dispatch is ordered longest-shared-prefix
first (KVFlow-style) to maximize hits.

The contribution is *portability*: prefix-stability realizes branching economics
**uniformly across opaque CLIs that lack server-side KV control**, owned by APEX
rather than by any provider. This is what makes "a branch is cheaper than an
independent trajectory" a *measured* fact instead of a theoretical hope (§16),
and it is what keeps the search layer (§9) net-positive on cost rather than a
money pit.

### 19.6 Reviewer-expected analyses (we will include all three)

A bare accuracy table is now insufficient; the contribution bar includes
interpretability and negative results. We commit to:

1. **Emergent-structure analysis.** Report how the learned policy organizes work
   over training: hub concentration (does one profile become the
   navigation/localization hub?), topology compression (do trajectories get
   shorter/cheaper at equal quality?), and cyclic-recheck patterns — mirroring
   the artifacts now expected from Puppeteer/AgensFlow. The interpretable anchors
   in `CapabilityProfile` (§19.2.1) and the journal (§19.5.1) are logged
   precisely to make this analysis reproducible.

2. **Capability-profile vs. one-hot ablation.** This is *both* an analysis and
   the falsification test for H1 condition 2 (§19.3.1). We report held-out-vendor
   performance under (a) learned profiles and (b) one-hot IDs, with confidence
   intervals; a null result here falsifies the headline mechanism and we say so.

3. **"When orchestration/branching hurts" negative result.** We explicitly
   characterize the regimes where the controller *correctly collapses to verified
   best-of-N* (thin/flaky feedback, giant suites where the verified-primary
   quorum saturates, easy tasks). This pre-empts the "is the controller doing
   anything?" objection, validates the best-of-N floor as a design guarantee, and
   is itself publishable (cf. ChromaFlow
   [2605.14102](https://arxiv.org/abs/2605.14102): more orchestration can
   decrease accuracy and raise cost). Measurement: per-task, log whether the
   controller chose to branch/route or to collapse, and report resolve-rate /
   cost deltas in each regime. If branching *never* helps on any slice, that is a
   strong negative result we report rather than bury.

### 19.7 Relationship to the panel stances (synthesis)

The four panel proposals (search-first, efficiency-first, pragmatic-evolution,
novelty-first) **converge** on the same scientific claim even though they weight
the engineering differently. We record this convergence because it strengthens
the novelty framing: it is robust to which stance ultimately drives the roadmap.

| Stance | Where its novelty story lands | Agreement with §19.2 |
|---|---|---|
| Novelty-first (moonshot) | Open-pool cross-vendor search-policy generalization is *the* contribution; H1/H2 as stated | Identical — this section adopts its framing |
| Search-first | "Execution-grounded, budget-aware adaptive branching with (vendor,model) as a search dimension over an open pool" | Same conjunction; emphasizes the search action space as the thing learned over |
| Efficiency-first | "Vendor-agnostic cost arbitrage + test-impact-pruning safety boundary + prefix-stability as portable cost substrate, on a cost-matched Pareto with a held-out split" | Same headline (open-pool, cost-matched, held-out); foregrounds the tertiary contributions (§19.5.2–3) |
| Pragmatic-evolution | "Vendor-agnostic active controller that generalizes zero-shot to a mixed Codex+Claude pool; risky redesign mechanisms ship as ablations" | Same primary claim; insists the risky mechanisms appear as ablations (exactly our §19.6.3 negative result) |

All four explicitly **disclaim** MCTS/blackboard/contracts as novelty and all
four **require** the held-out-vendor experiment. The tradeoff between them is
*risk allocation* (how much to bet on the controller vs. the substrate), not the
identity of the contribution. The plan resolves this by staging the controller
(§14: bandit → GEPA → RL), so the primary contribution is testable at the bandit
stage and the moonshot RL stage is upside, not a single point of failure.

### 19.8 Vendor-neutrality as a scientific asset, not a caveat

The contribution is *only* meaningful because the substrate is genuinely
vendor-neutral (§3). Two points must be explicit to reviewers:

- **The held-out-vendor split is feasible precisely because acceptance is
  vendor-blind.** Correctness is decided on the git diff via the execution
  cascade (§13), so a Codex worker, a Claude Code worker, and a held-out worker
  compete in the *same* best-of-N pool and are scored identically. There is no
  vendor-specific grading path; this is what makes "swap in an unseen vendor"
  scientifically clean rather than an apples-to-oranges comparison.

- **Cross-vendor diversity is a measured quality lever, not just portability.**
  Treating `(vendor, model)` as a profile-mediated routing/diversity axis
  decorrelates hallucinations (Multi-LLM AB-MCTS) and lets one search tree mix
  vendors per node. The *same single* experimental harness runs on Codex, on
  Claude Code, and on mixed fleets, so vendor-neutrality is a *property under
  test*, not an assumption.

### 19.9 Summary: the one-sentence claim and its kill switch

**Claim.** APEX-Ω contributes the first learned orchestration controller that
generalizes its search/route policy zero-shot to an open heterogeneous pool of
coding workers — via learned capability/cost profiles rather than vendor
identities — while keeping every accept execution-authoritative, on
contamination-resistant, cost-matched, long-horizon repo-level SWE with a
held-out-vendor split; plus the scientific result that execution-grounding is a
learnability requirement (H2), and the systems results that make it
reproducible (durable replay-by-artifact, a formal test-impact-pruning safety
boundary, and prefix-stability as a portable cost contract).

**Kill switch.** If the held-out split does not beat cost-equal best-of-N, or if
profiles do not beat one-hot IDs on the held-out split, the headline is
falsified and we publish the negative result. We do not have a paper without the
held-out-vendor experiment; it is run first and gates everything downstream
(Section 20).

## 20. Evaluation Plan & Experiment Matrix

This section specifies how APEX-Ω's claims are tested, what counts as evidence, and the exact baselines, ablations, metrics, and statistical procedures a coding agent (running on Codex or Claude Code workers, possibly mixed) must implement to produce a result a top-venue reviewer will accept. The governing rule, inherited from the Cardinal Safety Contract (see Section 13) and the v1 anti-cheat invariants (see Section 15), is unchanged here: **the upstream Docker harness is the only publishable number; every APEX-private signal is diagnostic-only and reported as a published delta.** The evaluation harness is itself orchestration-as-code (see Section 2) — it runs through the same `solve()` entry point used in production, so benchmark numbers come from the code path a library user exercises, distinguished only by `benchmark_metadata`.

The central claim we must defend (see Section 1 thesis; Section 19 novelty) is narrow and falsifiable: **a vendor-neutral, execution-authoritative dynamic-workflow orchestrator with an open-pool controller wins the cost-quality Pareto frontier on contamination-resistant, execution-verified repo-level SWE tasks, and generalizes to a model pool it never trained on, with no retraining.** Every design choice below exists to make that claim either provable or refutable on a cost-matched axis. We never make a beats-baseline claim off an uncontrolled-budget axis, never rely on a pre-cutoff public benchmark as a capability signal, and never accept exit-code-only passes.

### 20.1 Benchmarks

The benchmark landscape bifurcated in 2025-2026: [SWE-bench Verified](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) is deprecated as a capability signal (OpenAI, Feb 2026: 59.4% of audited hard problems had flawed test cases; all tested frontier models reproduced verbatim gold patches and release-note details; progress flattened 74.9% to 80.9% over six months). We therefore **never report SWE-bench Verified as a capability signal.** It may appear once, in an appendix, only as a contamination/leakage control witness (to demonstrate the gap between a contaminated public set and our contamination-resistant primaries), and is explicitly labeled non-capability.

| Tier | Benchmark | Role | Why | Notes |
|---|---|---|---|---|
| Primary | [SWE-bench Pro](https://arxiv.org/abs/2509.16941) public + commercial | Headline capability | Contamination-resistant (GPL-copyleft public + purchased commercial + held-out); 1,865 tasks/41 repos. Standardized-scaffold scores collapse ~35 pts vs Verified, exposing real orchestration headroom. | Always report public and commercial splits separately; commercial is the honest generalization signal (e.g., GPT-5 41.8% public to 14.9% commercial). |
| Primary | [SWE-bench-Live](https://github.com/microsoft/SWE-bench-Live) | Freshness capability | Monthly auto-updated, multi-language + Windows, post-cutoff tasks via RepoLaunch. Tracks any month's tasks ingested *after* the controller's most-recent training cutoff. | Report per-month cohorts; never pool a month that predates a retrain. |
| Secondary | [Terminal-Bench 2.0](https://www.tbench.ai/leaderboard/terminal-bench/2.0) | Harness-sensitivity sanity | 89 Docker tasks; confirms the Normalized Executor (see Section 3) does not regress shell/agentic fluency across vendors. | Only 89 tasks means wide CIs; 1-2 pt gaps are noise; never a headline. |
| Internal | Private rotating freshness-controlled set | Leakage-proof capability | Freshly authored tasks ingested strictly post-training-cutoff; rotated each cycle; never published task-by-task. | The decisive evidence against contamination skeptics. |

**Private rotating set + leakage scrubber.** We maintain a private, freshly-authored eval set authored after every model in the pool has its training cutoff, rotated each evaluation cycle (retire any task once any pool model's cutoff passes its authorship date). Construction follows the SWE-bench-Live/RepoLaunch model (LLM-driven containerized env provisioning) but with human-authored fail-to-pass tests. Each sandbox the agent sees passes through a **leakage scrubber** before any worker touches it:

```
scrub_sandbox(repo_dir):
  # Destroy every gold/future-state recovery channel; "block the channel, never the neuron".
  git -C repo_dir reflog expire --expire=now --all            # strip reflog
  remove all remotes (origin, upstream, ...)                  # no fetch-the-fix channel
  delete all branches except the working base                 # no future-commit branch
  delete all tags                                             # no release-tag leak
  rewrite/strip commit messages on HEAD..future to placeholder  # no future-commit message text
  for Commit0-style tasks: full git-history flatten           # v1 _flatten_repo_git_history (Section 15.2)
  assert: rev-list --all --count consistent with intended visible history
  assert: no path under repo contains gold_patch/test_patch bytes (structured fingerprint scan)
```

This generalizes v1's load-time gold-field discard (the `SWEBenchTask` dataclass literally has no field to hold `patch`/`test_patch`) and Commit0 history flatten. Grading happens **only** in an upstream-harness grader process the agent cannot reach: hidden/read-only fail-to-pass and pass-to-pass tests live in a separate container/process with no filesystem or network path back to any worker worktree (extends v1's per-rollout `fcntl`-locked isolation, Section 8.2). A worker that synthesizes a "pass" sees errors=1, never passed=1 (v1 silent-`rc==0` to errors=1 rule, Section 12.1).

### 20.2 Baselines

All baselines run under **identical standardized scaffolding** so the only moving variable is orchestration, not model or harness. We follow the Scale-AI standardized-scaffold discipline (the same SWE-Agent/mini-swe-agent shell exposed to every system) precisely because scaffold swings results 15-21 points on a fixed model -- a "beats baseline" claim that conflates scaffold with orchestration is marketing, not science.

| ID | Baseline | What it isolates | Implementation note |
|---|---|---|---|
| B0 | Strongest single vendor/model, single shot | The frontier-model counter-anchor: "is orchestration worth it over one great model?" | Pick per-split (models reshuffle: GPT leads Pro public, others elsewhere). The hardest bar to beat on quality-at-matched-cost. |
| B1 | Cost-equal best-of-N (single vendor) + execution-grounded selection | The inference-scaling floor (see Section 13). | N chosen so total $ approx APEX run $; uses the Cardinal-Contract selector so it is a *fair* best-of-N, not a strawman. This is the floor APEX must never do worse than. |
| B2 | APEX v1 as-shipped (full-cap-16, caps OFF) | The strong incumbent / cost-pathology witness. | Demonstrates adaptive low-K's cost win and that v1's default is the headline pathology (see Section 12 cost hotspots). |
| B3 | Re-trained published learned orchestrator on the *same* pool | Prior-art parity. | [Puppeteer](https://arxiv.org/abs/2505.19591) (NeurIPS'25 RL controller) and/or [AFlow](https://arxiv.org/abs/2410.10762) re-trained on our pool. Required: a reviewer treats learned-controller-that-prunes-for-cost as published prior art, so it must be a baseline we beat, not our headline. |
| B4 | Cross-vendor best-of-N (mixed pool) + selector, **static** routing | Isolates the *learned* controller's marginal value over a heterogeneous pool. | Same mixed pool, same selector, no learned policy -- so any delta over B4 is attributable to the controller, not to vendor diversity alone. |

B0 and B4 together pin down the two confounds reviewers probe hardest: B0 says "did you beat a single frontier model at equal cost," B4 says "is your win just from mixing vendors, or from *learning* how to route them." The minimum reviewer-demanded result (Section 20.6) is dominance over B0, B1, and B3 on a cost-matched Pareto plot **plus** the held-out-vendor split.

### 20.3 Ablations -- one per mechanism

Each ablation toggles exactly one accepted mechanism and **fails open to the heuristic baseline** when disabled, so flipping a switch can never silently move the headline (v1 default-off/fail-open discipline, Section 13.8). Negative controls are first-class: several ablations exist specifically to *demonstrate a safety violation* and confirm that the rejected variant degrades, which is itself a publishable result (cf. [ChromaFlow](https://arxiv.org/abs/2605.14102): more orchestration can hurt).

| # | Ablation | Arms | Hypothesis / expected | Maps to mechanism |
|---|---|---|---|---|
| A1 | Difficulty-adaptive low-K vs full-cap-16 | adaptive-K ON / full-cap-16 | Single biggest cost lever; optimal K often <10 ([Limits of Resampling](https://arxiv.org/abs/2411.17501)). Expect near-equal solve at large cost cut. | Difficulty-adaptive low-K allocation (adopt) |
| A2 | Bounded adaptive-branching | branching ON / collapse-to-best-of-N | Measures search's marginal value *and* the "when search hurts" regime; mandatory collapse below feedback-confidence floor. | Bounded adaptive-branching (adopt-modified) |
| A3 | Early localization-futility gate | gate ON / OFF | Kills the "15/16 doomed" waste; informs allocation, never suppresses a candidate without execution evidence. | Localization-futility gate (adopt) |
| A4 | CTDG test handling | prioritize+dynamic-prune / run-all / **static-CTDG-as-gate (neg. control)** | Reorder + dynamic-coverage prune approx safe with full-suite backstop; static-as-gate must show fault-revealing-test loss leading to solve-rate drop. | CTDG prioritizer (adopt-modified); static-AST gate (reject) |
| A5 | Blackboard | abstracted-negative phased / no-sharing / **raw share-all (neg. control)** | Abstracted negatives preserve diversity; share-all must degrade (~-3.7pp) and homogenize. | Blackboard 2.0 (adopt-modified); share-all (reject) |
| A6 | Model economy | sub-role cascade / heavy-everywhere / **thin-executor-everywhere (neg. control)** | Cheapen run/verify+narrow edits only; thin-executor-everywhere must show worst resolve-drop ([HyperAgent](https://arxiv.org/abs/2506.17208) navigation/multi-file finding). | Model economy (adopt-modified); thin-executor default (reject) |
| A7 | Hybrid verifier | execution+critic / execution-only / critic-only | Critic breaks ties only within execution-verified tier (R2E-Gym ~43% to 51%); critic-only must underperform and violate nothing it cannot promote. | Hybrid verifier (adopt) |
| A8 | **Vendor-mix** | single-vendor pool / heterogeneous pool, controller held constant | Cross-vendor diversity decorrelates hallucinations; expect coverage + selected-rate lift (CodeMonkeys "Barrel of Monkeys" 80.8% coverage). | (vendor,model) as diversity axis (adopt) |
| A9 | **Capability-profile vs one-hot** (H1 falsification) | learned capability/cost profile vectors / one-hot vendor IDs | Profiles are the open-pool enabler ([MoMA](https://arxiv.org/html/2509.07571v1)/[DAAO](https://arxiv.org/html/2509.11079v1)); one-hot cannot route an unseen vendor. H1 fails if one-hot matches profiles on the held-out split. | Open-pool controller (adopt-modified) |
| A10 | **Held-out-vendor generalization** | test-pool contains a vendor/model absent at train time, **no retraining** | The clearest unclaimed NeurIPS-grade gap; controller must route the unseen vendor via profile, fail-open to heuristic if profile is OOD. | Open-pool controller (adopt-modified) |
| A11 | **Cardinal-Contract relaxation** (H2 neg. control) | contract enforced / allow soft signals to **promote** | Must degrade: relaxing promote-ban ships LLM-preferred-but-unexecuted patches; expect solve-rate inversion (the [Inference Scaling Flaws](https://arxiv.org/abs/2411.17501) false-positive mode). | Cardinal Safety Contract (adopt) |

The three negative controls (A4 static-CTDG-gate, A5 raw share-all, A6 thin-executor-everywhere) plus A11 are the load-bearing honesty results: they show that the *rejected* dispositions in the Fusion Ledger (Section 18) are rejected for measured reasons, not taste. A11 specifically is the experiment that proves the Cardinal Contract earns its keep -- if relaxing the promote-ban did *not* degrade, the entire selection-trustworthiness argument would be unfounded.

### 20.4 Metrics

Every metric is reported per-split, per-vendor, and for the mixed fleet, with confidence intervals. The cost-matched axis is mandatory on every comparison plot.

| Metric | Definition | Why it matters |
|---|---|---|
| Solve rate | upstream-harness resolved % (fail-to-pass flips green AND pass-to-pass stays green) | The only publishable capability number; never exit-code-only. |
| Tokens / solve | total prompt+completion tokens / resolved tasks | Cost lever transparency; counts cache-read vs cache-creation separately. |
| Cost per verified-resolved (token yield) | $ (or tokens) / verified-resolved tasks | The honest efficiency number; the headline alongside solve rate. |
| Wall-clock p50 / p95 | per-task latency distribution | `pipeline()` (Section 16.3) should cut p95 from sum-of-slowest-per-stage toward slowest-single-chain. |
| Cost-quality Pareto frontier | resolved-rate vs $ and vs tokens | The reviewer-demanded plot; all dominance claims live here. |
| Best@K vs Pass@K gap | oracle coverage (any sample correct) minus realized selected-correct | Quantifies headroom the verifier leaves; CodeMonkeys 69.8% coverage vs 57.4% selected = 12.4pt gap to close. |
| Abstention rate + precision | fraction abstained; fraction of abstentions that were truly unsolvable-by-us | First-class outcome (Section 13.10); abstention beats a false-positive ship. |
| Futility-detection rate | fraction of doomed runs aborted before the patch loop | A3 gate efficacy; cf. [SWE-Effi](https://arxiv.org/abs/2509.09853) token-snowball avoidance. |
| Cache-hit rate (fleet SLO) | cache_read / (cache_read + cache_creation) tokens | Makes prefix-stable assembly's ~90%-off-cached-reads claim (Section 16) measurable; tracked as a fleet SLO. |

**Cost matching is the spine.** No comparison is permitted on a non-cost-matched axis. When two systems differ in spend, we either (a) match B1/B4's N so total $ equals APEX's, or (b) report only the Pareto frontier (resolved-rate vs $) and claim dominance only where one frontier strictly dominates another. "Beats B0 while spending 5x the tokens" is not a result; it is rejected at write-time by a harness assertion that refuses to emit a delta claim unless `abs(cost_a - cost_b)/max(cost_a,cost_b) <= cost_match_tol` (default 0.10) or the claim is explicitly framed as a Pareto-dominance claim.

### 20.5 Statistical rigor

```
config (eval/stats.yaml):
  cost_match_tol: 0.10              # |dcost|/maxcost ceiling for a point comparison
  flaky_prefilter_reruns: 50       # rerun each task to estimate flake rate
  flaky_youden_min: 0.0            # gate self-improvement on verifier Youden's J > 0 (strict >)
  ci_method: "bootstrap_paired"    # paired bootstrap on matched task sets
  ci_level: 0.95
  seeds_per_task: 3                # multiple seeds where engine allows; temp-0 not bitwise reproducible
  contamination_audit: required    # appendix gate; run fails to "publishable" without it
```

- **Cost-matched axis always.** Any uncontrolled-budget delta is rejected (see 20.4). This is non-negotiable and enforced in code.
- **Paired comparisons.** All A-vs-B deltas are computed on the *identical* task set; we report a paired bootstrap CI on the resolved-rate delta, not two independent CIs. Paired design removes per-task difficulty variance, which dominates SWE-bench-style variance.
- **Confidence intervals on deltas.** Every headline number ships with a 95% CI; Terminal-Bench 2.0's 89 tasks (+/-2-3pt) means 1-2pt gaps there are explicitly labeled "within noise."
- **Flaky pre-filter.** Before any task enters the scored set, rerun it ~50x under the unmodified base agent; tasks whose pass/fail is non-deterministic are quarantined (extends v1's NDFF flake firewall, which declares a flake only on positive evidence and never re-runs a real failure, Section 15). Any self-improvement loop (Section 17) is gated on the verifier's Youden's index J > 0 -- a verifier no better than chance may not steer learning.
- **Determinism via artifact replay, not token streams.** Temp-0 is **not** bitwise reproducible across hosted APIs (batch non-invariance); we do not claim bit-reproducible agent *output*. Instead we reproduce *artifacts*: the RunManifest (git sha, python/platform, seed, redacted `APEX_*` env, model ids, Docker digests, harness versions) plus Docker digest pinning plus `apex replay-deterministic --verify` re-apply the recorded diffs and re-run verification, asserting the *resolved/unresolved verdict* reproduces. Multiple seeds per task are used where the engine allows to bound run-to-run variance, but the determinism *claim* is scoped to environment + ordering + verdict, never to the model's token stream (Section 15).
- **Contamination-audit appendix (required).** A run is not "publishable" until the appendix witnesses: leakage-scrubber assertions passed on every sandbox; the verbatim-gold-recall probe (prompt each pool model with the issue and check for gold-patch recall) on every primary task; the private-set authorship-vs-cutoff dates; and the Verified-vs-primary gap as a contamination witness. This operationalizes the field consensus that contamination is the dominant validity threat.

### 20.6 The minimum reviewer-demanded result

A single composite figure + table, and nothing less, clears the bar:

1. **Cost-matched Pareto dominance** on a contamination-resistant benchmark (SWE-bench Pro public *and* commercial; corroborated on a SWE-bench-Live month and the private set): the open-pool controller's resolved-rate-vs-$ frontier **strictly dominates** B0 (strongest single model), B1 (cost-equal best-of-N), and B3 (re-trained published orchestrator on the same pool), with paired bootstrap CIs on the deltas.
2. **Held-out-vendor split, no retraining** (A10): at test time the pool contains a vendor/model absent at train time; the controller routes it via learned capability/cost profiles and still dominates B4 (static cross-vendor routing). This is the open-pool generalization claim -- the unclaimed gap in the literature (Section 19) -- and is what makes the contribution NeurIPS-grade rather than an engineering note.

If (1) holds but (2) fails, we have a strong systems result but **not** the novelty claim, and the paper is reframed accordingly (honest framing per Section 1: search/economy as bounded amplifiers; execution evidence as steering signal and brake; best-of-N as the floor). We never overclaim open-pool generalization off a same-pool result.

### 20.7 Standardized scaffolding & grader isolation

To isolate orchestration from model/harness, every system (B0-B4 and all APEX arms) runs through one **Normalized Executor** interface (see Section 3) exposing the same minimal worker contract:

```
WorkerSpec:
  vendor: str            # "codex" | "claude_code" | ... (mixed pools allowed in one run)
  model_id: str
  capability_profile: vec # learned; NOT a one-hot id (A9)
  cost_per_1k_in/out: float
  sandbox_caps: set      # negotiated; degrade-not-crash if a cap is absent

run_worker(spec, task, scaffold) -> WorkerResult:  # identical scaffold for all vendors
  # scaffold = same shell/tool surface for every vendor (Scale-AI standardized discipline)
  # results to disk, never re-flooded into an orchestrator context (filesystem-as-source-of-truth)
```

The grader is a **separate, agent-unreachable process**: hidden/read-only tests live outside every worker worktree, behind the per-rollout `fcntl` lock and network-default-none sandbox (Section 8/18). A worker can run *its own* tests (CTDG-prioritized, Section 10) but can never read or mutate the hidden fail-to-pass/pass-to-pass suite, and the upstream Docker harness -- pinned by digest -- produces the only published verdict. This closes the "benchmarking with the tests used as the verifier" pitfall (hold-out evaluation tests from selection tests) and the exit-code-only-pass pitfall (the harness asserts the specific fail-to-pass set flips and pass-to-pass holds; a bare `rc==0`, a `rc==124` timeout, or a zero-collected run never counts as a pass -- v1 counting precedence, Section 12).

### 20.8 Artifact replay & reproducibility package

Each published run ships a reproducibility bundle so reviewers can re-derive verdicts:

- **RunManifest** + Docker digest pins + redacted env + model/harness versions.
- **Per-`agent()`-call WAL** (the journaled, restart-survivable resume substrate, Section 15/17): the durable record of every worker dispatch, its inputs (input-hash keyed), and its emitted artifacts, doubling as off-policy credit substrate for the learned controller.
- **Escrow WAL** of confirmed-resolved candidates (idempotent, seq-ordered, latest-wins) so a preempted-but-correct run is never lost from the published coverage (the v1 Commit0 loss fix, Section 16.5).
- **`apex replay-deterministic --verify`**: replays recorded worker I/O and re-runs the harness, asserting the resolved/unresolved verdict reproduces. We state plainly that this guarantees *environment + ordering + verdict*, not the agent's token stream -- temp-0 is not bitwise reproducible across hosted APIs, so we reproduce artifacts (diffs + re-run verification), never token streams.

This package, together with the contamination-audit appendix (20.5) and the cost-matched Pareto + held-out-vendor result (20.6), is the complete evidentiary surface APEX-Omega stakes its central claim on: every dominance assertion is paired, cost-matched, CI-bounded, contamination-audited, and replayable -- and every rejected mechanism (Section 18) has a negative-control experiment showing it degrades.

## 21. Risk Register & Mitigations

This section is the project's standing list of ways APEX-Ω can fail and the engineered defenses against each. It is written to a single discipline borrowed from APEX v1's failure taxonomy (Section 4): **no risk is listed without a concrete mitigation AND a fallback that ships a still-defensible system if the mitigation does not hold.** The two highest-likelihood risks — R1 (the learned controller learns vendor identity, not capability) and R3 (cheap-executor ceiling) — are deliberately treated first and most heavily, because they are the ones most likely to actually fire on real repository SWE. (Note: these risk IDs **R1–R9** are a distinct namespace from the **rejected-mechanism IDs R1–R7 in Section 18**; a bare "R6" refers to the §18 reject *or* the §21 risk depending on the section.)

The register is organized as engineered controls, not aspirations. Every mitigation maps to a primitive, config key, journal record, or CI gate defined elsewhere in this plan, so a coding agent on either Codex or Claude Code can build the defense, not just read about it. The unifying invariants that backstop the whole register are the ones from the Central Thesis (Section 7): **execution evidence is authoritative; soft signals never promote; best-of-N is the floor we can never do worse than; speculation/economy/pruning are bounded amplifiers gated by the Cardinal Safety Contract (Section 13).** Where a mechanism is judged unproven by the adversarial verdicts, it is carried as guarded/optional/default-off, never as a certainty.

### 21.1 Risk register data model

Risks are not prose-only; they are first-class run-time and CI artifacts so the orchestrator can actually detect and react to them. Each risk has a typed record persisted to `<run_dir>/risk_register.jsonl` (atomic append, same `atomic_write_json` discipline as v1 artifacts) and a corresponding CI acceptance gate.

```python
@dataclass(frozen=True)
class RiskControl:
    risk_id: str                 # "R1".."R9"
    title: str
    likelihood: str              # "low" | "low-medium" | "medium" | "medium-high" | "high"
    capability_correlated: bool  # True => gets WORSE with stronger base models
    primary_invariant: str       # which Cardinal/thesis invariant this defends
    detectors: list[str]         # runtime signals + metric names that fire the alarm
    mitigations: list[str]       # engineered controls (config keys / primitives)
    fallback: str                # the still-shippable system if mitigation fails
    ci_gate: str | None          # acceptance test name that must pass to ship the feature
    owner_section: int           # cross-ref to the section that builds the control
```

The orchestrator emits a `RiskEvent {risk_id, ts_seq, detector, value, threshold, action_taken}` to the same journal whenever a detector trips. `ts_seq` is a monotonic sequence number (NOT wall-clock — same rule as the escrow WAL, Section 15), so the risk log is replay-stable. The default action for every capability-correlated risk (R1, R3, R6) is **fail toward the floor**: disable the amplifier, fall back to verified best-of-N, and keep going — never abort a succeeding run (v1's "a cap must never abort succeeding work" invariant, generalized).

The detector→action loop is itself a deterministic, journaled control path so a coding agent can build it identically on either Codex or Claude Code:

```python
# Evaluated at every phase boundary and before any amplifier is engaged.
# Pure function of recorded evidence + config; emits journaled RiskEvents; never raises.
def evaluate_risk_controls(run_state, registry: list[RiskControl], cfg) -> set[str]:
    disabled_amplifiers: set[str] = set()
    for ctrl in registry:                       # deterministic order by risk_id
        for det in ctrl.detectors:
            value = read_detector(run_state, det)   # fail-open: missing signal -> None
            if value is None:
                continue                            # absent evidence never trips a control
            threshold = cfg.thresholds[det]
            if breaches(det, value, threshold):
                # The action NEVER excludes a candidate; it only disables an amplifier
                # or quarantines (downgrade-only, per the Cardinal Safety Contract).
                action = ctrl_default_action(ctrl)  # e.g. "disable_learned_routing"
                journal(RiskEvent(ctrl.risk_id, next_seq(run_state),
                                  det, value, threshold, action))
                apply_to_floor(run_state, action)   # idempotent; flips a feature flag off
                disabled_amplifiers.add(ctrl.risk_id)
    return disabled_amplifiers   # caller runs the still-defensible floor for these

# A disabled amplifier is sticky for the run (a flapping signal cannot thrash the engine);
# re-enable only across runs after the gating CI test for that risk passes again.
```

The invariant enforced by `apply_to_floor` is that **every action is monotone toward the floor**: it can disable an amplifier, route heavier, or quarantine a candidate (True→False), but it can never promote an unverified candidate, exclude a candidate pre-execution, or abort an in-flight succeeding rollout.

| Risk | Title (abbrev) | Likelihood | Cap.-correlated | Default action on detect | Floor it falls back to | Owner §|
|------|----------------|------------|-----------------|--------------------------|------------------------|--------|
| R1 | Controller learns vendor identity | medium-high | yes | disable learned routing → heuristic | bandit router + cross-vendor diversity | 14 |
| R2 | Search signal too weak to beat floor | medium | no | collapse to verified best-of-N | adaptive-K best-of-N | 9 |
| R3 | Cheap-executor ceiling / contract underspec | medium-high | partly | escalate to frontier; route heavy | frontier-everywhere | 12 |
| R4 | Pruning/caching degrades trust anchor | medium | no | full-suite backstop; reorder-only | full regression prune | 10 |
| R5 | Determinism/replay broken by speculation | medium | no | journal cancels; seed from hash | sequential, no speculation | 15 |
| R6 | Reward-hacking / contamination | high | yes | quarantine candidate; hidden-test gate | upstream harness only | 13 |
| R7 | Harness-leak erases diversity | medium | no | conformance test fails loud | single-vendor floor | 3 |
| R8 | Vendor CLI drift | high | no | failure-memory eviction; fail-soft | other vendor / API path | 3 |
| R9 | Model-authored control-flow nondeterminism | low-medium | no | freeze-then-journal; ban RNG/clock | deterministic planner-authored | 2,15 |

---

### 21.2 R1 — The learned controller learns vendor IDENTITY, not transferable capability (kills the paper)

**Likelihood: medium-high. Capability-correlated: yes.** This is the single highest-stakes risk because the open-pool active controller (Section 14) is the plan's defensible NeurIPS-grade contribution, and the failure mode is plausible, not hypothetical. Capability-profile estimation for opaque CLI workers is hard, the strongest single model often dominates hard SWE, and a controller given `(vendor, model)` as features can trivially memorize "Claude on hard tasks" instead of learning "this *capability profile* on this *task profile*." If it does, the contribution is an overfit lookup table that will not transfer to a held-out vendor — and the headline scientific claim collapses.

#### Why the naive mitigation is insufficient

You cannot detect this with in-distribution validation accuracy: a vendor-identity memorizer scores *better* in-distribution than a capability learner. The detection MUST be a held-out-vendor protocol, and it must be built first, before any learned routing ships.

#### Mitigations (engineered, in build order)

1. **Held-out-vendor evaluation as a continuous gate, built FIRST.** Before Stage-0 of the controller (Section 14) ships, build the eval harness that trains the controller with vendor V excluded entirely from training, then measures routing quality on tasks where V is in the pool. Config: `controller.heldout_vendor_eval = ["<vendor>"]`, run on every controller artifact. The gate (`ci_r1_heldout_transfer`) requires the learned router to beat the heuristic router on the held-out vendor by a pre-registered margin; if it does not, the learned router is auto-disabled for production (`controller.learned_routing_enabled` flips to `false`, journaled as a `RiskEvent`). This mirrors the failure-modes guidance to validate only on uncontaminated, capability-attributable signal, not in-distribution deltas ([Limits of Inference Scaling Through Resampling](https://arxiv.org/abs/2411.17501)).
2. **Capability-profile-vs-one-hot ablation (the negative result is itself reportable).** Run the controller in two arms: (a) features = learned capability/cost profiles (per-role resolve rate, token yield, calibration, latency); (b) features = one-hot `(vendor, model)` identity. If arm (a) does not generalize better than arm (b) on held-out vendors, that *is* the paper's honest finding ("when learned routing helps and when it just memorizes"). Config: `controller.feature_ablation = ["profile", "onehot"]`. The profile features are deliberately **vendor-blinded at inference**: the policy sees the capability vector, not the vendor string, so identity cannot leak through the feature path.
3. **Staged design so partial failure still ships value (bandit → GEPA → RL).** Stage-0 ships a bandit router that learns per-role arm values from execution-grounded reward; even if GEPA (Section 14, deferred Stage 1) and full RL (deferred Stage 2) fail to generalize, the bandit + cross-vendor diversity (Section 3, the `(vendor, model)` diversity axis) still yield a working, cost-reducing system. This is the explicit "adopt-modified, staged, blend-not-switch, fail-open to heuristic" disposition from the accepted-mechanisms list.
4. **Blend-not-switch + fail-open.** The controller never hard-switches the whole fleet to one vendor; it blends allocation across the portfolio and falls open to the deterministic heuristic router on any controller error or low-confidence decision. A degenerate "always route to vendor X" policy is detectable (allocation entropy collapse) and is itself a R1 detector.

#### Detectors

- `controller.allocation_entropy` per task-difficulty bucket: a sharp drop toward a single vendor on hard tasks is a memorization warning.
- `ci_r1_heldout_transfer` margin: learned-minus-heuristic resolve rate on held-out vendor; negative ⇒ disable.
- Feature-attribution audit: if vendor-identity-correlated features dominate the learned policy's decisions, flag.

#### Fallback (still defensible without the learned win)

Ship APEX-Ω as a **cost-reducing systems contribution + a rigorous "when does learned routing help?" study.** The substrate (orchestration-as-code, durable resume, pipeline streaming, vendor-neutral executor), the Cardinal-Safety-gated verify-and-refute, cross-vendor diversity, adaptive-K, and the verification cascades all stand alone and beat the naive baselines independently. The negative result on RL generalization becomes a reported finding, not a project failure — exactly the framing the chief prose mandates.

---

### 21.3 R2 — Search steering signal too weak/noisy to beat the best-of-N floor

**Likelihood: medium. Capability-correlated: no.** Bounded adaptive-branching and agent-initiated `speculate()` (Section 9) only pay off when the feedback signal that steers branching is informative. Where it is not — giant or flaky test suites, weak F2P signals, the verified-primary quorum saturating early — search adds wall-clock and tokens for no resolve-rate gain, and can underperform plain repeated sampling. This is the well-established result that verifier-guided search loses to repeated sampling at scale because scorers are least reliable early ([Limits of PRM-Guided Tree Search](https://arxiv.org/abs/2510.20272)) and that under strict budgets adaptive-branching can lose to plain sampling ([BAVT](https://arxiv.org/abs/2603.12634)). The adversarial verdict on distributed MCTS is **unsound**; the adversarial verdict on speculative branching is **partially sound** (constant-factor, not exponential, and contingent on prefix reuse). The mitigations honor both.

#### Mitigations

1. **Pruning is always a HINT, never a gate.** Per the Cardinal Safety Contract and the CTDG verdict ("prioritize, don't prune"), any search-derived prioritization may only reorder exploration or set branch priors/budget share. The authoritative prune remains the full regression-prune-by-baseline (Section 10), which re-runs only baseline-passing tests and expands collection-error file keys — execution evidence decides, the search signal only schedules.
2. **Mandatory collapse to verified best-of-N below a feedback-confidence floor.** The branching controller computes a `feedback_confidence` scalar (signal-to-noise of the per-branch reward: F2P presence, suite flakiness estimate, quorum margin). When `feedback_confidence < branching.min_feedback_confidence` (default tuned per benchmark), the engine collapses to adaptive-K verified best-of-N for that task. This is the load-bearing "best-of-N is the floor" guarantee made operational.
3. **Hard growth bounds (reuse FrontierSearch machinery).** Virtual-loss de-duplication, `min_branch_reward`, `max_depth`, and `max_frontier_branching` (the v1 FrontierSearchController constants: `c_puct=1.25`, `virtual_loss=0.15`, `max_depth=6`, `max_frontier_branching=3`, `min_branch_reward=0.12`) bound the tree so an uncontrolled fan-out can never exceed the best-of-N baseline cost. `speculate()` forks feed the *same* ranking/budget machinery (Section 9), inheriting these bounds — the verdict's explicit recommendation.
4. **Reward honesty: verifier Youden gating + flake firewall.** Before any branch reward is trusted as a steering signal, the verifier's Youden index (J = TPR − FPR) must be > 0 on the per-repo signal, and the NDFF flake firewall (v1 Section 12) must not have flagged the deciding tests. A J<0 signal is treated as noise and the task collapses to the floor ([RLVeR](https://arxiv.org/abs/2601.04411)).
5. **"When search hurts" ablation per benchmark.** A standing experiment (Section 20) measures branching-on vs branching-off per benchmark; if branching does not beat the floor on a benchmark, branching defaults OFF for that benchmark and the result is reported.

#### Detectors

- `branching.feedback_confidence` distribution per benchmark.
- `branching.resolve_delta_vs_floor` (search resolve rate minus adaptive-K floor); ≤ 0 ⇒ default off for that benchmark.
- Verifier `youden_index` per repo; flake-firewall positive-evidence flags.

#### Fallback

**Engine on, branching gated off by default.** `search.enabled=true` runs the engine, but adaptive branching does not engage until `activation_min_nodes` (default 8) is reached — so a normal run is adaptive-K best-of-N + the verification cascades, which are independently proven (repeated sampling 15.9%→56% on SWE-bench Lite, [Large Language Monkeys](https://arxiv.org/abs/2407.21787)). Speculative branching is an opt-in amplifier, not a dependency.

---

### 21.4 R3 — Cheap-executor capability ceiling / contract underspecification erases cost savings

**Likelihood: medium-high on hard multi-file SWE. Capability-correlated: partly.** This is the second-highest-likelihood failure and must not be understated. The vendor-agnostic model economy (Section 12) routes cheap workers to narrow sub-roles to cut cost. The documented trap is the **"almost-right trap"**: a thin/cheap executor that needs 3–4 retries plus review can cost *more* than one frontier pass, and the HyperAgent ablation shows that cheapening navigation/multi-file editing causes the *worst* resolve-rate drops ([HyperAgent](https://arxiv.org/html/2409.16299v1)). The adversarial verdict on heavy-orchestrator/thin-executor is **partially sound — default rejected**; the verdict on the model economy as a verification-gated cascade is **adopt-modified**. Cheap executors also raise the false-positive accept rate, stressing verification harder (interaction with R6).

#### Mitigations

1. **Sub-role tiering keeps the hard roles heavy by default.** Per the HyperAgent ablation, only `run`/`verify`/single-tool/narrow-edit sub-roles are eligible for cheap workers. Codebase navigation and multi-file editing stay on frontier (or competent Sonnet-class, *not* "thin") by default. Config: `economy.cheap_eligible_roles = ["run", "verify", "narrow_edit"]`; `economy.heavy_roles = ["navigate", "multifile_edit", "plan", "final_review"]`. This is the difference between the rejected default (uniform thin executor) and the adopted cascade.
2. **Escalate to frontier on the FIRST verify-on-diff failure, with a hard rewrite-cycle cap.** A cheap executor's output goes through the cheap-first verification cascade (Section 13); the *first* verify-on-diff failure escalates that sub-task to a frontier worker. `economy.max_rewrite_cycles` (default small, e.g. 2) caps cheap retries so the almost-right trap cannot run away. This is the FrugalGPT/Aider cascade pattern ([FrugalGPT](https://arxiv.org/abs/2305.05176); [Aider architect/editor](https://aider.chat/2024/09/26/architect.html)).
3. **Per-repo calibrated escalation gate (self-test signal, not intuition).** The decision to trust a cheap worker uses a calibrated confidence signal — self-generated test pass on the cheap output (the code-cascade pattern) with a threshold tuned on in-domain labeled traces, not a hand-picked constant. Poorly-calibrated cheap models are *rejected* outright (the GATEKEEPER caution: bad calibration cannot be fixed by any threshold, [GATEKEEPER](https://arxiv.org/pdf/2502.19335)).
4. **Continuous token-yield monitoring; route heavy when arbitrage stops paying.** The orchestrator tracks **cost per verified-resolved task** (token *yield*), not gross executor tokens or invoice. When `economy.token_yield_cheap < economy.token_yield_heavy` over a sliding window, the cheap path is disabled for that repo/role and the work routes heavy. This directly answers the xRouter brittleness finding ([xRouter](https://arxiv.org/html/2510.08439v1)) — measure yield, never assume the route transfers.
5. **Contract drift defused by execution authority.** "Contract-driven" sub-tasks are accepted on the git diff against real tests (filesystem-as-source-of-truth), never on the contract text or the executor's self-report — so an underspecified contract surfaces as a verify failure (→ escalate), not a silent wrong-accept.

#### Detectors

- `economy.token_yield` per (repo, role, tier); cheap < heavy ⇒ route heavy.
- `economy.rewrite_cycles` per sub-task; hitting the cap ⇒ escalate + flag the contract as leaky.
- `economy.cheap_calibration` (self-test predictive value); below floor ⇒ reject the cheap pairing.

#### Fallback

**Frontier-everywhere.** The economy is opt-in and per-sub-role; disabling it returns to a single competent-model execution shape that is exactly v1's proven configuration. Cost savings are claimed as **net-of-verification on bounded/medium tasks and modest-but-real on hard repo SWE**, never as an unconditional order-of-magnitude — honoring the partially-sound verdict.

---

### 21.5 R4 — Pruning/caching silently degrades the trust anchor (false-negative test drop)

**Likelihood: medium. Capability-correlated: no.** The CTDG + bidirectional pruning layer (Section 10) and the input-hash resume cache (Section 15) both create a path by which a *correct* candidate is silently dropped or a *stale* result is replayed. Static call graphs are quantifiably lossy in dynamic Python (PyCG ~99.2% precision but only ~69.9% recall; eval/getattr/monkeypatch/fixtures invisible, [PyCG](https://arxiv.org/abs/2103.00587)); the pytest item set is not statically enumerable (only `pytest --collect-only` knows it); aggressive cache reuse can replay an answer for changed code. The adversarial verdict on static-AST-CTDG-as-a-gate is **unsound**; the verdict on CTDG-as-prioritizer + dynamic-coverage-prune + full-suite-backstop is **adopt-modified**. The mitigations enforce that split.

#### Mitigations

1. **Dynamic coverage is the ONLY gate; static graph only reorders.** The static CTDG is used solely to *prioritize* the test set (zero false-negative risk — reordering never excludes), accelerating time-to-first-failure. Any actual *pruning* (deselection) uses dynamic per-test coverage (testmon-style block-checksums over coverage.py / per-language tracer), where over-selection is cheap and a miss merely delays feedback. Config: `prune.static_mode = "reorder_only"` (hard-coded; static-as-gate is unreachable).
2. **Full-suite stabilization backstop at the final pre-accept state.** Before a candidate is accepted, the full baseline-passing suite runs once on the final state (the Google/Facebook "stabilization" pattern, [Predictive Test Selection](https://arxiv.org/pdf/1810.05286)). No candidate is ever accepted on a pruned suite alone — this is what makes pruning honest and preserves execution-authority.
3. **Per-repo safety-mode flag.** `prune.safety_mode ∈ {advisory, prune_with_backstop, prune_hard}`, **default `prune_with_backstop`**. `prune_hard` (no backstop) requires explicit opt-in and is never the default, so the orchestrator never silently gambles safety.
4. **Input-hash cache validity (no stale replay).** The resume WAL keys each cached `agent()` result on a hash of `(prompt, model, vendor, scoped_inputs, code_state_hash)` (Section 15). A changed code state changes the hash, so a cached result is *never* replayed for changed code — closing the stale-replay path. Over-select on hierarchy/lockfile/config/seed changes (testmon's bias toward false positives).

#### Detectors

- `prune.would_have_passed_canary`: periodically run the full suite on a pruned-accepted candidate and assert the pruned and full verdicts agree; disagreement is a false-negative alarm.
- Cache `hash_collision` / `stale_replay` counters (should be zero).

#### Fallback

**Full regression prune (v1's proven primitive).** Disable CTDG and dynamic-coverage selection entirely; re-run all baseline-passing tests in chunks of 50 per candidate, exactly as v1 does. The pruning layer is pure speedup; turning it off costs wall-clock, never correctness.

---

### 21.6 R5 — Determinism/replay broken by speculation (timing/sampling in orchestration code)

**Likelihood: medium. Capability-correlated: no.** The speculative/adaptive-branching layer introduces two classically nondeterministic primitives into orchestration: deadline-triggered hedging timers and Thompson-style wider-vs-deeper sampling (AB-MCTS). Either, if placed in the *workflow* (not *activity*) layer, breaks strict replay subtly — the durable-execution rule is that orchestration code must be deterministic ([Temporal](https://learn.temporal.io/tutorials/go/background-check/durable-execution/); [prompt-caching/durable-execution pitfall](https://arxiv.org/html/2601.06007v2)). The adversarial verdict here is **sound_with_caveats**: determinism *can* be preserved, but only with seeding/journaling, and "bit-for-bit" overstates what v1 guarantees (v1 reproduces artifacts + trajectory-via-replay, not from-scratch-identical trees).

#### Mitigations

1. **Seed all samplers from content hashes.** Any Thompson/Random draw in the search-control layer is seeded from a content hash of the node state, reusing v1's `Random(0)` / content-sha discipline. The wider-vs-deeper decision becomes a pure function of recorded evidence, so a re-run from the journal reproduces it.
2. **Journal every node expansion and cancellation by monotonic seq + idempotency key.** Generalize the escrow WAL (CCEDF) pattern (Section 15): each `speculate()`/expand/prune/cancel gets a `seq` and an idempotency key `(run_id, node_id, attempt)`; replay reconstructs the tree in seq order regardless of original timing, and duplicate appends are harmless. The deterministic selection tuple (terminating in content sha1, *never* insertion order) already makes the *winner* order-independent — exploration order cannot change the result given the same accepted set.
3. **Remove wall-clock hedging from the deterministic path, or journal the cancel decision.** On the deterministic path, replace deadline-triggered hedging with budget/evidence triggers; if a wall-clock hedge is kept, the *cancel decision* is journaled so replay reproduces it from the log rather than re-deriving it from timing. Ban `Date.now`/`Math.random`-equivalents in orchestration (shared with R9).
4. **CI acceptance test: kill mid-run → identical resumed trajectory + identical winner.** `ci_r5_kill_resume` kills a speculative run at a random `seq`, resumes from the WAL, and asserts (a) the resumed trajectory matches the recorded one and (b) the selected winner is identical. This is the standard durable-execution acceptance test and is a gating check before the speculation layer ships.

#### Detectors

- `ci_r5_kill_resume` pass/fail (gating).
- Replay-divergence counter: any node whose replayed decision differs from its journaled decision.

#### Fallback

**Sequential, no speculation.** The speculation layer is an optional amplifier over the verified best-of-N floor; disabling it yields a fully deterministic, replayable run with no hedging or sampling in orchestration — exactly v1's determinism posture.

---

### 21.7 R6 — Reward-hacking / contamination amplified by heterogeneous fleets

**Likelihood: high. Capability-correlated: yes (this gets WORSE with stronger models).** Reward hacking scales with capability (GPT-5 exploits impossible tests 76% of the time, [ImpossibleBench](https://arxiv.org/html/2510.20270v1)) and a heterogeneous fleet fans out *more* attempts, multiplying the surface for false-positive passes. Contamination is the dominant validity threat: OpenAI retired SWE-bench Verified for verbatim gold-patch recall; ~32–33% of "successes" can be solution leakage; widespread harness-level cheating is documented across 9 benchmarks ([DebugML/Meerkat](https://debugml.github.io/cheating-agents/)). This is non-negotiable for a mixed-vendor fleet and is defended by v1's anti-cheat/fairness machinery, generalized.

#### Mitigations

1. **Upstream harness is the ONLY publishable number.** APEX-private rescoring is diagnostic and published as a delta (v1's `fairness_audit.json`, FLAG_THRESHOLD 0.02). The fleet's internal selection signals never become the headline.
2. **Hidden/read-only tests in an agent-unreachable grader process.** The grader runs in a sandbox the worker cannot read or write; default test files are hidden or read-only. This is the single most effective anti-hacking lever ([ImpossibleBench](https://arxiv.org/abs/2510.20270)). The grader process is never the agent's process.
3. **Scrub-at-load + leakage scrubber.** Gold fields are *discarded* (not masked) at load time (v1: the task dataclass has no field to hold the gold patch). The sandbox is scrubbed of git reflog/all-branches/origins/tags/future-commit messages (SWE-bench issue #465 loophole); Commit0 uses a true git-history flatten.
4. **Transcript auditing for hack signatures.** Automated flags for the four ImpossibleBench strategies plus mock fakery: `__eq__`/operator overloads, call-count hidden-state tracking, hardcoded test-input returns, test-file edits, exit-code-only "pass," over-mocking. v1's anti-hack ledger (structured fingerprint, false-negative bias, soft-downweight) is the substrate; cross-vendor it is vendor-blind.
5. **Canary impossible-tasks in production monitoring.** Seed a few tasks whose tests contradict the spec; any "pass" is a direct reward-hacking alarm, monitored continuously, not just at eval time.
6. **Private freshness-controlled eval only.** Capability claims are validated on post-training-cutoff issues from controlled repos (SWE-Rebench / SWE-bench Pro pattern), never on public pre-cutoff sets.

#### Detectors

- `anti_hack.signature_hits` per transcript; nonzero ⇒ quarantine the candidate (downgrade-only, per Cardinal Contract).
- `canary.impossible_pass_rate`; any pass ⇒ alarm.
- `fairness_audit.delta` between upstream and private scorers; > FLAG_THRESHOLD ⇒ investigate.

#### Fallback

**Upstream harness only + abstention.** With every soft/diagnostic signal stripped, the system still selects on execution-authoritative, hidden-test, upstream-harness evidence and abstains (first-class outcome) when there is no positive evidence — the floor where reward-hacking has the least leverage.

---

### 21.8 R7 — Harness-leak erases the cross-vendor diversity gain

**Likelihood: medium. Capability-correlated: no.** The whole cross-vendor diversity advantage (Section 3) is contingent on the executor faithfully realizing each vendor's flags/sandbox/schema. Terminal-Bench 2.0 shows 30–50pt same-model swings across harnesses; standardized scaffolds swing 15–21pt ([Terminal-Bench 2.0](https://www.tbench.ai/leaderboard/terminal-bench/2.0)). A single mismapped flag, wrong sandbox mode, or lossy schema delivery can *nullify* the diversity gain — turning a claimed advantage into a loss. The vendor-neutrality verdict is **sound_with_caveats**: feasibility is sound, but "without losing paradigm benefits" must be downgraded to *bounded, declared degradation*.

#### Mitigations

1. **Treat the executor as part of the harness.** The normalized Executor (Section 3) is versioned and tested as a first-class component, not glue. Capability differences are consolidated into one `CapabilityProfile` (internet, native_schema, sandbox_levels, effort, tool-interception, bidirectional_stream, mcp) with explicit graceful degradation to APEX's own floor (no native schema → embed + post-parse; no read-only sandbox → APEX worktree + fcntl lock).
2. **Per-vendor conformance tests asserting the sandbox/schema actually take effect.** `ci_r7_conformance` per vendor asserts that the mapped sandbox mode is genuinely enforced (e.g., a write outside the workspace is actually blocked) and that the declared schema delivery actually produces validated structured output. A leaky mapping fails the gate loudly, before it can silently erase diversity.
3. **Pin + record resolved CLI versions in the RunManifest.** `npm i -g pkg@X.Y.Z`; the resolved version is recorded so a drift that changes harness behavior surfaces in provenance (shared with R8).
4. **Capability-negotiation handshake with explicit declared support.** Model the ACP `initialize` pattern ([Agent Client Protocol](https://agentclientprotocol.com/get-started/introduction)): a vendor declares what it supports; unsupported capabilities degrade rather than crash, and the degradation is recorded so the diversity claim is measured against what actually ran.

#### Detectors

- `ci_r7_conformance` pass/fail per vendor (gating).
- `harness.schema_postparse_failure_rate` per vendor; a spike means schema delivery silently degraded.
- Cross-vendor `resolve_delta` audit: if a vendor's contribution to the fleet collapses after a CLI update, suspect harness leak.

#### Fallback

**Single-vendor floor.** If a vendor's conformance test fails, drop that vendor from the fleet for the run (failure-memory eviction, shared with R8) and run the remaining vendors or a single competent vendor. Cross-vendor diversity is an additive amplifier; its absence returns to a proven single-vendor best-of-N.

---

### 21.9 R8 — Vendor CLI drift / breaking changes

**Likelihood: high (fast-moving). Capability-correlated: no.** npm-distributed CLIs change fast and break the normalization layer silently (e.g., Codex profile semantics broke at 0.134.0; `--full-auto` deprecated). Unpinned `@latest` is a standing liability for any vendor-neutral orchestrator.

#### Mitigations

1. **Version pinning + recorded resolution.** `npm i -g pkg@X.Y.Z`; the resolved version goes into the RunManifest (digest-pinning discipline, v1 Section 17). No `@latest` in any production path.
2. **Two-tier failure memory + self-evicting BackendPortfolio.** v1's distinction between *call-failover* (current-stage reroute on 429/stall) and *backend-level global reroute* (auth/missing-binary/SDK breakage) means a broken backend is isolated, not propagated; the self-evicting portfolio drops a structurally-broken vendor for the run so a transient blip on one vendor cannot poison a healthy heterogeneous fleet.
3. **Adapters fail soft to other vendors.** A vendor that fails its capability handshake or returns malformed output degrades to another vendor or the OpenAI-compatible API path; the run continues. The executor `run_structured_prompt` analog (Section 3) never raises to the caller — every abnormal exit becomes a typed result.

#### Detectors

- `backend.global_reroute_count` per vendor; a spike after a version change ⇒ pin-and-investigate.
- `manifest.resolved_version` diff between runs (drift surfaces in provenance).

#### Fallback

**Other vendor / API path.** Vendor neutrality is itself the fallback: any single CLI breaking is survivable because the fleet and the API path remain. The worst case degrades to a single working backend, never to a dead run.

---

### 21.10 R9 — Engine determinism regression from model-authored control flow

**Likelihood: low-medium. Capability-correlated: no.** Orchestration-as-code (Section 2) lets a model (or a deterministic planner) author the workflow program. If *live, stochastic* model output becomes the un-journaled control flow, it injects nondeterminism into the exact layer v1 keeps pure (deterministic ranking, pure `assign_strategy`/`get_temperature`, bit-identical snapshot SHAs), undermining strict replay and the RunManifest reproducibility guarantee. The substrate verdict is **sound_with_caveats**: the engine win is real, but model-authored control flow must be frozen/journaled.

#### Mitigations

1. **Freeze-then-journal any model-emitted script.** If a model emits the workflow, snapshot + hash the emitted script into the RunManifest and journal it as a deterministic activity, so replay runs over a *frozen* script. Live model output is never un-journaled control flow. Prefer a **deterministic planner** authoring the workflow where possible; the model proposes, the engine freezes.
2. **Ban `Date.now`/`Math.random` equivalents in orchestration code.** A lint/CI rule (`ci_r9_no_nondeterminism`) statically rejects timestamps, RNG without a content seed, and network calls in the *workflow* layer (these belong only in *activity*/worker calls). This is the durable-execution determinism contract enforced as a build gate.
3. **Wire (or remove) v1's flagged `library_enabled` kill switch.** v1 carries a flagged `library_enabled` switch; it must be either fully wired (so model-authored extensions can be deterministically disabled) or removed (so it cannot become a silent nondeterminism path). Carrying a dead flag is itself a regression risk. *(Genuine uncertainty: the ingest flags `library_enabled` as a flagged switch but does not specify its current wiring state; the re-implementer must verify and resolve it.)*

#### Detectors

- `ci_r9_no_nondeterminism` lint pass/fail (gating).
- `manifest.workflow_script_hash`: a control-flow path not covered by a frozen, hashed script ⇒ flag.
- Shares `ci_r5_kill_resume`: a determinism regression in control flow surfaces as a resume divergence.

#### Fallback

**Deterministic planner-authored workflow.** If model-authored control flow cannot be made reliably deterministic, fall back to the IssuePlan-driven, deterministic-planner workflow (v1's existing shape lifted into the engine). The orchestration-as-code flexibility is then offered only over a frozen, hashed, journaled script — never live.

---

### 21.11 Cross-risk interactions and the unifying control

Several risks compound and must be reasoned about together; the register's detectors are designed so the compounding is caught, not masked:

- **R3 × R6:** cheap executors raise the false-positive accept rate, so the model economy makes reward-hacking detection *more* load-bearing. The cheap-first cascade and hidden-test grader (R6) are therefore prerequisites for enabling the economy (R3), not independent features.
- **R2 × R4 × R5:** the speculation layer (R2/R5) and the pruning layer (R4) both create paths to drop a correct candidate. The unifying brake is identical for all three: **execution evidence is authoritative; soft/static/search signals may only reorder or downweight, never exclude pre-execution; a full-suite/verified-best-of-N backstop always exists.** This single invariant, enforced structurally by the Cardinal Safety Contract's downgrade-only review (`accepted` flips True→False only) and the deterministic ranking tuple (every soft key strictly below every execution key), defends all three at once.
- **R1 × R7 × R8:** the controller (R1), harness conformance (R7), and CLI drift (R8) all concern the integrity of the vendor-neutral substrate. They share the RunManifest version-pinning and the failure-memory eviction machinery as common controls.

The single most important property of this register is that **every amplifier degrades to a proven floor.** Search collapses to verified best-of-N (R2); the economy collapses to frontier-everywhere (R3); pruning collapses to full regression prune (R4); speculation collapses to sequential deterministic execution (R5); the learned controller collapses to a bandit + heuristic router (R1); and the whole vendor fleet collapses to a single working backend (R7/R8). Because the floor is itself a defensible, SOTA-competitive system (execution-grounded verify-and-refute over isolated rollouts with Cardinal-Safety selection), APEX-Ω cannot fail *catastrophically* from any single risk firing — it can only fail *gracefully* back toward a system that already beats the naive baselines. That is the register's design contract, and it is what makes the bolder extensions safe to attempt.

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

**Prefix-stability contract + cache adapter.** Make every agent template emit `[stable: tooling + system + policies]` then `[volatile: task + live context]`; lint/forbid timestamps, UUIDs, session IDs, and dynamic tool sets inside the stable region. A pure API consumer cannot literally share KV across forks, so the portable win is maximizing prefix hit-rate plus dispatch ordering (longest-shared-prefix-first). Compile the stable spans into Anthropic `cache_control` breakpoints or rely on OpenAI/Google auto-cache + a pinned `prompt_cache_key`. Caution: caching below the per-model min token threshold is associated with ~10–18% TTFT variance — selectivity is mandatory.

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

## 23. Comparison Matrices, Glossary & Bibliography

This section is the plan's quick-reference appendix. It is deliberately dense and self-contained: a coding agent (on Codex or Claude Code) or a reviewer should be able to answer "where did mechanism X come from, what did we decide, and which evidence backs it" without re-reading Sections 1-22. Nothing here introduces a new disposition; every entry is traceable to the Fusion Ledger (see Section 18). Where a claim rests on evidence judged *unproven* in the adversarial review, it is flagged as guarded.

### 23.1 The Master Comparison Matrix (v1 vs Redesign vs SOTA-best vs APEX-Ω)

This is the canonical cross-axis comparison. APEX-Ω = "this plan." SOTA-best = the strongest published behavior on each axis as of mid-2026. The redesign column is the v3 *as proposed*, before the ledger pruned it.

| Axis | APEX v1 | Redesign (v3, as proposed) | SOTA-best | THIS PLAN (APEX-Ω) |
|---|---|---|---|---|
| Substrate / orchestration | Bespoke hard-coded `solver.py` pipeline | Prose blueprint; speculative-MCTS spine | Durable execution ([Temporal](https://temporal.io)/[DBOS](https://www.dbos.dev)); orchestration-as-code; [Conductor](https://opensource.microsoft.com/blog/2026/05/14/conductor-deterministic-orchestration-for-multi-agent-ai-workflows/) | Re-implementable deterministic engine (`agent/parallel/pipeline/phase/budget`) lifting v1 |
| Vendor-neutrality | Multi-backend portfolio (6 backends), diff-as-truth | Implicit; opaque-CLI tension unaddressed | Filesystem-as-truth; [ACP](https://agentclientprotocol.com/get-started/introduction)/[A2A](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) interop | Normalized Executor + ACP-style negotiation; **Codex/Claude/mixed in one run** |
| Parallelism | Continuous worker pool (`parallel_workers=3`) under wave-level planning | Speculative forks (uncapped) | parallel + pipeline streaming | Barrier `parallel()` + net-new `pipeline()` streaming |
| Search | FrontierSearch (PUCT) present but planner-driven | "Distributed MCTS" (re-describes FrontierSearch) | Adaptive-branching ([AB-MCTS](https://arxiv.org/abs/2503.04412)) > MCTS only at large budget | Bounded adaptive-branching over FrontierSearch; **best-of-N floor**; MCTS rejected |
| Pruning | Regression-prune (baseline-passers, chunks of 50) | Static CTDG as gate (unsafe) | Dynamic coverage ([testmon](https://testmon.org)) near-safe; full-suite backstop | Static prioritize + dynamic-coverage prune + backstop; **hint never gate** |
| Knowledge sharing | EpisodicMemoryBus (pull, negative sharing) | "Instant push" Blackboard 2.0 (raw, risky) | Abstracted negative + selective admission ([MEMOIR](https://arxiv.org/abs/2605.17539)/LTS, [bMAS](https://arxiv.org/abs/2507.01701)) | Phased, abstracted negative-constraint push at turn boundaries |
| Model economy | Single-vendor default (codex_cli:gpt-5.5, claude failover; `--effort max` on Claude) | Heavy planner + thin executor (default) | Heavy/cheap split with competent editor + calibrated cascade ([Aider](https://aider.chat)) | Sub-role, verification-gated cascade; thin-executor-default rejected |
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
| **localization-futility gate** | An early gate that routes budget away from doomed hypotheses (the "15/16 doomed" waste) before the patch loop; informs allocation, never suppresses a candidate without execution evidence. | Section 16.6 |
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
| Blackboard 2.0 (phased abstracted negatives) | [MEMOIR](https://arxiv.org/abs/2605.17539)/LTS; v1 EpisodicMemoryBus | adopt-modified | Abstracted negatives preserve diversity. |
| Model economy as verification-gated cascade | [Aider](https://aider.chat) architect/editor; [HyperAgent](https://arxiv.org/abs/2409.16299) | adopt-modified | Cheapen run/verify/narrow edits; frontier on navigation. |
| Open-pool active controller via capability profiles | [Puppeteer](https://arxiv.org/abs/2505.19591)/[MoMA](https://arxiv.org/html/2509.07571v1) | adopt-modified | The defensible contribution; staged; fail-open. |
| GEPA-style reflective prompt evolution of controller | [GEPA](https://arxiv.org/abs/2507.19457) | defer | Stage 1; 35x cheaper than RL; vendor-agnostic. |
| Full RL (Puppeteer/Conductor GRPO) over orchestrator | [Puppeteer](https://arxiv.org/abs/2505.19591) | defer | Stage 2; only when volume justifies. |
| Distributed/classical MCTS as the core loop | redesign | reject | Re-describes FrontierSearch; brittle vs container state. |
| Static-AST CTDG as a test-pruning gate | [PyCG](https://github.com/vitsalis/PyCG) ~70% recall | reject | Silently drops fault-revealing tests. |
| Cheap pre-execution plan scoring as a hard gate | redesign | reject | False-negative pruning violates Cardinal Contract. |
| Raw share-all / mid-subprocess injection blackboard | redesign | reject | -3.7pp accuracy; homogenizes; breaks replay. |
| Heavy-orchestrator + thin-executor as the default | [HyperAgent](https://arxiv.org/abs/2409.16299) ablation | reject | Cheapening navigation/multi-file editing hurts most. |
| Non-adaptive fixed-K default (5, up to the 16 cap), caps OFF | v1 default | reject | The headline cost pathology. |
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
| 8 | [MEMOIR / Lifelong/Long-Term-Sharing memory](https://arxiv.org/abs/2605.17539) | Abstracted negatives preserve diversity; share-all loses accuracy. Basis for Blackboard 2.0's abstracted-negative push (Section 11). |
| 9 | [SWE-Effi](https://arxiv.org/pdf/2509.09853) | Token/time-budget AUC metrics; token-snowball + expensive-failure pathologies from missing futility detection. Basis for token-yield metric + futility gate (Section 16). |
| 10 | [ImpossibleBench](https://arxiv.org/abs/2510.20270) | Reward-hacking scales with capability; agents exploit broken/satisfiable-by-cheating tests. Basis for anti-cheat + execution-authoritative selection (Section 13). |
| 11 | [Thinking Machines: Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | Temp-0 is not bitwise reproducible across hosted APIs (batch non-invariance). Basis for rejecting output-replay; reproduce artifacts (Section 15). |
| 12 | [Dissecting the SWE-Bench Leaderboards (arXiv 2506.17208)](https://arxiv.org/html/2506.17208v2) | No single architecture wins; localization + verification + ranking dominate; top systems ensemble multiple vendors (Devlo 70.2%, TRAE 70.4%). Basis for (vendor,model) diversity axis (Section 3, 13). |
| 13 | [Why SWE-bench Verified no longer measures frontier capabilities (OpenAI, Feb 2026)](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) + [SWE-bench Pro](https://arxiv.org/abs/2509.16941) | Verified is contaminated (59.4% flawed hard tests); use Pro/Live under standardized scaffold. Basis for the evaluation plan's benchmark choice (Section 20). |
| 14 | [Agent Client Protocol (ACP)](https://agentclientprotocol.com/get-started/introduction) / [A2A (Linux Foundation)](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) | `initialize` capability negotiation + Agent Cards are the portable templates for the Executor handshake / capability profile (Section 3). |
| 15 | [Live-SWE-agent (arXiv 2511.13646)](https://arxiv.org/abs/2511.13646) / [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) | Runtime tool synthesis from a minimal bash agent (~$0.73/issue, +22.6% for strong models). Basis for the minimal-worker-plus-orchestration shape (Section 8). |
| 16 | [xRouter (arXiv 2510.08439)](https://arxiv.org/html/2510.08439v1) / [RouteLLM](https://www.morphllm.com/llm-router) | Static routing trees are brittle and don't transfer; measure token yield not invoice; prefer cascade-with-verification. Basis for cascade-over-route (Section 12, 16). |
| 17 | [MoMA](https://arxiv.org/html/2509.07571v1) / [DAAO](https://arxiv.org/html/2509.11079v1) | Capability/cost profiles (not one-hot identity) enable routing to a growing heterogeneous pool. Basis for open-pool capability profiles (Section 14). |

Guarded claims (honor the adversarial posture): the open-pool cross-vendor controller "beats the single best model in the pool at matched cost on SWE-bench Pro" is **plausible but unproven** — it is framed as the experiment to run (Section 20), not a result. Classical MCTS, static-CTDG-as-gate, share-all blackboard, thin-executor-default, the non-adaptive fixed-K default, and output-replay are **rejected** and must not be reintroduced. Where a mechanism is `defer`, it ships behind a flag with a fail-open path to the prior stage.
