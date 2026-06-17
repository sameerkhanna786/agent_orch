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
| Default full-cap 16 redundant trajectories, caps OFF | **reject** | The headline cost pathology; replaced by adaptive low-K + budget-aware deepening, full-cap kept only as the thin-feedback floor. |
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

v1's defaults are tuned for "SOTA, never for cost": `enable_adaptive_allocation=False` runs the full `max_rollouts` (default 16) regardless of difficulty, `repo_token_cap=None`, no wall-clock kill. APEX-Ω drives wall-clock and cost down *without lowering the solve ceiling* through composed levers, every one of which respects progress-based liveness (never a flat wall-clock kill of a working agent — v1's S1-S7 watchdog discipline is preserved):

- **Difficulty-adaptive low-K allocation (default ON).** Compute-optimal K is often < 10 ([Limits of Resampling](https://arxiv.org/abs/2411.17501); CodeMonkeys uses 10 trajectories). v1 already *has* `enable_adaptive_allocation` and `compute_rollout_count` over buckets `[1,4,8,16]` — they are simply OFF. Flipping them on is the single biggest cost lever (Section 16).
- **`pipeline()` per-item streaming.** No barrier between reproduce → localize → patch → verify stages; item A can be patching while item B is still reproducing. Wall-clock collapses from sum-of-slowest-per-stage to slowest-single-chain (Section 2, 16). This is the largest net-new build — v1 has only barrier waves.
- **Early localization-futility gate.** Kills the "15/16 doomed" patch-loop waste by routing budget to surviving hypotheses *before* the expensive patch attempts; informs allocation, never suppresses a candidate without execution evidence (Section 10).
- **Prefix-stable prompt assembly + provider-cache adapter.** Makes branching's constant-factor saving real (~90% off cached reads) and gives a portable cost contract across opaque vendor CLIs (Section 12, 16).
- **CTDG/test-impact-pruned verification** to bound the O(N²) cross-validation matrix (v1's dominant per-task verification cost), with a full-suite backstop (Section 10, 16).
- **Futility / token-snowball detection** and warm CoW worktree pools (v1's `WorktreePool`, ~10× cheaper recycling).

### 1.5 The Novelty Story: Open-Pool Cross-Vendor Control

The marketable headline — "an RL controller that prunes/sequences agents for cost-quality" — is **already published** ([Puppeteer, NeurIPS 2025](https://arxiv.org/abs/2505.19591); AgentConductor; AFlow; AOrchestra). Re-proposing it is derivative; it must be a *baseline we beat*, not the contribution. Every published learned orchestrator trains and tests on a **fixed, known** agent/model pool.

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
