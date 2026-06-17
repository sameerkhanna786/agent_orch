# APEX_NEXTGEN_PLAN.md — Critical-Review Response

Date: 2026-06-14. This documents how `APEX_NEXTGEN_CRITICAL_REVIEW.md` (no-go-as-written) was addressed in `APEX_NEXTGEN_PLAN.md`. The revisions live in the plan's new **§24 (Critical-Review Response & Revisions, 7 subsections, ~15k words)** plus 26 in-place edits to the existing sections.

## Method: verify the review's own premises first

The review was treated as a hypothesis, not ground truth. A 21-agent workflow verified its load-bearing factual premises before any edit, defaulting to skepticism (arXiv direct-fetch for papers; file:line for v1 source; the real run artifacts for test history):

| Premise | Verdict | Outcome |
|---|---|---|
| **MACA = arXiv:2605.25746 (real)** | **confirmed_real** | The review was right; my earlier pass was wrong (see below). Added as prior art + baseline B5 + ablation A12. |
| Routing prior art (AOrchestra, Router-R1, DAAO, RouteLLM) | partially_real | Confirmed real; characterized precisely (what each does and does *not* cover). |
| Other prior art (AB-MCTS, AFlow, AgentConductor, R2E-Gym, OpenHands, SWE-agent, Agentless, HyperAgent, PatchPilot, BOAD, AgentForge) | partially_real | Confirmed; **DeLM could NOT be verified → excluded, not cited.** |
| #3 v1 kernel boundaries (no Executor/journal/pipeline; replay exact-only; routing static; search default-off) | **code_confirmed** | Confirmed against source → "staged extraction," not "preserve kernel." |
| #8 selector ranking tuple | partially_real | The review is right: `_best_cluster_by_deterministic_ranking` sorts by `combined_score` (soft) first. The false claim is deleted and the invariant restated. |
| #4 test/benchmark history | mixed | "active hardening," not "hardened substrate" — confirmed; concrete stabilization gates added. |
| #5 Claude attribution | confirmed_real | The 5 primitives / determinism / "ultracode" are APEX extensions, not Claude guarantees — attribution corrected. |

### The MACA correction (stated plainly)

In an earlier turn I reported MACA as "not a real, citable system." **That was wrong.** Direct WebFetch of `arxiv.org/abs/2605.25746` and its HTML confirms MACA ("Multi-Agent Coordination Adaptation via Structure-Guided Orchestration"; GraphSpec structural prior + GRPO; +8.42% acc / 43.19% fewer tokens vs DyLAN/AgentVerse/MacNet/AgentPrune/MaAS/Puppeteer) is real. The earlier failure was a method error — abstract/web-search lookup that missed body-only terms. The verification protocol is corrected to fetch the canonical arXiv URL directly, and the plan (§24.2) records this discrepancy openly. MACA does *not* sink APEX's contribution (it is fixed-pool, QA/math/function-code, no execution-authoritative patch gate), but it does own "learned structure-guided budget-aware orchestration," so APEX no longer claims that.

## Narrowed novelty (the single defensible claim)

> APEX-Ω's single defensible contribution is the instantiation of no-retrain, held-out-vendor search/route-policy generalization under execution-authoritative, repo-level SWE patch acceptance with cost-matched budgets. Stated as a falsifiable thesis: a learned controller that, via learned capability/cost profile vectors (never one-hot vendor identity), routes to a vendor/model held out from training — with no controller retraining and no online updates on the held-out vendor's tasks — and retains cost-quality-Pareto dominance over the strongest single model, cost-equal verified best-of-N, a re-trained published learned orchestrator (Puppeteer/AFlow/MACA-style), an AOrchestra-style fixed-pool dynamic-subagent orchestrator, and a Router-R1/RouteLLM-style descriptor-routing baseline, while every accept stays gated exclusively by the execution-grounded acceptance bar. This is explicitly NOT a claim that no-retrain held-out routing is itself novel (Router-R1 and RouteLLM already do descriptor-conditioned no-retrain routing to unseen models — but only on QA, with no execution gate), and NOT a claim that learned budget-aware/structure-guided orchestration is novel (MACA/Puppeteer/AgentConductor already establish that — but on QA/math/function-or-competition code over a fixed pool, with no execution-authoritative patch gate). The invention is precisely the unoccupied PRODUCT of the two axes: AXIS-A (no-retrain held-out routing, owned for QA by Router-R1/RouteLLM) × AXIS-B (repo-level SWE + execution-authoritative acceptance + vendor-mixed pool, owned over a fixed pool by AOrchestra) — a cell no single paper occupies, evaluated under a pre-registered held-out-vendor protocol with separate training/calibration/inference cost accounting.

## Finding → resolution map

## Response Map: Review Findings + Required Changes -> Resolution

### Findings 1-10

| Finding | Resolution |
|---|---|
| **#1 Missing MACA (major novelty gap)** | §24.2 (MACA added as real prior art in collision map, exact ID, "does NOT cover" column); §24.3 B5 baseline + A12 GraphSpec-like comparator with training/data-cost axis; novelty reframed in §24.1. In-place: E1 (§6.7 two-axis decomposition adds MACA), E2 (§19.1 add MACA + AOrchestra rows). |
| **#2 Novelty narrower than claimed** | §24.2 component-collision map presents branching/routing/search/verification as integrated prior art (5 families, exact IDs, owns/does-NOT-own columns) + two-axis thesis. In-place: E1 (§6.7), E3 (§19.2 AOrchestra sharpened to "does NOT"), E2 (§19.1). |
| **#3 v1 not a clean stable kernel** | §24.5 staged-extraction sequence (wrap->journal->extract->pipeline-behind-flag->kill/resume->shadow-then-live routing). In-place: E4 (§4 "actively-hardening"), E5 (§8.2 staged extraction), E6 (§13 verification substrate), E7 (§15 substrate framing), E8 (§22 roadmap). |
| **#4 Test/benchmark history vs "hardened substrate"** | §24.6 pre-implementation stabilization gates G1-G9 (each tied to verified failure; lastfailed down-scoped to ~6 orchestrator tests). In-place: E4/E6/E7/E8 reframe "hardened"->"actively-hardening" across §4/§13/§15/§22; E9 (§1 exec thesis); E10 (§18 ledger legend). |
| **#5 Claude attribution too loose** | §24.8 attribution-fix + Claude-leaf containment rules. In-place: E11 (title line drop "ultracode"), E12 (§Exec-Summary line 33), E13 (§2 line 144), E14 (§2.2 primitives are APEX-defined). |
| **#6 Eval plan incomplete (baselines)** | §24.3 baseline/ablation matrix B5-B8 + A12; §24.4 held-out-vendor protocol with required baselines AOrchestra + descriptor-routing. In-place: E15 (§20.2 add B5-B8), E16 (§A10/A9 bind to protocol). |
| **#7 Open-pool held-out-vendor underspecified** | §24.4 pre-registered protocol + capability-profile construction rules + per-decision logging + OOD diagnostics. In-place: E17 (§14.10 eval row -> protocol pointer). |
| **#8 Verification invariant overstated** | §24.7 restated two-part contract (acceptance gate + ranking scope) + 3 property tests + trace assertion. In-place: E18 (§13.1 DELETE false tuple claim), E19 (§14.2 invariant #1 re-anchor). |
| **#9 pipeline() largest engineering risk** | §24.5 (22.2.2.a) seven explicit pipeline contracts; staged behind scaffolded/local flag before provider-backed. In-place: E20 (§2.2/§16/§9 soften "largest net-new build"; add seven-contract pointer). |
| **#10 Replay/resume = two products** | §24.7-journal subsection: exact replay kept verbatim + artifact replay added; kill/resume CI smoke = §15.5 reworded to "same terminal dispatch set + accepted artifact verdict". In-place: E21 (§15.3/§15.5 two-products framing), E22 (§3.7 artifact-replay clarified as additive). |

### Required Changes 1-10

| Required change | Resolution |
|---|---|
| **1. Add MACA/AOrchestra/Router-R1/DAAO/RouteLLM/AFlow/AB-MCTS/AgentConductor/BOAD/AgentForge/R2E-Gym/(DeLM) to related work + baselines** | §24.2 collision map (exact IDs, DeLM explicitly dropped as unverifiable); §24.3 baseline families B5-B8. |
| **2. Rewrite novelty around narrow intersection** | §24.1 narrowed-novelty statement (two-axis product); in-place E1/E3. |
| **3. Pre-implementation stabilization phase** | §24.6 gates G1-G9 (selector/verifier invariants, provider transport, Docker/no-network, worktree lifecycle, finalization/drain, local-validation trust, replay, kill/resume). |
| **4. Replace "preserve v1 kernel" with "staged extraction"** | §24.5 staged-extraction sequence; in-place E4/E5/E6/E7/E8. |
| **5. Define workflow journal schema (full field set)** | §24.7 durable-journal record with every prescribed field (run/node/parent ids, stage, input/prompt/profile hashes, resolved model id, CLI version, tool permissions, output artifact paths, diff pointer, verifier status, terminal state, cost/tokens/latency). In-place E21 (§2.5 defers to §24.7 as authoritative). |
| **6. Native Claude workflow containment explicit** | §24.8 Claude-leaf containment Rules 1-5 (disableWorkflows; no ultracode/trigger phrasing; noninteractive-safe permissions; log version+model; human-judgment-as-boundaries). |
| **7. Pre-register held-out-vendor protocol + capability-profile rules** | §24.4 frozen split + leakage controls + per-decision log + OOD diagnostics + profile-construction rules. |
| **8. Executable benchmark fairness CI gates** | §24.6 fairness CI gates F1-F5 wired to apex/core/fairness_audit.py (honest "scorer wiring is Phase-1 TODO" caveat). |
| **9. CTDG first as telemetry + test prioritization, not a gate** | §24.7-CTDG subsection: build-order mandate (telemetry first, prune gated, never on acceptance path). In-place E23 (§10 add build-order + re-anchor to corrected invariant). |
| **10. Fallback paper positions** | §24.1 fallback-paper-positions (held-out works -> main claim; durable orchestration works -> systems paper; only v1 stabilization works -> engineering report). |

## Residual notes

- An internal contradiction introduced during revision (the body still repeated the deleted "ranking tuple places soft below execution" claim in ~9 places) was reconciled to the accurate invariant + §24.7.3 pointers.
- The plan is a *design synthesis*; the held-out-vendor result (§24.4) remains the central, unproven research bet, framed with fallback paper positions (§24.1).
- Backup of the pre-revision plan: `APEX_NEXTGEN_PLAN.bak.md`.
