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
| R6 | Default full-cap 16 redundant full trajectories (caps OFF) | v1 default | the headline cost pathology | Replace with adaptive low-K + budget-aware deepening; full-cap kept **only** as the thin-feedback floor. | A12 (adaptive low-K) + M1 (deepening) |
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

The verdict on cross-branch sharing is `sound_with_caveats`; the *specific* rejected forms are (a) **unconditional share-all** and (b) **real-time mid-subprocess push**. Share-all measurably *lowers* accuracy (LTS Table 2: −3.7pp on GAIA) and homogenizes attempts, destroying the diversity that makes sampling work (pass@k gains "vanish when candidates are highly correlated"). Real-time push is infeasible against opaque external CLIs (no mid-subprocess prompt-mutation channel) and breaks determinism/replay (each `agent()` call's inputs become non-deterministic). **Reject** the share-all and real-time forms. The diversity-preserving mechanism — **phased**, **selective**, **abstracted**, **negative-constraint** sharing at turn boundaries, evolving v1's `EpisodicMemoryBus` (which already shares negative discoveries, relevance-ranks, dedups, and excludes the caller's own `rollout_id`) — is **M5**, with the strict guardrail that the verifier never sees producer context (anti-collective-delusion, MAST).

#### 18.3.5 R5 — Heavy-orchestrator + thin executor as the default shape (verdict: partially_sound → default rejected)

The verdict is `partially_sound`: the role-split *spine* is supported (Aider architect/editor improved every model, ~14× cheaper), but the claim's own scope word — *hard repo SWE* with a *thin* executor — is the documented failure case. The HyperAgent ablation shows weakening the Navigator (codebase exploration) or Editor roles causes the **worst** resolve-rate drops, and <13B models score <5% on SWE-bench Verified. The "almost-right trap" means a thin executor needing 3–4 retries costs *more* than one frontier pass. **The default thin-everywhere shape is Reject.** The verification-gated cascade — frontier planner + frontier reviewer + cheap models confined to run/verify and narrow well-specified edits, escalating on first verify-on-diff failure — is **M6**.

#### 18.3.6 R6 — Default full-cap 16 redundant full trajectories (the cost pathology)

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
| Default full-cap 16 redundant full trajectories (caps OFF) | reject | R6 |
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
8. **Section 16 (speed/cost)** must build A15 prefix-stable assembly and replace the full-cap-16 default (R6) with adaptive low-K (A12); full-cap 16 survives only as the thin-feedback floor.

Any downstream section that needs to deviate from one of these must first amend this ledger — updating both the prose disposition and the `accepted_mechanisms` cross-reference index (18.5) — with a fresh verdict or digest basis. That is the single mechanism by which the in/out list stays canonical.
