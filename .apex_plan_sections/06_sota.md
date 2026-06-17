## 6. State of the Art: Synthesis & Exploitable Gaps

This section distills the 2024–2026 literature on inference-time scaling, tree search, process/outcome verification, program analysis, fleet efficiency, learned orchestration, and the dynamic-workflow paradigm into a single load-bearing claim and a small set of engineering invariants that the rest of APEX-Ω is built on. The thesis is deliberately narrow and decisive: **selection (the verifier), not search topology, is the binding constraint on agentic coding capability; coverage is cheap and search is a bounded amplifier whose ROI depends on the verifier and on candidate diversity.** Everything downstream — the speculative tree-search layer (Section 9), CTDG pruning (Section 10), the blackboard (Section 11), the model economy (Section 12), the verifier (Section 13), the active controller (Section 14) — is justified or restrained by the synthesis here.

The reader should treat each subsection as a *design constraint with a regime condition*, never a universal win. The single most important meta-lesson from the corpus is that almost every headline number is regime-dependent (budget, verifier soundness, candidate correlation, repo scale, model strength), and the most common failure of a redesign is to import a technique that won in one regime into a regime where it loses or actively harms. Where a mechanism is genuinely unproven for our regime, it is carried as guarded/optional with a mitigation and a fallback to the verified floor.

### 6.1 The Eight Synthesized Findings (the binding constraints)

The table below is the executive map of the evidence. Each row names the finding, the load-bearing numbers, the regime in which it holds, and the APEX-Ω disposition (where it is realized — see the cross-referenced section). The dispositions are consistent with the canonical accepted-mechanisms list and with the adversarial verdicts.

| # | Finding | Load-bearing evidence | Regime condition (when it holds / breaks) | APEX-Ω disposition |
|---|---------|----------------------|-------------------------------------------|--------------------|
| F1 | **Selection/verifier is the binding constraint, not search topology** | Best@K trails Pass@K by ~11pt ([SWE-Gym](https://arxiv.org/abs/2412.21139): Best@16 32.0% vs Pass@16 42.8%); SWE-Search value-fn 73%→discriminator 84% ([ICLR 2025](https://arxiv.org/abs/2410.20285)); CodeMonkeys 45.8% random → 57.4% selected vs 69.8% coverage ceiling ([2501.14723](https://arxiv.org/abs/2501.14723)) | Always — strongest single result across math, code, repo-SWE. Gap *widens* as the verifier weakens | Cardinal Safety Contract + hybrid verifier (Sections 13); invest engineering in selection, cap search |
| F2 | **Repeated sampling scales coverage log-linearly, but optimal K is often <10 against imperfect verifiers** | Coverage c(k)≈exp(a·k^-b) over ~4 orders of magnitude; SWE-bench Lite 15.9%@1→56%@250 ([Large Language Monkeys](https://arxiv.org/abs/2407.21787)); but optimal K≤5 at cost-benefit 4, often <10 ([Limits of Resampling](https://arxiv.org/abs/2411.17501)) | Coverage scaling needs a *near-oracle* verifier to convert. With noisy verifiers, false-positive risk *rises* with K | Difficulty-adaptive low-K default ON (Section 9/14); high-K only as thin-feedback floor |
| F3 | **Adaptive-branching beats best-of-N and MCTS only above ~64 calls; budget-aware schedule matters** | [AB-MCTS](https://arxiv.org/abs/2503.04412): comparable below 64 calls, pulls ahead above ~64 (DeepSeek-V3 CodeContests); [BAVT](https://arxiv.org/abs/2603.12634): budget-agnostic AB-MCTS-M can *lose* to repeated sampling under strict budgets | Only large budgets + rich feedback. Below the crossover, plain verified sampling wins | Bounded adaptive branching inside FrontierSearch caps; collapse to verified best-of-N below feedback-confidence floor (Section 9) |
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

**F6 (model economy).** The split is verification-gated, not a blanket "cheap orchestrator." Aider/opusplan show 5–14× savings with a *competent* editor, and RouteLLM/FrugalGPT-style cascades save 30–90% — but only on the *right* axis. HyperAgent's ablation is the load-bearing warning: cheapening navigation and multi-file editing causes the worst resolve-rate drops, the "almost-right trap" that can cost more than one frontier pass. So APEX-Ω cheapens run/verify and narrow edits, keeps the frontier model on navigation/multi-file edits, and escalates on first verify-on-diff failure with a rewrite-cycle cap. The **heavy-orchestrator + thin-executor default is rejected** (verdict partially_sound → default rejected). Cost-saving primitives that *are* portable and adopted: prefix-stable prompt assembly + provider-cache adapter (~90% off cached reads; selective caching beats full-context, which can *raise* latency and regresses 10–18% TTFT below the min-token threshold — [Don't Break the Cache](https://arxiv.org/html/2601.06007v2)); a vendor-agnostic API consumer cannot literally share KV across forks, only maximize prefix hit-rate via byte-identical prefixes + dispatch ordering (KVFlow steps-to-execution scheduling over the workflow graph).

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
| Default full-cap 16 redundant trajectories | F2/F8: optimal K<10; cost pathology; expensive failures | difficulty-adaptive low-K default ON; full-cap = thin-feedback floor only |
| Plain MCTS / FrontierSearch as core loop | F3: re-describes controller; brittle vs non-serializable state; loses below 64 calls | bounded adaptive branching inside budget caps; collapse to verified BoN |
| Raw share-all / mid-subprocess injection | F7: −3.7pp, homogenizes; infeasible vs opaque CLIs; breaks replay | abstracted negatives at turn boundaries only |
| Cheapening navigation/multi-file editing | F6: HyperAgent worst-drop ablation; almost-right trap | frontier on navigation; cheapen run/verify/narrow-edit only |
| Trusting bit-reproducible agent OUTPUT replay | impossible across hosted APIs (temp-0 batch non-invariance) | reproduce *artifacts* (diffs + re-run verification), not token streams |
| Trusting any static public benchmark >12–18mo | F8: SWE-bench Verified contaminated/deprecated | private rotating eval; standardized scaffold on uncontaminated splits |
| Caching below the min-token threshold | 10–18% TTFT regression; full-context caching raises latency | selective prefix-stable caching only |
| Hedging/speculative fan-out without a circuit breaker | tail-at-scale: degraded backend → every request crosses p95 → doubled load | deadline-triggered dispatch gated by circuit breaker + budget kill-switch |

### 6.9 Net synthesis (the load-bearing claim, restated)

The honest framing the whole plan rests on: **search and economy are bounded amplifiers; execution evidence is both the steering signal and the brake; verified best-of-N is the floor we can never do worse than.** No single technique is a universal win — every adopted mechanism carries its regime condition and a mandatory collapse to the verified floor when that condition is not met. The substrate that makes amplification *safe* is APEX v1's execution-authoritative kernel (Cardinal Contract, cheap-first cascade, worktree isolation, anti-cheat, determinism/RunManifest); the redesign mechanisms are admitted only as workflow-pattern extensions over vendor-neutral workers, only where the verdicts show net-positive ROI, and only in forms that respect that contract. The one clearly unclaimed, defensible contribution — **open-pool generalization of the active controller** — is the scientific stake (Section 19), built on the durable journaling substrate that doubles as its off-policy credit ledger.

Cross-references: the verifier and selection mechanics are specified in Section 13; the bounded search layer in Section 9; CTDG/pruning in Section 10; the blackboard in Section 11; the model economy in Section 12; the active controller and learned policy in Section 14; isolation/determinism/durable resume in Section 15; speed/cost engineering in Section 16; self-improvement/memory in Section 17; and the contamination-resistant evaluation in Section 20.
