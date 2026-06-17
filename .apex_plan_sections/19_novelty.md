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
is the **worker representation**. Every published learned orchestrator
(Puppeteer, AFlow, MaAS, AgentConductor, AOrchestra) trains and tests on a
*fixed, known* pool and represents each agent by an *identity* (one-hot). APEX-Ω
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
