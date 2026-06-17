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
