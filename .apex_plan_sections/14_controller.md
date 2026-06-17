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
