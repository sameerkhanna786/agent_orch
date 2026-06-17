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

AB-MCTS only beats repeated sampling **above ~64 calls** ([Inoue et al.](https://arxiv.org/abs/2503.04412)); under strict small budgets, naive AB-MCTS-M can *lose* to plain repeated sampling ([BAVT](https://arxiv.org/abs/2603.12634)). APEX therefore conditions explore/exploit on `budget.remaining()` and, critically, defaults to **adaptive low-K best-of-N** (Section 12, the canonical "Difficulty-adaptive low-K allocation, default ON"). The adaptive-branching layer only *activates beyond a budget threshold* (`search.activation_min_nodes`, default 8). Below it, we are pure best-of-N — the regime where search has no proven edge. This is the single most important honesty constraint in the section.

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

All keys live under `search.*` in `ApexConfig`. Defaults are conservative (best-of-N-leaning) per the canonical "default full-cap 16 redundant trajectories — REJECT" and "adaptive low-K — default ON."

| Key | Type | Default | Meaning |
|---|---|---|---|
| `search.enabled` | bool | `true` | Master switch; `false` == pure adaptive low-K best-of-N (Section 12) |
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
