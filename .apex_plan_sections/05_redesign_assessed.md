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

**What we keep — bounded adaptive-branching (disposition: adopt-modified).** The genuinely-winning part of the search literature is *adaptive allocation*, not MCTS rollouts/backtracking. [AB-MCTS / TreeQuest](https://arxiv.org/abs/2503.04412) generalizes best-of-N by deciding **go-wider vs go-deeper** per fan-out point and beats both repeated sampling and standard MCTS — but **only above ~64 calls**; under strict small budgets naive AB-MCTS-M can [lose to plain repeated sampling (BAVT)](https://arxiv.org/abs/2603.12634). The disposition is therefore:

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

**Verdict: sound_with_caveats (medium confidence).** "Improves over isolated rollouts" is well-supported — isolation is itself a bad baseline ([MAST](https://arxiv.org/abs/2503.13657): independent-parallel-no-comms is the *weakest* variant, 0.370 mean, 17.2× error amplification). But "without collapsing diversity" is true **only for a specific mechanism** and false for sharing in general: [naive share-all dropped accuracy up to 3.7pp on GAIA (LTS)](https://arxiv.org/abs/2602.05965); broadcasting raw trajectories "biases generation toward local patches rather than new designs" ([MEMOIR](https://arxiv.org/html/2605.17539)); pass@k gains "vanish when candidates are highly correlated" ([Monkeys](https://arxiv.org/abs/2407.21787)). And "real-time" is the weakest word — the two strongest supporting works (MEMOIR, LTS) are **phased/post-commit, not real-time**; [DReaMAD](https://arxiv.org/abs/2503.16814) shows early/shared context amplifies the dominant initial belief regardless of correctness. The only true real-time data point, [Hogwild!](https://arxiv.org/abs/2504.06261), is for tightly-coupled subtasks *within* one rollout, not across independent rollouts.

**Reject: raw share-all / instant mid-subprocess push.** Mid-subprocess prompt mutation is infeasible against opaque CLIs and breaks determinism/replay (each `agent()` call's inputs would become nondeterministic, defeating the durable journal). Share-all measurably lowers accuracy and homogenizes attempts.

**Adopt-modified: phased, abstracted, negative-constraint sharing — an evolution of `EpisodicMemoryBus`, not a rebuild.** This is the largest under-cited overlap: v1's `EpisodicMemoryBus` *already* shares negative/ruled-out discoveries with relevance ranking, confidence floors, dedup-by-signature, and own-rollout exclusion. The redesign's only real delta is the delivery *schedule*. So:

1. **Diversity-by-construction at spawn** — per-rollout unique prompts/personas, heterogeneous (vendor, model), independent seeds (DReaMAD perspective diversity).
2. **Phase the bus, not always-on.** Keep the first exploratory wave fully isolated (v1 already runs barrier waves); open the epistemic layer only **after** rollouts commit to distinct strategies. This single change reconciles "improves outcomes" with "preserves diversity."
3. **Two-tier memory (MEMOIR).** Private full traces per rollout (already v1's worktree+`RepoContext` discipline); a thin global layer of ~200–300-token abstracted entries — verified codebase facts, failure modes, and **negative/avoidance directives** ("do NOT retry X; it deadlocks test Z"). Never broadcast raw solution trajectories. Implement as a generalization of `EpisodicMemoryBus` (append-only, relevance-ranked, self-excluding, **artifact-backed** so it stays filesystem-as-source-of-truth and resume-deterministic).
4. **Selective admission controller (LTS)** — ~85% admit, not 100%; admit only broadly-applicable cross-rollout facts. Add blackboard roles ([LbMAS](https://arxiv.org/abs/2507.01701)): a `cleaner` (prune stale/contradicted facts — proven to control token blowup), a `conflict_resolver` (reconcile contradictory facts about the same code object), a `critic` (catch hallucinated facts pre-propagation).
5. **Strict producer-only scope (the load-bearing guardrail).** The shared epistemic layer feeds **only generation**; it must **never** touch the execution-grounded selector / EG-critic / FinalAcceptanceReviewer, or per [MAST](https://arxiv.org/abs/2503.13657) the verifier "becomes another participant in collective delusion." This preserves the Cardinal Safety Contract.
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

**Verdict: partially_sound (high confidence).** The role-split *spine* is well-supported: [Aider architect/editor](https://aider.chat/2024/09/26/architect.html) improved **every** tested model over its solo baseline (o1-preview 79.7%→85.0%; o1-mini 61.1%→71.4%) and hit polyglot SOTA at ~14× lower cost; [Claude Code orchestrator patterns](https://www.mindstudio.ai/blog/smart-orchestrator-cheaper-sub-agent-models-claude-code) report 5–10× cuts on bounded tasks. But every such win is on **bounded/editing** benchmarks with a **competent (Sonnet-class) editor**, not a "thin cheap" executor on hard repo SWE. The redesign's own scope word — *hard repo SWE* — is the documented failure case: the [HyperAgent ablation](https://arxiv.org/html/2409.16299v1) shows weakening the **Navigator** (codebase exploration) or **Editor** roles causes the *worst* resolve-rate drops, while only run/verify is safely substitutable; [<13B models score <5% on SWE-bench Verified](https://benchmarkingagents.com/swe-bench/). The almost-right trap means a thin executor needing 3–4 rewrites can cost more than one frontier pass, and [SWE-Effi documents token-snowball "expensive failures" (4×+)](https://arxiv.org/pdf/2509.09853).

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
