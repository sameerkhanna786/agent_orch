## 11. The Epistemic Blackboard 2.0

### 11.1 Purpose, Scope, and the One Load-Bearing Claim

The Epistemic Blackboard 2.0 is the cross-rollout knowledge substrate of APEX-Ω: the workflow pattern by which independent coding WORKERS (Codex, Claude Code, or mixed — see Section 3) learn from each other's *dead ends and execution-grounded facts* without learning each other's *solutions*. It is the v1 `EpisodicMemoryBus` (see Section 4; `apex/rollout/engine.py`) promoted from a pull-at-stage-boundary append-only store into a **phased, selectively-admitted, push-at-turn-boundary** delivery layer — and nothing more aggressive than that, because the evidence ceiling is sharp.

The single load-bearing claim, and the one the adversarial verdict rated only `sound_with_caveats`:

> Selective, abstracted, *negatively-framed* cross-branch sharing, layered on top of diverse independent rollouts and *phased* so the first exploratory burst stays isolated, beats both (a) the no-sharing parallel baseline and (b) naive share-all — but the word "real-time" is unproven for independent rollouts and the word "sharing" must be narrowed to this specific mechanism.

Everything below is engineered to capture exactly the part the evidence supports and to refuse the part it does not. Three numbers anchor the design. Naive share-all *dropped* accuracy up to 3.7pp below a no-memory parallel baseline ([LTS, arXiv:2602.05965](https://arxiv.org/abs/2602.05965), Table 2). MEMOIR's two-level abstracted/negative sharing raised validity to 96.7% (+9.2pp) and cut run-to-run variance >10x ([MEMOIR, arXiv:2605.17539](https://arxiv.org/html/2605.17539)). A learned ~85%-admit controller beat no-memory by +1.2–5.6pp while cutting runtime 25–55% at ~0.2% controller overhead (LTS). The mechanism dominates the decision to parallelize at all: independent-parallel-no-communication was the *weakest* MAS variant (0.370 mean, 17.2x trace-level error amplification) in [MAST, arXiv:2503.13657](https://arxiv.org/abs/2503.13657).

#### 11.1.1 Three inviolable invariants (the brakes)

These are non-negotiable and trace directly to APEX's foundational frame and v1's Cardinal Safety Contract (Section 13).

1. **Producer-only scope.** The blackboard feeds *generation* only. It MUST NEVER feed the execution-grounded selector, the EG-critic, the VerificationAmplifier, or the FinalAcceptanceReviewer. A verifier that shares producer context "becomes another participant in collective delusion rather than an objective validator" (MAST). This is enforced structurally, not by convention (Section 11.7).
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
| Hallucinated shared facts | MAST collective delusion | Critic role; require corroboration/execution grounding before promotion |
| Verifier contamination | MAST "judge as participant" | Producer-only scope enforced structurally + CI guard (11.7) |
| Token blow-up of shared space | LbMAS 4.7M→13.9M | Mandatory cleaner; pool cap; per-entry token budget; dedup |
| Gains unrealizable without good verifier | CMU self-selection plateaus ~55%; [arXiv:2411.17501](https://arxiv.org/abs/2411.17501) optimal-N often <10 | Bus is a *generation* amplifier; execution-grounded selection (Section 13) is the ceiling; benchmark vs strong single-agent + verifier under matched compute |
| "Real-time" overclaim | MEMOIR/LTS are phased, not real-time | Renamed to phased streaming; cross-branch real-time NOT built; Hogwild scoped intra-rollout only |
| Mid-subprocess injection infeasible | opaque CLI workers | Turn-boundary delivery only (Inv. 3) |

**Net disposition (consistent with the Fusion Ledger, Section 18):** *adopt-modified.* The valuable, evidence-backed core — phased, selective, abstracted, negative-constraint sharing on diverse independent rollouts, producer-only, turn-boundary delivery, with relevance/confidence/dedup/own-rollout-exclusion preserved from v1 — is built. The parts the evidence does not support — share-all, positive-solution broadcast, real-time cross-branch coupling, mid-subprocess injection, and any path from the bus to the selector — are explicitly rejected and structurally prevented. Best-of-N with execution-grounded selection remains the floor we can never do worse than; the blackboard is a bounded amplifier on top of it.
