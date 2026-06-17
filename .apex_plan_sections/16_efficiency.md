## 16. Speed & Cost Engineering

> **Frame.** Speed and cost are not a separate subsystem; they are properties of how the workflow engine (Section 2) schedules `agent`/`parallel`/`pipeline`/`phase`/`budget` primitives over vendor-neutral workers. This section specifies the cost/latency contract APEX-Ω enforces on top of APEX v1's "optimize for SOTA, never for cost" substrate. The governing rule throughout: **speed comes from not *starting* doomed work, never from *killing* working agents** (v1's progress-based liveness, Section 15, is preserved verbatim). Every cost lever here is a *bounded amplifier* of the three capability properties in the Central Thesis (isolation, verify-and-refute, orchestration-as-code) — it may reduce wasted spend, but it may **never** abort an in-flight succeeding rollout (the escrow-WAL/CCEDF invariant, Section 15) and may **never** promote an unverified candidate to make a deadline (the Cardinal Safety Contract, Section 13).

APEX v1 is, by design, the worst-case cost profile: `enable_adaptive_allocation=False` runs the **full cap of 16 redundant full trajectories** regardless of difficulty; `repo_token_cap=None` and `max_tokens_per_repo_followup=0` leave all token-budget machinery inert; the default backend is `claude-opus-4-8[1m]` at `--effort max` for *every* stage. The remarkable fact — established in the v1 ingest's `change_seams` — is that v1 already *built* nearly every cost-control hook (adaptive allocation, token caps, worktree pool, model-id indirection) and left them **OFF**. APEX-Ω's job is therefore mostly to **flip and wire** these hooks behind a quality SLA, plus add the genuinely new primitives (per-item `pipeline()` streaming, the provider-cache adapter, prefix-stability linting). This is a "do better than the reference impl" mandate executed by configuration and scheduling discipline, not a rewrite.

### 16.1 The Eight Cost Layers and Where the Money Goes

v1's cost stacks multiplicatively across four generation/selection layers plus orchestration overhead. APEX-Ω attacks each with a specific, bounded lever:

| # | Cost layer (v1) | Dominant driver | APEX-Ω lever | Adopted mechanism / disposition |
|---|---|---|---|---|
| 1 | Full-trajectory rollouts (up to 16) | redundant full solves | **Difficulty-adaptive low-K** (default ON) | adopt: "single biggest cost lever" |
| 2 | Per-rollout agent loop length / context | many turns × 80–120k ctx | prefix-stable assembly + provider cache | adopt |
| 3 | Per-rollout setup (worktree create+warmup ~4s) | duplicated provisioning | **warm CoW worktree pool (~10×)** | adopt (v1 `use_worktree_pool`) |
| 4 | N×N cross-validation matrix | O(N²) sandboxed test runs | **clustering-before-matrix + test-impact prune** | adopt-modified (CTDG) |
| 5 | Selection verification cascade | baseline + regression reruns | cheap-first cascade (unchanged) | adopt verbatim (Section 13) |
| 6 | Escalation / progressive waves / 4 follow-up loops | "spend more on partial progress" | **futility gate + budget kill-switch** | adopt |
| 7 | Optional LLM selection layers (voters/amplifier/reviewers) | contested-tie LLM calls | cost-bounded mode disables, deterministic fallback | adopt (already gated) |
| 8 | Subprocess retry/salvage (≤4 attempts) | flaky infra | unchanged (happy path pays ~0) | keep v1 |

**Wall-clock**, separately, is dominated by the *longest single rollout* plus serial selection. v1 runs 16 rollouts at `parallel_workers=3` ≈ 6 sequential batches × slowest-rollout, then serial O(N²) cross-validation. The two biggest wall-clock wins are (a) lowering K so fewer batches run, and (b) `pipeline()` streaming so a stage-2 worker starts the moment a *single* item clears stage-1, rather than waiting for the slowest stage-1 worker (Section 16.3).

### 16.2 Difficulty-Adaptive Low-K Allocation (default ON)

This is the headline change versus v1. v1's adaptive path is **fully wired but disabled** (`estimate_difficulty → compute_rollout_count → evaluate_policy_model('planning.rollout_count') → _clamp_rollout_bucket`); APEX-Ω makes it the default and binds it to a quality SLA rather than full-cap. Evidence: Snell et al. compute-optimal scaling shows 2–4× compute savings at matched quality, and the optimal number of independent attempts K is **often < 10** even on hard tasks; pure best-of-N coverage scales only log-linearly (Large Language Monkeys, [arXiv:2407.21787](https://arxiv.org/abs/2407.21787)) so the 9th–16th trajectory buys almost nothing on most issues. Self-consistency's edge is *diminishing* on modern models ([arXiv:2511.00751](https://arxiv.org/html/2511.00751): 0.4–1.6% over 20 samples).

```python
# allocation.py — runs once per task, after Phase-2 localization (amortized)
RolloutBuckets = [1, 4, 8, 16]   # v1 quantization, kept

def select_K(task, ctx, cfg) -> AllocationDecision:
    d = estimate_difficulty(task, ctx)          # 0..1; reuse v1 estimate_difficulty
    # difficulty -> bucket thresholds (v1 Section 5.2)
    if   d <= 0.25: k = RolloutBuckets[0]        # easy: 1 (+ speculative-first cheap attempt)
    elif d <= 0.55: k = RolloutBuckets[1]        # 4
    elif d <= 0.80: k = RolloutBuckets[2]        # 8
    else:           k = RolloutBuckets[3]        # 16 (thin-feedback / hard floor)

    # Portfolio floor RAISES only (v1 invariant): guarantee >=1 of each distinct
    # (vendor, model) profile we want for diversity decorrelation (Section 3 / 12).
    k = max(k, portfolio_rollout_floor(cfg))     # never lowers below diversity floor

    # Below the feedback-confidence floor we cannot trust difficulty -> collapse to
    # the verified best-of-N FLOOR we can never do worse than (Central Thesis).
    if feedback_confidence(ctx) < cfg.feedback_floor:
        k = RolloutBuckets[3]                     # full-cap as the thin-signal backstop
    return AllocationDecision(k=k, difficulty=d, reason=...)
```

Config keys (all promoted from v1 seams):

```yaml
allocation:
  enable_adaptive_allocation: true        # v1 default False -> ON
  rollout_buckets: [1, 4, 8, 16]
  feedback_floor: 0.35                     # below this, collapse to full-cap
  portfolio:
    min_distinct_profiles: 2               # diversity floor (cross-vendor; Section 3)
    max_distinct_profiles: 6               # cap profile budget when cost-bounded
  quality_sla:                             # adaptive must hold these vs full-cap ablation
    max_resolve_rate_delta: 0.01           # gate adaptive ON only if within 1pt of full-cap
```

**Honesty / pitfall.** Snell's 2–4× is *matched-quality on math-reasoning regimes*; repo-SWE difficulty estimation is noisier. APEX-Ω therefore (1) keeps the **full-cap path as the thin-feedback floor** so a mis-estimated hard task degrades to v1 behavior, not to under-spend; (2) gates the *default-ON* decision on the `quality_sla` ablation in Section 20 (adaptive must land within 1pt resolve-rate of full-cap on a contamination-resistant split); and (3) keeps the portfolio floor *raise-only* so the diversity axis that decorrelates hallucinations (Devlo/TRAE cross-vendor, Section 3) is never sacrificed for cost. The early-localization-futility gate (Section 16.6) further routes the freed budget toward *surviving* hypotheses rather than blindly cutting K.

### 16.3 `pipeline()` Per-Item Streaming — the One Net-New Primitive

v1 runs the rollout as a hard-coded staged trajectory (reproduce → localize → patch → verify) but materializes each stage as a *batch* across rollouts: stage N+1 waits for the slowest worker of stage N. `pipeline()` is the genuinely new engine primitive (accepted as **adopt**): it streams *per item* so each item flows stage→stage independently.

The cost identity is the whole point:

```
batch-staged wall-clock   = Σ_stages  max_item( stage_latency )      # sum-of-slowest-per-stage
pipeline per-item streamed = max_item( Σ_stages item_stage_latency ) # slowest SINGLE chain
```

For a 4-stage chain where each stage's slowest item differs (the common case), this collapses wall-clock from the *sum of four per-stage maxima* to the *single slowest end-to-end chain*.

```python
# engine/pipeline.py — vendor-neutral; workers are opaque Executors (Section 3)
def pipeline(items, stages, *, workers, budget) -> list[Result]:
    """
    stages: ordered list of Stage(name, fn, worker_profile, timeout_policy)
    Each item advances independently; a free worker pulls the next ready item
    from the earliest non-empty stage queue (longest-shared-prefix-first; 16.5).
    """
    queues = [Queue() for _ in stages]; queues[0].extend(items)
    results = []
    with WorkerPool(workers) as pool:
        while not all_drained(queues) and budget.alive():
            stage_idx, item = pick_ready(queues, policy=LONGEST_SHARED_PREFIX_FIRST)
            stage = stages[stage_idx]
            fut = pool.submit(stage.fn, item, profile=stage.worker_profile)
            on_done(fut, lambda r:
                (queues[stage_idx+1].put(r) if stage_idx+1 < len(stages)
                 else results.append(r)))
            journal.append(StageEvent(item.id, stage.name, attempt=item.attempt))  # 15.x WAL
    return results
```

Per-item streaming composes with everything else: a fast-localized item can be *patching* while a slow item is still *reproducing*, and the patcher worker can be a *cheaper* tier than the navigator (Section 12 model economy). Crucially, `pipeline()` changes only *scheduling order*, never *acceptance*: every streamed item still terminates in the unchanged cheap-first verification cascade, so streaming cannot leak an unverified pass.

### 16.4 Prefix-Stability Discipline (portable across all providers)

The single highest-leverage, fully-portable rule: **structure every worker prompt so the largest possible prefix is byte-identical across forks and turns**, then schedule to hit hot caches. This is what makes branching's constant-factor saving real (the adversarial verdict on speculative branching is explicit: the win "comes from prefix/KV-cache reuse, not the tree per se," and is *bounded constant*, not exponential — [Tree-GRPO arXiv:2509.21240](https://arxiv.org/abs/2509.21240), [RadixAttention arXiv:2312.07104](https://arxiv.org/pdf/2312.07104)).

**Prompt-assembly contract.** Every `agent()` template emits two regions:

```
[ STABLE  ] tooling defs + system prompt + policies + repo-invariant context
[ VOLATILE] task + scoped hypothesis + live context + per-rollout discovery_scope
```

**Lint rules (build-time + assembly-time, fail-loud).** The stable region is forbidden from containing anything that drifts. One byte of drift = total cache miss + write premium ([Don't Break the Cache, arXiv:2601.06007](https://arxiv.org/html/2601.06007v2)).

| Forbidden in STABLE region | Why | Enforcement |
|---|---|---|
| timestamps / dates / `now()` | drifts every call | regex lint, hard error |
| UUIDs / session IDs / `rollout_id` | per-rollout divergence | regex lint, hard error |
| dynamic tool definitions reordered | invalidates tools→system→messages | canonical sort of tool defs |
| summarized/pruned tool history | mutates mid-loop | summaries go in VOLATILE only |

**v1 conflict that must be resolved (from the adversarial verdict).** v1 *deliberately* injects per-rollout-unique state into prompts — air-gapped HOME, per-rollout lock paths, `discovery_scope` drawn from the `EpisodicMemoryBus` excluding the caller's own `rollout_id`, snapshot commit messages embedding `source_head8+dirty_hash8`. These are the *exact* anti-pattern for prefix caching. APEX-Ω resolves this by **physically separating** the stable shared-prefix region (identical across all rollouts of a task) from the per-rollout-divergent scoping, which is appended *strictly in the VOLATILE tail*. The blackboard's abstracted negative constraints (Section 11) likewise land in VOLATILE, never in the cached prefix.

#### 16.4.1 Provider-Cache Adapter (declare-stable-prefix API)

Because APEX-Ω is vendor-neutral it cannot assume any single provider's KV internals. The adapter exposes a uniform **`mark_cacheable(span)`** API compiled per-provider:

| Provider | Mechanism | Compiled behavior | Verified economics |
|---|---|---|---|
| Anthropic (Claude Code) | explicit `cache_control` breakpoints | ≤4 breakpoints at STABLE/VOLATILE boundary; 1h TTL (write 2.0×) for hot shared prefixes, else 5m (1.25×) | cache read **0.10× input (90% off)**; min 1,024 tok (Opus/Sonnet 4.x) |
| OpenAI (Codex) | auto-cache >1,024 tok, routes by first ~256-tok hash | keep STABLE prefix byte-stable; pin `prompt_cache_key` per task | ~50% off cached tokens |
| Gemini CLI | auto-cache | identical stable-prefix discipline | provider-reported |
| self-hosted (SGLang/vLLM) | RadixAttention / block-hash | optional: target true cross-fork KV reuse | up to 5–6.4× throughput on prefix-heavy work |

```python
class ProviderCacheAdapter(Protocol):
    def mark_cacheable(self, prompt: AssembledPrompt) -> ProviderPayload: ...
    def read_cache_metrics(self, resp) -> CacheMetrics:
        # cache_read_tokens, cache_creation_tokens, uncached_tokens
        ...
```

**SLO + degrade-gracefully (mandatory pitfall mitigation).** *Do not assume server-side prefix caching is on.* APEX-Ω tracks `cache_read_tokens` vs `cache_creation_tokens` as a first-class fleet SLO per `(vendor, model)` and **detects whether caching actually fired**:

```yaml
cache_slo:
  min_cache_read_ratio: 0.50        # below this on a stable-prefix run -> WARN + investigate
  below_min_token_guard: 1024       # prompts under threshold: DO NOT request caching
  on_cache_miss: degrade            # cap fan-out, fall toward sequential; never assume the win
```

Two honesty caveats baked into the design: (1) caching **below the min-token threshold causes a 10–18% TTFT regression** ([arXiv:2601.06007](https://arxiv.org/html/2601.06007v2)) — the `below_min_token_guard` suppresses cache requests for short prompts; (2) a pure API consumer **cannot literally share KV tensors across forks** — against black-box provider caches APEX-Ω can only maximize *hit-rate* via byte-identical prefixes + dispatch ordering, and the docs/comments must never promise true cross-fork reuse on hosted APIs.

### 16.5 Dispatch Ordering: Longest-Shared-Prefix-First (KVFlow-style)

Given the prefix-stability contract, *the order in which forks are dispatched* determines provider-cache hit-rate even when APEX-Ω has zero control over the KV store. APEX-Ω schedules the workflow graph KVFlow-style ([arXiv:2507.07400](https://arxiv.org/html/2507.07400v1)): assign each pending branch a "steps-to-execution" priority and **dispatch longest-shared-prefix-first / depth-first** so a freshly-warmed prefix is immediately reused by sibling forks before it ages out of the provider's lookback window (Anthropic 20-block lookback; OpenAI first-~256-tok routing).

```python
def pick_ready(queues, policy):
    ready = collect_ready(queues)
    if policy is LONGEST_SHARED_PREFIX_FIRST:
        # group by stable-prefix hash; serve the largest hot group first so its
        # cache entry is reused before eviction. Ties -> depth-first (finish chains).
        return max(ready, key=lambda it: (hot_group_size(it.prefix_hash),
                                          it.depth))
    ...
```

For self-hosted serving APEX-Ω can additionally target SGLang for genuine cross-fork KV reuse and KVFlow-style eviction (evict far-off branches first, prefetch soon-to-run). For hosted CLIs this is purely a dispatch-ordering win — portable, free, and it degrades to no-op (never worse) if caching is off.

### 16.6 Futility / Token-Snowball Detection (start-less, not kill)

The most important honesty point in this section: **failures cost more than successes.** SWE-Effi documents the "token snowball" — off-track runs cost **4×+** a success ([arXiv:2509.09853](https://arxiv.org/pdf/2509.09853)). v1's hidden cost amplifier is exactly this: on partial progress the controller *adds* waves (escalation cap 20, progressive waves cap 6, follow-up loops cap 24, near-miss ×3 multiplier) with tokens uncapped. The lever is to **not start doomed work**, never to wall-clock-kill a working agent (v1's progress-based liveness is preserved verbatim — the four kill paths remain K1-stall, emergency-silence, preempt, dead-future; no `wallclock_deadline`).

Two gates, both **routing decisions made at turn/checkpoint boundaries**, never mid-subprocess (mid-subprocess prompt mutation is infeasible against opaque CLIs and breaks determinism/replay):

1. **Early localization-futility gate** (adopt). After localization is amortized once, route the next wave's budget *toward surviving hypotheses* rather than spawning K identical patch attempts on a dead frontier. This kills the "15/16 doomed at localization" waste *before* the patch loop. It **informs allocation; it never suppresses a candidate without execution evidence** (that would invert the Cardinal Contract — see Section 13 and the rejected "plan scoring as a hard prune").

2. **Token-snowball detector** (futility-based early termination, EET-style, [arXiv:2601.05777](https://arxiv.org/html/2601.05777): ~31.8% avg cost cut, negligible resolution loss). Detect a run *unlikely to produce usable output* and **stop spawning further turns for that one rollout** — at a checkpoint boundary, on progress-derived signal, not wall-clock.

```python
# futility.py — evaluated only at checkpoint boundaries (turn end / stage end)
def snowball_score(rollout, history) -> float:
    # progress-derived signals; ALL fail-open (can only delay a stop, never accelerate)
    s = 0.0
    s += w1 * no_diff_progress_turns(rollout)          # turns since last worktree edit
    s += w2 * repeated_blocker_count(rollout, history)  # same error N times (v1: stop after 3)
    s += w3 * token_burn_vs_p75(rollout)                # this rollout's burn vs fleet p75
    s -= w4 * verified_partial_progress(rollout)        # DOWN-weight if tests improving
    return s

def maybe_stop_spawning(rollout, history, cfg) -> bool:
    if has_confirmed_pass(rollout):        # CCEDF guard: NEVER stop a succeeding rollout
        return False
    return snowball_score(rollout, history) >= cfg.snowball_threshold
```

**Hard invariants (the three pitfalls, encoded):**

- *Do not flat-wall-clock-kill a working agent.* `maybe_stop_spawning` fires only on progress-derived futility at a checkpoint; a long legitimate thinking turn or CPU-busy test run is never stopped (v1 liveness preserved).
- *Do not let a budget cap abort an in-flight succeeding rollout.* `has_confirmed_pass()` (escrow-WAL/CCEDF lookup, Section 15) short-circuits every stop. A confirmed full-scope pass is durable and final.
- *Do not assume the snowball signal is correct.* All sub-signals fail-open; the detector can only *stop spawning new turns for one rollout*, never invalidate work already done.

### 16.7 Budget Kill-Switches & Cascade Routing (inside the loop)

v1's token-cap plumbing (`repo_token_cap`, `max_tokens_per_repo_followup`, `_cap_followup_rollouts_for_token_budget`, p75 cost estimation) is **built but inert**. APEX-Ω exposes it as an **opt-in `budget{}` primitive, defaulted unbounded** (honoring v1's "never optimize for cost" stance as the default, while making cost a first-class *optional* objective).

```yaml
budget:                                  # opt-in; defaults below = v1 behavior (unbounded)
  max_tokens_per_task: null              # null = unbounded (v1 default)
  max_usd_per_task: null
  max_followup_waves: 6                  # v1 cap
  kill_switch:
    scope: spawn_only                    # caps gate NEW spawns; never abort in-flight
    spare_confirmed_pass: true           # MANDATORY: a cap can never drop a verified pass
  early_termination:
    enable_snowball_detector: true
    snowball_threshold: 0.70
```

The kill-switch semantics are the load-bearing safety property: **a budget cap gates *new* spawns and *new* follow-up waves; it can never abort a rollout that is in flight, and it always spares a confirmed pass.** This is the direct resolution of the adversarial verdict's v1 tension — cost becomes an opt-in `budget{}` primitive, defaulted unbounded, never a gate that drops a verified pass (which would collide with the escrow-WAL invariant). Budget accounting is sourced from the per-vendor `CostLedgerEntry` sub-accounts defined in Section 12 (USD-normalized, because token units differ across tokenizers and Claude subscription `-p` draws a separate Agent-SDK credit pool from 2026-06-15); the kill-switch evaluates against estimated USD, never against a raw token count, and never against acceptance.

**Cascade routing** puts cheap executors on the happy path (Section 12 model economy; verdict: **adopt-modified, verification-gated**). Cheap/read-only sub-roles (reproducer, localizer, run/verify, narrow single-tool edits) route to a cheaper tier; the frontier tier stays on navigation and multi-file editing (HyperAgent ablation: cheapening navigation/editing causes the worst resolve-rate drops, [arXiv:2409.16299](https://arxiv.org/html/2409.16299v1)). Escalation is **cascade, not blind routing** — try cheap, escalate to frontier on the *first verify-on-diff failure*, with a rewrite-cycle cap — which both fits APEX's verify-on-diff loop and avoids xRouter's documented brittleness of static routing trees ([arXiv:2510.08439](https://arxiv.org/html/2510.08439v1)). The honesty caveat (modeleconomy verdict `partially_sound`): measure **cost-per-resolved-task net of verification** (including the N×N matrix), not gross executor tokens — the "almost-right trap" means a thin executor needing 3–4 retries can cost *more* than one frontier pass.

### 16.8 Warm CoW Worktree Pools & Snapshot-Restore Sandboxes

v1 already ships `WorktreePool` (`use_worktree_pool`): pre-warmed per-`(task, base_commit)` worktrees recycled via `git reset` are **~10× cheaper** than create+warmup (~4s each). APEX-Ω keeps this and hardens the seam the v1 ingest flagged: the pool is silently *defeated* when any request carries a `workspace_seed` (different baselines defeat pre-warming), which is exactly what the seed-carrying escalation/recovery paths do on long runs. APEX-Ω therefore maintains **per-baseline pools** so escalation waves still hit a warm pool rather than falling back to cold creates.

```yaml
worktree_pool:
  use_worktree_pool: true
  shared_object_store: true            # one fetch, N worktrees (Augment pattern)
  warm_pool_size_per_baseline: 4       # absorb K-wide bursts without cold creates
  cow_clone: reflink_or_btrfs          # millisecond CoW from golden image when available
  preserve_pool_on_seeded_requests: true   # fix: don't disable pool on escalation seeds
```

This makes the **millisecond fork substrate** that makes branching net-positive: warm CoW worktrees over a single shared git object store (one fetch, N working dirs) provision in milliseconds vs seconds. Where real runtime isolation is needed (ports/processes/kernel), layer worktrees inside snapshot-restored sandboxes (Firecracker snapshot memory-map+resume, tens of ms vs ~125ms–1s cold boot; warm pools still pay off for burst absorption).

**Honesty / pitfalls (do not over-claim):** practitioner snapshot numbers (28ms restore; 59ms p95 create→exec) are *indicative*, not peer-reviewed — real end-to-end was 2.7s p95 in one case study before optimization. Firecracker needs KVM (absent on macOS/many CI). Pure CoW wastes page-cache at high density (OverlayFS shares it). Git worktrees give **no** runtime isolation — they must be layered inside containers/microVMs for ports/processes, which is exactly why APEX-Ω keeps v1's per-rollout `fcntl`-locked worktree isolation (Section 15) as the *floor* under any faster substrate. The branching-economics verdict is explicit: without a millisecond-class fork substrate, re-running can be *faster* than forking (naive CRIU/E2B: hundreds of ms to seconds) — so the pool is the precondition that makes adaptive branching net-positive, not a guaranteed win on its own.

### 16.9 Bounding the O(N²) Cross-Validation Matrix

The N×N cross-validation matrix (each candidate's tests run on every other candidate's worktree) is the quadratic selection-cost hotspot. v1 already bounds it with **two-pass semantic clustering before the matrix** (exact AST fingerprint, then single-linkage merge at `ast_similarity_threshold=0.95`), which dedups behaviorally-equivalent patches so N = *clusters*, not raw candidates. APEX-Ω keeps clustering-before-matrix and adds **test-impact pruning** (CTDG as test prioritizer, verdict **adopt-modified**) to shrink the per-cell test set:

- **Reorder + dynamic-coverage prune** the tests each cell runs (reordering has *zero* false-negative risk; dynamic coverage is near-safe), then **full-suite backstop** keeps it honest. Static-AST gating is **rejected** (PyCG ~70% recall; reflection/monkeypatch/fixtures invisible; gating silently drops fault-revealing tests, violating execution-authority).
- For large surviving pools, optionally **sample pairs** rather than full N×N, but only above a pool-size threshold and always with the deterministic ranking as the tie-break floor.

```yaml
cross_validation:
  cluster_before_matrix: true
  ast_similarity_threshold: 0.95
  test_impact_prune: reorder_and_dynamic_coverage   # never static-AST gate
  full_suite_backstop: true                          # honesty backstop on the winner
  pair_sampling_above_n: 12                           # sample pairs only for large pools
```

This bounds the dominant quadratic term without ever silently dropping a fault-revealing test — the prune is *near-safe by construction* (reorder = lossless; dynamic coverage = observed, not predicted) and the backstop catches the residual.

### 16.10 Batch API for Non-Interactive Fleet Work

Non-interactive fleet work — benchmark evals, bulk refactors, test generation, backfills — routes to the async **Batch API** (OpenAI/Anthropic: **50% off** input+output for ≤24h async). This **stacks with caching**: batch + cached prefix reaches **~95% combined savings**. Only user-blocking / interactive work stays on the real-time path.

```yaml
execution_lane:
  interactive:  realtime          # user-blocking solves
  fleet:        batch_api         # evals, bulk refactors, test-gen, backfills
  batch_api:
    max_wait_hours: 24
    stack_with_cache: true        # prefix-stable + batch -> ~95% off
```

This is the cleanest "free" lever for the evaluation matrix (Section 20) and any bulk maintenance run: the same prefix-stability discipline that helps interactive caching makes batch+cache compound.

### 16.11 Putting It Together: The Cost/Latency Decision Flow

```
per task:
  1. amortize Phase-1 RepoContext + Phase-2 localization ONCE (v1; ~15% up-front)
  2. K = select_K(task)                          # adaptive low-K, portfolio floor raise-only (16.2)
  3. assemble prompts: [STABLE | VOLATILE]       # prefix-stable, lint-enforced (16.4)
       -> ProviderCacheAdapter.mark_cacheable()  # 90% off cached reads when it fires
  4. pipeline(items=K seeds, stages=[reproduce, localize, patch, verify],
              workers=warm_cow_pool,             # ~10x cheaper provisioning (16.8)
              dispatch=LONGEST_SHARED_PREFIX_FIRST)   # KVFlow ordering (16.5)
       per turn/checkpoint:
         - cascade-route stage to cheapest safe tier; escalate on verify-fail (16.7)
         - if maybe_stop_spawning(rollout): stop SPAWNING new turns   # futility (16.6)
         - track cache_read vs cache_creation as SLO; degrade if miss  (16.4.1)
         - budget kill-switch gates NEW spawns only; spares confirmed pass (16.7)
  5. cluster-before-matrix -> test-impact-pruned N×N cross-validation (16.9)
  6. cheap-first verification cascade -> Cardinal Contract selection (Section 13)

INVARIANTS (never violated for speed/cost):
  * progress-based liveness only; NO wall-clock kill of a working agent
  * a budget cap NEVER aborts an in-flight succeeding rollout (CCEDF)
  * cost levers re-rank/route/allocate; they NEVER promote an unverified candidate
  * best-of-N (full-cap) is the FLOOR we can never do worse than (thin-feedback backstop)
```

### 16.12 What We Explicitly Do *Not* Do (rejected for cost reasons)

| Tempting cost lever | Disposition | Why rejected |
|---|---|---|
| Flat wall-clock kill of slow agents | **reject** | kills slow-but-legitimate runs; v1's deliberately-avoided failure mode. Speed = not *starting* doomed work. |
| Budget cap that aborts in-flight rollouts | **reject** | collides with escrow-WAL/CCEDF; would drop confirmed passes. |
| Static-AST CTDG as a test-pruning *gate* | **reject** | ~70% recall; silently drops fault-revealing tests; violates execution-authority. |
| Pre-execution plan scoring as a *hard prune* | **reject** | false-negative pruning suppresses correct-but-unverified plans before evidence; inverse Cardinal-Contract violation. |
| Default full-cap 16 with caps OFF | **reject** | the headline cost pathology; replaced by adaptive low-K + budget-aware deepening (full-cap kept only as thin-feedback floor). |
| Assume server-side prefix caching is on | **reject** | must detect (cache_read vs cache_creation SLO) and degrade gracefully; promising true cross-fork KV reuse on hosted APIs is dishonest. |
| Heavy-orchestrator + *thin* executor as the default on hard repo SWE | **reject** (verdict `partially_sound`) | HyperAgent: cheapening navigation/multi-file editing causes worst resolve drops; the almost-right trap can cost more than one frontier pass. Cheapen only narrow run/verify/single-tool sub-roles. |
| Bit-reproducible agent *output* replay | **reject** | impossible across hosted APIs (temp-0 batch non-invariance); reproduce *artifacts* (diffs + re-run verification), not token streams. |

The net effect: APEX-Ω turns v1's "16 redundant full trajectories at `--effort max`, caps off" into a difficulty-adaptive, prefix-cached, pipeline-streamed, futility-gated fleet whose cost levers are all *bounded amplifiers* of the execution-authoritative kernel — every one of which degrades gracefully to v1 behavior, and none of which can ever make a deadline by shipping an unverified patch or by killing an agent that is still working.
