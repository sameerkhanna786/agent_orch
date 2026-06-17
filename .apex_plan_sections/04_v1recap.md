## 4. APEX v1: Foundation, Strengths & Ceiling

APEX-Ω is not a rewrite. It is an extension of a working, hardened kernel. This section is the load-bearing recap of that kernel — APEX v1, the "Adaptive Parallel EXecution" orchestrator — and it serves three jobs simultaneously. First, it fixes the **substrate**: the exact contract, control flow, and invariants the redesign builds *on top of* and is forbidden to weaken (see Section 7 for how these become first principles, and Section 18 for the kept/modified/dropped ledger). Second, it inventories the **reusable assets** so later sections cite concrete code seams rather than reinvent them (the engine in Section 2/8, verification in Section 13, the controller in Section 14, the model economy in Section 12). Third, it names the **ceiling** — the four change-seams and the cost stack — that every redesign mechanism is justified against.

The framing of this whole plan is honest about that order of operations: APEX v1 is the *substrate*; speculative search, CTDG pruning, the epistemic blackboard, the model economy, and the active controller are *extensions expressed as workflow patterns over vendor-neutral workers*. Nothing below is a proposal. It is the description of what already runs, what we keep verbatim, and where the next-generation work attaches.

### 4.1 The Contract: One `solve()`, One Diff, Four-Way Status

APEX v1 exposes exactly one entrypoint, and the entire plan inherits its signature as the public contract:

```python
ApexOrchestrator.solve(
    repo_path: str,
    issue_description: str,
    test_command: str | None = None,
    benchmark_metadata: dict | None = None,
    verification_test_command = _INHERIT_VERIFICATION_TEST_COMMAND,
) -> ApexResult        # apex/orchestration/solver.py:646
```

Given `(repo_path, issue, optional test_command)` it produces **one unified diff plus one terminal `Status`** drawn from a 4-way enum (`apex/core/status.py`):

| `Status`      | Meaning                                                            | `success` |
|---------------|-------------------------------------------------------------------|-----------|
| `SOLVED`      | A candidate passed execution-grounded verification and was accepted | `True`    |
| `ABSTAINED`   | No candidate earned positive execution evidence; we decline to guess | `False`   |
| `FAILED`      | Genuine APEX miss (a real attempt produced no acceptable patch)   | `False`   |
| `ENV_SKIPPED` | Environment/infra failure; *not charged to the model*             | `False`   |

`success == (status is SOLVED)`. The four-way split — and especially `ABSTAINED` as a first-class peer of `SOLVED` rather than a degenerate `FAILED` — is the structural expression of the central thesis: with an imperfect verifier, the cost of a confident-wrong accept dominates, so abstention-over-guessing must be representable in the type system, not buried in a confidence float. APEX-Ω keeps this enum byte-for-byte.

The result is produced by a strict five-phase pipeline, driven by `solve()` as a thin, stateless-across-runs sequential coordinator:

```
preprocess (1) -> plan (2) -> rollouts (3) -> verify (4) -> select (5)
```

```
solve()
  ├─ _maybe_solve_via_in_container_v5     # benchmark short-circuit (gated by benchmark_metadata)
  ├─ _prepare_run                         # Phases 1-2 + baseline -> 9-tuple
  │     (repo_context, verifier, planner, strategy, issue_plan,
  │      task_state_graph, baseline_result, resolved_verification_test_command,
  │      orchestration_transitions)
  ├─ RolloutEngine(config, repo_path, repo_context)
  ├─ _run_pipeline                        # Phases 3-5 + 4 follow-up recovery loops
  └─ _build_final_result -> ApexResult
```

Phase data-flow and the load-bearing objects:

| Phase | Producer | Output object | Lifecycle discipline |
|-------|----------|---------------|----------------------|
| 1 preprocess | `RepoAnalyzer.analyze()` | `RepoContext` | **Built once, read-only thereafter** (amortized context — scan the repo once, reuse across all parallel attempts) |
| 2 plan | `planner.build_execution_strategy()` | `PlanningDecision` + `IssuePlan` | `IssuePlan` is **the central mutated-throughout object**, threaded into every rollout, escalation, follow-up, and selection |
| baseline | `verifier.capture_baseline()` | `BaselineResult` | One full-suite run per `(repo, command)`, cached |
| 3 rollouts | `RolloutEngine.execute_rollouts()` | `list[RolloutResult]` | `RolloutResult` is the **atomic unit** flowing generation → verification → selection → recovery |
| 4–5 verify+select | `PatchVerifier` + `PatchSelector.select_best_patch()` | winning `RolloutResult` | Phase 4 is **not** a standalone method — verification runs *inside* the selector and inside per-rollout `quick_verification` |

One subtlety to carry forward: **there is no separate benchmark pipeline**. The published numbers come from the same `solve()` a library user runs; `benchmark_metadata` is the *only* differentiator and merely gates two short-circuits (the V5 in-container path and artifact-safe plan stripping). This is a deliberate anti-cheat property — any divergent benchmark path reopens benchmark-specific cheating — and Section 3's vendor-neutrality argument and Section 20's evaluation plan both depend on it.

### 4.2 The Non-Negotiable Invariants (and Why Each Is Load-Bearing)

The credibility of best-of-N orchestration rests on a small set of invariants. The redesign's job is to *amplify* what these protect, never to relax them. Each is stated with the guarantee it provides and the specific failure that re-emerges if it is removed.

#### 4.2.1 The Cardinal Safety Contract — execution-evidence-authoritative selection

> *"Execution evidence is authoritative. Soft signals may re-rank within an execution-verified tier, or downgrade an already-accepted candidate — they may NEVER promote an unverified candidate."* (Blueprint §13.1)

This is the single rule running through every selection component. It is enforced *structurally*, not by convention:

- `_apply_evidence_bound_review` flips `accepted` **only `True → False`**. The adversarial veto, the fresh-context `FinalAcceptanceReviewer`, and the clarification-abstain arm are all downgrade-only. (`VerificationResult` has no `status`/`passed` field; `accepted` is the gate.)
- The deterministic ranking key is a fixed lexicographic tuple in which every soft/learned/LLM key sits strictly below every execution + critic key, terminating in a content-derived tiebreak — never insertion order:

  ```
  rank_key = (
      combined_score, accepted, public_signal_score, critic_score, size,
      verification_score, eg_critic_tiebreak, perspective_score,
      len(changed_files), -cluster_id            # never insertion order
  )
  ```

**Load-bearing because:** this directly counters the *Inference Scaling Flaws* failure mode — with an imperfect verifier, repeated-sampling gains saturate or invert because wrong code that the verifier (or an LLM judge) likes gets shipped. Pointwise LLM judges capture only a fraction of best-of-N gain, so the LLM is bounded to a downstream tie-break. Remove this rule — let any soft signal (deterministic `SelectionCritic`, learned EG-critic, `SelectorAgent` vote, perspective/final-acceptance reviewers, process quality, evidence ledger) promote an unverified candidate — and the selector ships LLM-preferred-but-unexecuted patches; the published number stops measuring capability; the entire credibility argument collapses. This is why every redesign mechanism (CTDG pruning §10, plan scoring §14, blackboard sharing §11, the generative critic §13) is admitted only in **re-rank-within-tier or downgrade-only** form. The Cardinal Contract is what converts log-linear *coverage* into trustworthy *resolved issues*.

#### 4.2.2 The cheap-first verification cascade that never synthesizes a pass

A confidence-ordered ladder runs high-precision cheap filters first, so expensive selection only sees survivors:

```
syntax (AST py_compile of changed files; hard 0.0 on fail)
  -> lint (flake8 --select=E9,F63,F7,F82; no bonus if flake8 absent)
  -> reproduction (+0.35)
  -> regression-prune (re-run ONLY baseline-passing tests, chunks of 50, in candidate worktree; +0.35)
  -> cross-validate (+0.10 * mean)
  -> score (pass_rate adds 0.10 * pass_rate)
  -> accept
```

Three encoded countermeasures define the "never synthesize a pass" discipline:

- `rc == 124` (timeout) → a **separate axis** `regression_inconclusive` (+0.15 partial), *not* a failure.
- Silent `rc == 0` no-op → `errors = 1`, **never `passed = 1`**.
- Singletons **abstain** (empty cross-validation list), never a synthetic `0.5` prior; the self-index `M[i][i]` is excluded.

Regression pruning is a precision *and* speed primitive: it re-runs only baseline-passers in targeted chunks of 50 nodeids in the candidate worktree (not the whole suite), and on a collection error it expands the file key to every baseline nodeid under that file prefix so import-time breakage cannot silently survive. `PruneResult.is_valid == False` drops the candidate.

**Load-bearing because:** without cheap-first ordering, cost explodes (cross-validation and LLM votes run on syntactically-broken candidates). Without the guards, a no-op command, a timed-out-but-passing suite, or a zero-collected run is recorded as a pass — the exact false-`SOLVED` paths the system exists to prevent. This cascade *is* the safe per-candidate prune the redesign wanted (Section 10, Section 13); it is already proven, and CTDG/coverage pruning attaches as a *prioritizer and backstop around it*, never as a replacement gate.

#### 4.2.3 Per-rollout git-worktree isolation + scoped `fcntl` locks

Every rollout gets a private workspace. The per-rollout `fcntl` advisory lock (`flock LOCK_EX|LOCK_NB` at `workspace_dir/.locks/rollout_<id>.lock`) is taken **before** touching the workspace path and released on every failure path; it raises `ConcurrentWorktreeError` if held (Windows fallback = PID marker). Isolation degrades downward through three tiers plus an override:

```
seed_clone -> worktree -> snapshot -> synthetic
```

Snapshot baselines are deterministic (`_bootstrap_git_snapshot` commits with fixed author/committer and date `2026-01-01T00:00:00+0000`, message derived from `source_head8 + dirty_hash8`), so two identical source states produce **bit-identical commit SHAs → bit-identical diff text**. Critically, **there is no machine-wide mutex** — all concurrency, kills, registries, and secret boundaries are scoped to a single rollout (the `RolloutCLIRegistry` binds each CLI pid to a `rollout_id`; `terminate_rollout_children` escalates `SIGTERM → SIGKILL` across only that rollout's pids).

**Load-bearing because:** this is the primitive that makes *any* parallelism or branching safe. The CAID ablation is explicit — soft isolation 55.5 < single agent 57.2 < worktree **63.3**. Remove it and K parallel attempts corrupt each other's filesystem, cross-rollout writes pollute candidate diffs, reaping one stalled rollout touches its siblings, and untrusted external agent code running test suites leaks into shared state. The speculative tree-search of Section 9 and the parallel/pipeline fan-out of Section 8 are *only* sound because every branch lands in an isolated, lock-guarded worktree.

#### 4.2.4 Filesystem-as-source-of-truth (the vendor-neutrality enabler)

Every phase boundary persists a durable atomic-write artifact (`repo_context.json`, `baseline_result.json`, `issue_plan.json`, `controller_decisions.jsonl`, `apex_result.json`, `run_manifest.json`). `status`/`watch` subcommands **read artifacts only** — they never attach to the scheduler or hold a lock. Results are written to disk, **not flooded back into an orchestrator context window**. The atomic-write pattern is uniform: normalize target → `json.dumps(indent=2, sort_keys=True, default=str)` → `mkstemp(...).tmp` → write → flush → `os.fsync` → `os.replace` (readers see old-or-new, never torn) → unlink tmp on failure.

**Load-bearing because:** this is the concrete mechanism of vendor neutrality (Sections 2–3). Because state lives on disk and each agent is observed via stdout turn-parsing + artifacts rather than driven in-process, opaque external CLIs (`claude`/`codex`/`gemini`/`opencode`/`metacode`) are interchangeable behind `AGENT_NAME_TO_CONFIG`. It is also independently corroborated by context-rot findings (frontier models degrade well below their window limit, lost-in-the-middle): keeping intermediate results in script variables and a durable journal — never a conversation window — is the scaling unlock that lets a large run avoid drift. Remove it and the system couples to a specific in-runtime agent framework, monitoring perturbs runs, crashes lose all intermediate state, and there is no auditable record of why a rollout ended or which candidate won.

#### 4.2.5 Determinism + `RunManifest` + Docker digest pinning + strict replay

Determinism is **best-effort around irreducibly-stochastic agents** (temperature 0.0 default; CLI backends ignore temperature entirely) but pinned everywhere pinnable: candidate ordering by `(rollout_id, content_hash)`, cluster verification in `cluster_id` order, `Random(0)` mutation seeds, content-sha tie-breaks. `RunManifest` captures git sha/dirty, python/platform, seed, redacted `APEX_*` env, model ids, Docker digests, harness versions. `resolve_image` pins tags to `repo@sha256:` via `prepinned -> registry -> docker_inspect -> bare`, with an `@sha256:`-in-tag short-circuit that avoids the malformed double-pin bug; it never raises and records which path won. `apex replay-deterministic --verify` re-runs a recorded session and asserts the reproduced trajectory matches.

**Crucial honesty constraint (carried into Section 15):** manifest pinning guarantees *environment + ordering*, and replay guarantees the *trajectory*; neither guarantees agent **output**. Bit-reproducible agent OUTPUT replay is impossible across hosted APIs (temperature-0 batch non-invariance) and is therefore explicitly rejected in the canonical disposition list — APEX reproduces *artifacts* (diffs + re-run verification), not token streams.

**Load-bearing because:** without manifest + pinning, image drift silently changes scores and provenance cannot distinguish environment from model; without deterministic ordering/tie-breaks, re-runs select different winners and slot-0 bias creeps in; without replay, there is no debugging substrate for a stochastic agent. This is also what makes a *learned* controller (Section 14) tractable: reproducible off-policy credit assignment over journaled decisions.

#### 4.2.6 Escrow WAL / commit-then-publish / idempotent exactly-once (CCEDF)

The escrow WAL at `<run_dir>/escrow/confirmed_wal.jsonl` is fsync-durable, `flock`-guarded, monotonic-seq, idempotent exactly-once durability for confirmed candidates. Replay is latest-wins by `seq` per `idempotency_key` (`task_id::candidate_id`) then best-by-task by `(score, seq)` — **ordering uses `seq`, not wall-clock**, so replay is order-stable and a duplicate append is harmless. The engine wraps the call in a bare `except`: durability is a backstop that must never become fatal.

**Load-bearing because:** it fixes the dominant Commit0 loss — a rollout that reached `pass_rate == 1.0` then dropped to `scheduler_cancelled` and was lost. Remove it and a confirmed full-scope pass produced early, then preempted during a later wasted wave, is lost forever; long-running preemptible parallel search becomes lossy and non-resumable; published rates understate true coverage. The redesign promotes this narrow WAL plus v1's unused `ReplayRecorder` into a **per-`agent()`-call WAL** (Section 15) that doubles as the off-policy credit substrate for the learned controller — but the durability semantics here are kept verbatim.

#### 4.2.7 Anti-cheat / fairness / failure taxonomy / first-class abstention

A coding-agent orchestrator is only as credible as its number, and capable agents are reliable adversaries against weak oracles. The non-negotiables:

- **Upstream Docker harness is the only publishable number.** APEX-private rescoring is diagnostic-only and published as a delta (`fairness_audit.json`, `FLAG_THRESHOLD = 0.02`); the fairness audit runs two scorers over the *same* pre-computed evaluation (O(N), not O(2N)).
- **Gold fields are scrubbed at LOAD time — discard, not mask.** The classic `SWEBenchTask` dataclass literally has no field to hold `patch`/`test_patch` (only `scrubbed_test_patch_summary` + integer F2P/P2P counts). Masking is rejected because a redacted field is one prompt-template bug from leaking.
- **Commit0 uses a true git-history flatten** (`rm -rf .git` + nested submodule `.git`, rootless re-init, post-conditions assert `rev-list --all --count == 1`) — *block the channel, never the neuron*: the gold-recovery channel is destroyed while weights-encoded capability is untouched.
- **NDFF flake firewall** declares flakes only on positive evidence and never re-runs a real failure; the **anti-hack ledger** grounds assertions by structured fingerprint (op+shape+repr), biases to false-negatives, soft-downweights.
- **`patch_sanitizer`** distinguishes `VENDORED_UPSTREAM_ARTIFACT` (stripped silently, requalified as a soft advisory signal) from `GOLD_PROTECTED_TEST` (reject wins) so an attacker cannot disguise a gold-test edit under a vendored `testing/` dir.
- **Failure taxonomy** (`FailureClass`, 8 buckets + `UNCLASSIFIED`, phase-ordered, never raises): `charges_apex == {APEX_MISS, UNCLASSIFIED}` only; `HARNESS_BUG` is excluded from *both* `is_environment` and `charges_apex`; `ENV_*`/`NON_DETERMINISTIC` are retryable. Phase makes `ModuleNotFoundError` an install-failure during setup but an APEX-miss during test execution.
- **Abstention is first-class** (echoing §4.2.1): all-env-failed → `None`; `cross_candidate_voter` returns `winner=None` when all `oracle_scores == 0` (caller keeps baseline pick); the strict gate `_selected_result_is_accepted` requires positive evidence (legacy `overall_score >= 0.9` shortcut removed); the deterministic `HeuristicRepairAgent` fallback stamps a hard-coded `0.35` confidence + "LLM execution was unavailable" so a heuristic patch is never mistaken for a real solution.

**Load-bearing because:** reward-hacking *scales* with capability and contamination is the dominant validity threat — non-negotiable for a mixed-vendor fleet (Section 3). Remove any one and a sufficiently capable agent finds a cheaper-than-solving path to a green checkmark; the headline number stops measuring capability; environment failures get charged as model misses (or vice-versa), corrupting both the published rate and the retry policy.

### 4.3 Existing Assets the Plan Reuses

The redesign explicitly lifts and re-implements, rather than rebuilds, a mature vendor-agnostic foundation. The canonical disposition for the workflow engine is **adopt**: lift v1's `run_structured_prompt` (= the `agent()` primitive) and `execute_rollout_requests` (= the `parallel()` primitive) into a re-implementable engine. The assets below are the concrete seams.

| Asset (v1 location) | What it is | Where the plan reuses it |
|---------------------|-----------|--------------------------|
| **`FrontierSearchController` (PUCT)** | The existing bounded best-first search over rollout frontier with virtual-loss / `min_branch_reward` accounting | Section 9 — the **antecedent** for adaptive branching; *not* re-described as new |
| **`EpisodicMemoryBus`** (`rollout/engine.py`) | Append-only, thread-safe Discovery store; cross-sibling + cross-solve priors (reserved negative `rollout_id`s ≤ −1); negative/ruled-out sharing; `query()` excludes caller's own id; caps `positive_limit=5 + negative_limit=3`; `extract_durable_insights` caps at 64 | Section 11 — Blackboard 2.0 *evolves its delivery*, keeping relevance/confidence/dedup/own-rollout-exclusion |
| **`RepoGraph` / `RepoContext`** | Built-once, read-only repo model | Section 10 (CTDG attaches to it) and the amortized-context discipline |
| **`contract_slice.py`** | Localization / scope-slicing primitive | Section 10 (CTDG test prioritizer + dynamic-coverage prune) |
| **`BackendPortfolio`** (`core/backend_portfolio.py`) | Per-run persisted ledger (`run_backend_portfolio.json`) of disabled `(backend, command)` fingerprints with `retry_after_seconds`; `is_disabled` self-evicts expired entries | Section 14 — two-tier failure memory; prevents a 429 on one vendor poisoning a heterogeneous fleet |
| **`enable_speculative_first_attempt`** | The existing cheap speculative first attempt (gated to easy tasks, `difficulty <= 0.25`) | Section 9 — `speculate()` fork is admitted as an *extension* of this, at turn/checkpoint boundaries only |
| `CLIModelClient.run_structured_prompt` | Launches an opaque CLI subprocess running its own multi-turn loop; observes via stdout + watchdog; **never raises** (typed `CLIModelResult`) | Section 2/8 — the `agent()` worker primitive |
| `execute_rollout_requests` + `WorktreePool` | K-wide parallel rollout dispatch with ~10x worktree recycling | Section 2/8 — the `parallel()` primitive |
| `CLITurnParser` + `turn_observer` | Splits stdout into `Turn` objects; the *only* mid-flight steering channel over an opaque CLI | Section 11/14 — turn-boundary sharing & control |
| `controller_decisions.jsonl` | Already logs every controller decision | Section 14 — substrate for the learned active controller |

**A standing caution the plan must honor:** `FrontierSearchController` is the antecedent, not a novelty. The redesign's "speculative tree-search" *re-describes* this existing search. The disposition is therefore **adopt-modified** — keep the AB-MCTS-style adaptive allocation that wins, run it *inside* `FrontierSearch` budget caps, and make collapse to verified best-of-N mandatory below a feedback-confidence floor. Classical/distributed MCTS as the core loop is **rejected**: that verdict is unsound (it re-describes `FrontierSearch`, plain MCTS does not reliably beat verified sampling at repo scale, and it is brittle against non-serializable container state).

### 4.4 The Four Change-Seams

The redesign attaches at exactly four seams. Each is a clean integration point with existing dataclass boundaries; none requires touching the §4.2 invariants.

| # | Seam | Current v1 behavior | Redesign attachment (target section) |
|---|------|---------------------|--------------------------------------|
| 1 | **Linear scaffolded pipeline** | Strict one-directional JSON-artifact handoff `Reproducer → Localizer → Patcher [→ TestWriter]`; no inter-agent dialogue; agents "deliberately ignorant of waves/escalation/selection" | The `pipeline()` per-item staged-streaming primitive (Section 8) — the one genuinely net-new engine primitive; cuts wall-clock from sum-of-slowest-per-stage to slowest-single-chain across reproduce→localize→patch→verify. The JSON-only anti-misalignment discipline is preserved |
| 2 | **Passive controller** | Two layers: the wave/escalation loop only decides count/escalation; the calibrated policy layer is *blend-not-switch* (`evaluate_policy_model`: `applied=False ⇒ value==baseline`) and its intended kill switch `library_enabled` is **unwired** (zero runtime consumers) | The active adaptive controller (Section 14) — staged bandit → GEPA → RL; **blend-not-switch, fail-open to heuristic** preserved; `library_enabled` finally wired or removed |
| 3 | **Blind redundant rollouts** | N redundant attempts at the SAME task in isolated worktrees; diversity by strategy-axis/prompt/seed; no coordination during generation | Bounded adaptive branching + `(vendor, model)` as a diversity axis (Sections 9, 12) — steer later rollouts away from confirmed dead ends; worktree isolation kept as the hard safety primitive |
| 4 | **Append-only blackboard** | `EpisodicMemoryBus` + typed `TaskBlackboard` are append-only discovery stores | Blackboard 2.0 (Section 11) — phased, abstracted negative-constraint sharing at turn boundaries; verifier must not see producer context |

Two seam dispositions are explicitly **rejected** and must not creep back in: raw *share-all / "instant push" mid-subprocess injection* (share-all measurably lowers accuracy and homogenizes attempts; mid-subprocess prompt mutation is infeasible against opaque CLIs and breaks determinism/replay), and *heavy-orchestrator + thin-executor as the default shape* (HyperAgent ablation shows cheapening navigation/multi-file editing causes the worst resolve-rate drops).

### 4.5 The Ceiling: The Cost Stack and "15/16 Doomed"

The ceiling is cost, and it is structural. APEX v1 is built to **"optimize for SOTA, never for cost,"** and the defaults make that literal:

- **Full-cap-16 default, adaptive OFF.** `RolloutConfig.enable_adaptive_allocation = False`, so the planner selects `max_rollouts` (default **16**, buckets `[1, 4, 8, 16]`) regardless of difficulty; the portfolio floor only *raises* the count, never lowers it. The difficulty-adaptive low-K path exists and is fully wired (`estimate_difficulty → compute_rollout_count → evaluate_policy_model('planning.rollout_count') → _clamp_rollout_bucket`) — it is just off by default. Turning it **ON by default** is the canonical **adopt** disposition and the single biggest cost lever (optimal K is often < 10).
- **Caps off.** `repo_token_cap = None`, `max_tokens_per_repo_followup = 0`; the entire cumulative-token-cap machinery is inert in a normal run. Default full-cap-16 redundant trajectories is **rejected** as the headline cost pathology — replaced by adaptive low-K + budget-aware deepening, keeping full-cap only as the thin-feedback floor.
- **No wall-clock kill of a working agent.** Liveness is progress-based (S1–S7 inner watchdog; K1 outer stall, window 1200s × size_factor up to 6; emergency-silence ceiling 14400s = 4h; hard timeout opt-in, floored at 1800s). This is correct (it avoids killing slow-but-legitimate work) but it is also why a single rollout can run for hours.

The cost is paid in four multiplicatively-stacked layers — the **K × N² × waves** stack:

```
total ≈  K  (generation: up to 16 full agent trajectories, each up to
             max_iterations_per_rollout turns at an 80–120k context ceiling)
       × per-rollout in-loop verification (targeted pytest, cached per patch)
       × N² selection (cross-validation matrix: each candidate's tests run on
             every other candidate's worktree, sandboxed)
       × waves (escalation loop cap 20, progressive waves cap 6,
             follow-up iterations cap 24, selection rounds 4 ×3 on near-miss)
```

The **"15/16 doomed at localization"** pattern is the headline waste, stated honestly: **localization is amortized once** (top-K hypotheses from a single localizer seed K rollouts via `hypothesis i → rollout i mod K`), but the **full patch-and-verify trajectory is replicated K times**. On an easy task that one good rollout would solve, the other 15 trajectories are pure redundant spend. The precise framing matters for the redesign: the redundancy lives in *patching*, not localization, and coverage scales log-linearly — so the binding constraint is *selection*, not localization. The **early localization-futility gate** (canonical **adopt**) routes budget to surviving hypotheses *before* the patch loop, informing allocation but — per the Cardinal Contract — never suppressing a candidate without execution evidence.

The dominant absolute cost hotspots, in priority order for Section 16 to attack:

| Hotspot | Why expensive | Rough magnitude |
|---------|---------------|-----------------|
| **Parallel rollout generation** (K opaque CLI agents/task) | Each rollout is a full multi-turn agent solve in its own worktree; default backend at `--effort max`; no token cap | The largest absolute driver: up to **16× a single full solve**, × up to 6 waves / 20 strategy iterations on hard tasks. Inner parallelism capped at `parallel_workers=3` ⇒ ~`ceil(16/3)` sequential batches of the slowest rollout |
| **N×N cross-validation** (`build_cross_validation_matrix`) | Each candidate's suite executed against every other candidate's patched worktree, each a full sandboxed run (per-suite timeout 120s) | **O(N²)** sandboxed test executions/task; the dominant per-task verification cost for large ensembles. Two-pass AST clustering (threshold 0.95) dedups first, the main lever bounding N |
| **Regression baseline + prune** | Baseline = one full-suite run/`(repo,command)` up to 900s (cached); prune re-runs baseline-passers in chunks of 50 per candidate | `1 × full-suite` + `O(candidates × baseline_passers/50)` chunked pytest invocations |
| **LLM selection arms** | `SelectorAgent` up to 5 voters × 8 iterations; `PerspectiveReviewer` 4 lenses; `FinalAcceptanceReviewer` 1 fresh-context pass | Only on the tie-break path (≥2 selectable clusters); all default-off-or-fail-open, capping cost-and-variance |
| **Escalation + 4 follow-up loops** | On partial progress the controller *adds* rollouts rather than stopping; near-miss (≥0.95 pass rate) triggers a 3× multiplier on selection rounds | Bounded by iteration caps (20/6/24/4), **not** by tokens |
| **F2P oracle + dual-version voting** | Clones the repo twice, applies gold patch, runs the suite on both checkouts; dual-version generalizes to tests × surrogate-patches | `2 clones + 2 full suite runs`/F2P eval; `~(T + T·P)` sandboxed runs for the dual-version matrix |

### 4.6 What the Recap Establishes

This recap is the motivation to **keep**. The seven invariants of §4.2 are the substrate APEX-Ω inherits unchanged; removing any one collapses a named, specific guarantee, so the redesign is constrained to *amplify within them*. The assets of §4.3 are re-implemented, not rebuilt, and `FrontierSearchController` in particular is the antecedent the redesign re-describes — credit it as existing, not novel. The four seams of §4.4 are precisely where bounded, evidence-respecting extensions attach. And the cost stack of §4.5 — full-cap-16 with caps off, the K × N² × waves multiplier, and "15/16 doomed" in the *patch* loop — is the ceiling every later mechanism is justified against, with difficulty-adaptive low-K allocation (default ON) and the early localization-futility gate as the first, cheapest, highest-leverage wins. The honest through-line, carried forward to Section 7: **search and economy are bounded amplifiers; execution evidence is both the steering signal and the brake; verified best-of-N is the floor we can never do worse than.**
