## 21. Risk Register & Mitigations

This section is the project's standing list of ways APEX-Ω can fail and the engineered defenses against each. It is written to a single discipline borrowed from APEX v1's failure taxonomy (Section 4): **no risk is listed without a concrete mitigation AND a fallback that ships a still-defensible system if the mitigation does not hold.** The two highest-likelihood risks — R1 (the learned controller learns vendor identity, not capability) and R3 (cheap-executor ceiling) — are deliberately treated first and most heavily, because they are the ones most likely to actually fire on real repository SWE.

The register is organized as engineered controls, not aspirations. Every mitigation maps to a primitive, config key, journal record, or CI gate defined elsewhere in this plan, so a coding agent on either Codex or Claude Code can build the defense, not just read about it. The unifying invariants that backstop the whole register are the ones from the Central Thesis (Section 7): **execution evidence is authoritative; soft signals never promote; best-of-N is the floor we can never do worse than; speculation/economy/pruning are bounded amplifiers gated by the Cardinal Safety Contract (Section 13).** Where a mechanism is judged unproven by the adversarial verdicts, it is carried as guarded/optional/default-off, never as a certainty.

### 21.1 Risk register data model

Risks are not prose-only; they are first-class run-time and CI artifacts so the orchestrator can actually detect and react to them. Each risk has a typed record persisted to `<run_dir>/risk_register.jsonl` (atomic append, same `atomic_write_json` discipline as v1 artifacts) and a corresponding CI acceptance gate.

```python
@dataclass(frozen=True)
class RiskControl:
    risk_id: str                 # "R1".."R9"
    title: str
    likelihood: str              # "low" | "low-medium" | "medium" | "medium-high" | "high"
    capability_correlated: bool  # True => gets WORSE with stronger base models
    primary_invariant: str       # which Cardinal/thesis invariant this defends
    detectors: list[str]         # runtime signals + metric names that fire the alarm
    mitigations: list[str]       # engineered controls (config keys / primitives)
    fallback: str                # the still-shippable system if mitigation fails
    ci_gate: str | None          # acceptance test name that must pass to ship the feature
    owner_section: int           # cross-ref to the section that builds the control
```

The orchestrator emits a `RiskEvent {risk_id, ts_seq, detector, value, threshold, action_taken}` to the same journal whenever a detector trips. `ts_seq` is a monotonic sequence number (NOT wall-clock — same rule as the escrow WAL, Section 15), so the risk log is replay-stable. The default action for every capability-correlated risk (R1, R3, R6) is **fail toward the floor**: disable the amplifier, fall back to verified best-of-N, and keep going — never abort a succeeding run (v1's "a cap must never abort succeeding work" invariant, generalized).

The detector→action loop is itself a deterministic, journaled control path so a coding agent can build it identically on either Codex or Claude Code:

```python
# Evaluated at every phase boundary and before any amplifier is engaged.
# Pure function of recorded evidence + config; emits journaled RiskEvents; never raises.
def evaluate_risk_controls(run_state, registry: list[RiskControl], cfg) -> set[str]:
    disabled_amplifiers: set[str] = set()
    for ctrl in registry:                       # deterministic order by risk_id
        for det in ctrl.detectors:
            value = read_detector(run_state, det)   # fail-open: missing signal -> None
            if value is None:
                continue                            # absent evidence never trips a control
            threshold = cfg.thresholds[det]
            if breaches(det, value, threshold):
                # The action NEVER excludes a candidate; it only disables an amplifier
                # or quarantines (downgrade-only, per the Cardinal Safety Contract).
                action = ctrl_default_action(ctrl)  # e.g. "disable_learned_routing"
                journal(RiskEvent(ctrl.risk_id, next_seq(run_state),
                                  det, value, threshold, action))
                apply_to_floor(run_state, action)   # idempotent; flips a feature flag off
                disabled_amplifiers.add(ctrl.risk_id)
    return disabled_amplifiers   # caller runs the still-defensible floor for these

# A disabled amplifier is sticky for the run (a flapping signal cannot thrash the engine);
# re-enable only across runs after the gating CI test for that risk passes again.
```

The invariant enforced by `apply_to_floor` is that **every action is monotone toward the floor**: it can disable an amplifier, route heavier, or quarantine a candidate (True→False), but it can never promote an unverified candidate, exclude a candidate pre-execution, or abort an in-flight succeeding rollout.

| Risk | Title (abbrev) | Likelihood | Cap.-correlated | Default action on detect | Floor it falls back to | Owner §|
|------|----------------|------------|-----------------|--------------------------|------------------------|--------|
| R1 | Controller learns vendor identity | medium-high | yes | disable learned routing → heuristic | bandit router + cross-vendor diversity | 14 |
| R2 | Search signal too weak to beat floor | medium | no | collapse to verified best-of-N | adaptive-K best-of-N | 9 |
| R3 | Cheap-executor ceiling / contract underspec | medium-high | partly | escalate to frontier; route heavy | frontier-everywhere | 12 |
| R4 | Pruning/caching degrades trust anchor | medium | no | full-suite backstop; reorder-only | full regression prune | 10 |
| R5 | Determinism/replay broken by speculation | medium | no | journal cancels; seed from hash | sequential, no speculation | 15 |
| R6 | Reward-hacking / contamination | high | yes | quarantine candidate; hidden-test gate | upstream harness only | 13 |
| R7 | Harness-leak erases diversity | medium | no | conformance test fails loud | single-vendor floor | 3 |
| R8 | Vendor CLI drift | high | no | failure-memory eviction; fail-soft | other vendor / API path | 3 |
| R9 | Model-authored control-flow nondeterminism | low-medium | no | freeze-then-journal; ban RNG/clock | deterministic planner-authored | 2,15 |

---

### 21.2 R1 — The learned controller learns vendor IDENTITY, not transferable capability (kills the paper)

**Likelihood: medium-high. Capability-correlated: yes.** This is the single highest-stakes risk because the open-pool active controller (Section 14) is the plan's defensible NeurIPS-grade contribution, and the failure mode is plausible, not hypothetical. Capability-profile estimation for opaque CLI workers is hard, the strongest single model often dominates hard SWE, and a controller given `(vendor, model)` as features can trivially memorize "Claude on hard tasks" instead of learning "this *capability profile* on this *task profile*." If it does, the contribution is an overfit lookup table that will not transfer to a held-out vendor — and the headline scientific claim collapses.

#### Why the naive mitigation is insufficient

You cannot detect this with in-distribution validation accuracy: a vendor-identity memorizer scores *better* in-distribution than a capability learner. The detection MUST be a held-out-vendor protocol, and it must be built first, before any learned routing ships.

#### Mitigations (engineered, in build order)

1. **Held-out-vendor evaluation as a continuous gate, built FIRST.** Before Stage-0 of the controller (Section 14) ships, build the eval harness that trains the controller with vendor V excluded entirely from training, then measures routing quality on tasks where V is in the pool. Config: `controller.heldout_vendor_eval = ["<vendor>"]`, run on every controller artifact. The gate (`ci_r1_heldout_transfer`) requires the learned router to beat the heuristic router on the held-out vendor by a pre-registered margin; if it does not, the learned router is auto-disabled for production (`controller.learned_routing_enabled` flips to `false`, journaled as a `RiskEvent`). This mirrors the failure-modes guidance to validate only on uncontaminated, capability-attributable signal, not in-distribution deltas ([Limits of Inference Scaling Through Resampling](https://arxiv.org/abs/2411.17501)).
2. **Capability-profile-vs-one-hot ablation (the negative result is itself reportable).** Run the controller in two arms: (a) features = learned capability/cost profiles (per-role resolve rate, token yield, calibration, latency); (b) features = one-hot `(vendor, model)` identity. If arm (a) does not generalize better than arm (b) on held-out vendors, that *is* the paper's honest finding ("when learned routing helps and when it just memorizes"). Config: `controller.feature_ablation = ["profile", "onehot"]`. The profile features are deliberately **vendor-blinded at inference**: the policy sees the capability vector, not the vendor string, so identity cannot leak through the feature path.
3. **Staged design so partial failure still ships value (bandit → GEPA → RL).** Stage-0 ships a bandit router that learns per-role arm values from execution-grounded reward; even if GEPA (Section 14, deferred Stage 1) and full RL (deferred Stage 2) fail to generalize, the bandit + cross-vendor diversity (Section 3, the `(vendor, model)` diversity axis) still yield a working, cost-reducing system. This is the explicit "adopt-modified, staged, blend-not-switch, fail-open to heuristic" disposition from the accepted-mechanisms list.
4. **Blend-not-switch + fail-open.** The controller never hard-switches the whole fleet to one vendor; it blends allocation across the portfolio and falls open to the deterministic heuristic router on any controller error or low-confidence decision. A degenerate "always route to vendor X" policy is detectable (allocation entropy collapse) and is itself a R1 detector.

#### Detectors

- `controller.allocation_entropy` per task-difficulty bucket: a sharp drop toward a single vendor on hard tasks is a memorization warning.
- `ci_r1_heldout_transfer` margin: learned-minus-heuristic resolve rate on held-out vendor; negative ⇒ disable.
- Feature-attribution audit: if vendor-identity-correlated features dominate the learned policy's decisions, flag.

#### Fallback (still defensible without the learned win)

Ship APEX-Ω as a **cost-reducing systems contribution + a rigorous "when does learned routing help?" study.** The substrate (orchestration-as-code, durable resume, pipeline streaming, vendor-neutral executor), the Cardinal-Safety-gated verify-and-refute, cross-vendor diversity, adaptive-K, and the verification cascades all stand alone and beat the naive baselines independently. The negative result on RL generalization becomes a reported finding, not a project failure — exactly the framing the chief prose mandates.

---

### 21.3 R2 — Search steering signal too weak/noisy to beat the best-of-N floor

**Likelihood: medium. Capability-correlated: no.** Bounded adaptive-branching and agent-initiated `speculate()` (Section 9) only pay off when the feedback signal that steers branching is informative. Where it is not — giant or flaky test suites, weak F2P signals, the verified-primary quorum saturating early — search adds wall-clock and tokens for no resolve-rate gain, and can underperform plain repeated sampling. This is the well-established result that verifier-guided search loses to repeated sampling at scale because scorers are least reliable early ([Limits of PRM-Guided Tree Search](https://arxiv.org/abs/2510.20272)) and that under strict budgets adaptive-branching can lose to plain sampling ([BAVT](https://arxiv.org/abs/2603.12634)). The adversarial verdict on distributed MCTS is **unsound**; the adversarial verdict on speculative branching is **partially sound** (constant-factor, not exponential, and contingent on prefix reuse). The mitigations honor both.

#### Mitigations

1. **Pruning is always a HINT, never a gate.** Per the Cardinal Safety Contract and the CTDG verdict ("prioritize, don't prune"), any search-derived prioritization may only reorder exploration or set branch priors/budget share. The authoritative prune remains the full regression-prune-by-baseline (Section 10), which re-runs only baseline-passing tests and expands collection-error file keys — execution evidence decides, the search signal only schedules.
2. **Mandatory collapse to verified best-of-N below a feedback-confidence floor.** The branching controller computes a `feedback_confidence` scalar (signal-to-noise of the per-branch reward: F2P presence, suite flakiness estimate, quorum margin). When `feedback_confidence < branching.min_feedback_confidence` (default tuned per benchmark), the engine collapses to adaptive-K verified best-of-N for that task. This is the load-bearing "best-of-N is the floor" guarantee made operational.
3. **Hard growth bounds (reuse FrontierSearch machinery).** Virtual-loss de-duplication, `min_branch_reward`, `max_depth`, and `max_frontier_branching` (the v1 FrontierSearchController constants: `c_puct=1.25`, `virtual_loss=0.15`, `max_depth=6`, `max_frontier_branching=3`, `min_branch_reward=0.12`) bound the tree so an uncontrolled fan-out can never exceed the best-of-N baseline cost. `speculate()` forks feed the *same* ranking/budget machinery (Section 9), inheriting these bounds — the verdict's explicit recommendation.
4. **Reward honesty: verifier Youden gating + flake firewall.** Before any branch reward is trusted as a steering signal, the verifier's Youden index (J = TPR − FPR) must be > 0 on the per-repo signal, and the NDFF flake firewall (v1 Section 12) must not have flagged the deciding tests. A J<0 signal is treated as noise and the task collapses to the floor ([RLVeR](https://arxiv.org/abs/2601.04411)).
5. **"When search hurts" ablation per benchmark.** A standing experiment (Section 20) measures branching-on vs branching-off per benchmark; if branching does not beat the floor on a benchmark, branching defaults OFF for that benchmark and the result is reported.

#### Detectors

- `branching.feedback_confidence` distribution per benchmark.
- `branching.resolve_delta_vs_floor` (search resolve rate minus adaptive-K floor); ≤ 0 ⇒ default off for that benchmark.
- Verifier `youden_index` per repo; flake-firewall positive-evidence flags.

#### Fallback

**Default branching OFF.** Ship the engine + adaptive-K best-of-N + the verification cascades, which are independently proven (repeated sampling 15.9%→56% on SWE-bench Lite, [Large Language Monkeys](https://arxiv.org/abs/2407.21787)). Search is an opt-in amplifier, not a dependency.

---

### 21.4 R3 — Cheap-executor capability ceiling / contract underspecification erases cost savings

**Likelihood: medium-high on hard multi-file SWE. Capability-correlated: partly.** This is the second-highest-likelihood failure and must not be understated. The vendor-agnostic model economy (Section 12) routes cheap workers to narrow sub-roles to cut cost. The documented trap is the **"almost-right trap"**: a thin/cheap executor that needs 3–4 retries plus review can cost *more* than one frontier pass, and the HyperAgent ablation shows that cheapening navigation/multi-file editing causes the *worst* resolve-rate drops ([HyperAgent](https://arxiv.org/html/2409.16299v1)). The adversarial verdict on heavy-orchestrator/thin-executor is **partially sound — default rejected**; the verdict on the model economy as a verification-gated cascade is **adopt-modified**. Cheap executors also raise the false-positive accept rate, stressing verification harder (interaction with R6).

#### Mitigations

1. **Sub-role tiering keeps the hard roles heavy by default.** Per the HyperAgent ablation, only `run`/`verify`/single-tool/narrow-edit sub-roles are eligible for cheap workers. Codebase navigation and multi-file editing stay on frontier (or competent Sonnet-class, *not* "thin") by default. Config: `economy.cheap_eligible_roles = ["run", "verify", "narrow_edit"]`; `economy.heavy_roles = ["navigate", "multifile_edit", "plan", "final_review"]`. This is the difference between the rejected default (uniform thin executor) and the adopted cascade.
2. **Escalate to frontier on the FIRST verify-on-diff failure, with a hard rewrite-cycle cap.** A cheap executor's output goes through the cheap-first verification cascade (Section 13); the *first* verify-on-diff failure escalates that sub-task to a frontier worker. `economy.max_rewrite_cycles` (default small, e.g. 2) caps cheap retries so the almost-right trap cannot run away. This is the FrugalGPT/Aider cascade pattern ([FrugalGPT](https://arxiv.org/abs/2305.05176); [Aider architect/editor](https://aider.chat/2024/09/26/architect.html)).
3. **Per-repo calibrated escalation gate (self-test signal, not intuition).** The decision to trust a cheap worker uses a calibrated confidence signal — self-generated test pass on the cheap output (the code-cascade pattern) with a threshold tuned on in-domain labeled traces, not a hand-picked constant. Poorly-calibrated cheap models are *rejected* outright (the GATEKEEPER caution: bad calibration cannot be fixed by any threshold, [GATEKEEPER](https://arxiv.org/pdf/2502.19335)).
4. **Continuous token-yield monitoring; route heavy when arbitrage stops paying.** The orchestrator tracks **cost per verified-resolved task** (token *yield*), not gross executor tokens or invoice. When `economy.token_yield_cheap < economy.token_yield_heavy` over a sliding window, the cheap path is disabled for that repo/role and the work routes heavy. This directly answers the xRouter brittleness finding ([xRouter](https://arxiv.org/html/2510.08439v1)) — measure yield, never assume the route transfers.
5. **Contract drift defused by execution authority.** "Contract-driven" sub-tasks are accepted on the git diff against real tests (filesystem-as-source-of-truth), never on the contract text or the executor's self-report — so an underspecified contract surfaces as a verify failure (→ escalate), not a silent wrong-accept.

#### Detectors

- `economy.token_yield` per (repo, role, tier); cheap < heavy ⇒ route heavy.
- `economy.rewrite_cycles` per sub-task; hitting the cap ⇒ escalate + flag the contract as leaky.
- `economy.cheap_calibration` (self-test predictive value); below floor ⇒ reject the cheap pairing.

#### Fallback

**Frontier-everywhere.** The economy is opt-in and per-sub-role; disabling it returns to a single competent-model execution shape that is exactly v1's proven configuration. Cost savings are claimed as **net-of-verification on bounded/medium tasks and modest-but-real on hard repo SWE**, never as an unconditional order-of-magnitude — honoring the partially-sound verdict.

---

### 21.5 R4 — Pruning/caching silently degrades the trust anchor (false-negative test drop)

**Likelihood: medium. Capability-correlated: no.** The CTDG + bidirectional pruning layer (Section 10) and the input-hash resume cache (Section 15) both create a path by which a *correct* candidate is silently dropped or a *stale* result is replayed. Static call graphs are quantifiably lossy in dynamic Python (PyCG ~99.2% precision but only ~69.9% recall; eval/getattr/monkeypatch/fixtures invisible, [PyCG](https://arxiv.org/abs/2103.00587)); the pytest item set is not statically enumerable (only `pytest --collect-only` knows it); aggressive cache reuse can replay an answer for changed code. The adversarial verdict on static-AST-CTDG-as-a-gate is **unsound**; the verdict on CTDG-as-prioritizer + dynamic-coverage-prune + full-suite-backstop is **adopt-modified**. The mitigations enforce that split.

#### Mitigations

1. **Dynamic coverage is the ONLY gate; static graph only reorders.** The static CTDG is used solely to *prioritize* the test set (zero false-negative risk — reordering never excludes), accelerating time-to-first-failure. Any actual *pruning* (deselection) uses dynamic per-test coverage (testmon-style block-checksums over coverage.py / per-language tracer), where over-selection is cheap and a miss merely delays feedback. Config: `prune.static_mode = "reorder_only"` (hard-coded; static-as-gate is unreachable).
2. **Full-suite stabilization backstop at the final pre-accept state.** Before a candidate is accepted, the full baseline-passing suite runs once on the final state (the Google/Facebook "stabilization" pattern, [Predictive Test Selection](https://arxiv.org/pdf/1810.05286)). No candidate is ever accepted on a pruned suite alone — this is what makes pruning honest and preserves execution-authority.
3. **Per-repo safety-mode flag.** `prune.safety_mode ∈ {advisory, prune_with_backstop, prune_hard}`, **default `prune_with_backstop`**. `prune_hard` (no backstop) requires explicit opt-in and is never the default, so the orchestrator never silently gambles safety.
4. **Input-hash cache validity (no stale replay).** The resume WAL keys each cached `agent()` result on a hash of `(prompt, model, vendor, scoped_inputs, code_state_hash)` (Section 15). A changed code state changes the hash, so a cached result is *never* replayed for changed code — closing the stale-replay path. Over-select on hierarchy/lockfile/config/seed changes (testmon's bias toward false positives).

#### Detectors

- `prune.would_have_passed_canary`: periodically run the full suite on a pruned-accepted candidate and assert the pruned and full verdicts agree; disagreement is a false-negative alarm.
- Cache `hash_collision` / `stale_replay` counters (should be zero).

#### Fallback

**Full regression prune (v1's proven primitive).** Disable CTDG and dynamic-coverage selection entirely; re-run all baseline-passing tests in chunks of 50 per candidate, exactly as v1 does. The pruning layer is pure speedup; turning it off costs wall-clock, never correctness.

---

### 21.6 R5 — Determinism/replay broken by speculation (timing/sampling in orchestration code)

**Likelihood: medium. Capability-correlated: no.** The speculative/adaptive-branching layer introduces two classically nondeterministic primitives into orchestration: deadline-triggered hedging timers and Thompson-style wider-vs-deeper sampling (AB-MCTS). Either, if placed in the *workflow* (not *activity*) layer, breaks strict replay subtly — the durable-execution rule is that orchestration code must be deterministic ([Temporal](https://learn.temporal.io/tutorials/go/background-check/durable-execution/); [prompt-caching/durable-execution pitfall](https://arxiv.org/html/2601.06007v2)). The adversarial verdict here is **sound_with_caveats**: determinism *can* be preserved, but only with seeding/journaling, and "bit-for-bit" overstates what v1 guarantees (v1 reproduces artifacts + trajectory-via-replay, not from-scratch-identical trees).

#### Mitigations

1. **Seed all samplers from content hashes.** Any Thompson/Random draw in the search-control layer is seeded from a content hash of the node state, reusing v1's `Random(0)` / content-sha discipline. The wider-vs-deeper decision becomes a pure function of recorded evidence, so a re-run from the journal reproduces it.
2. **Journal every node expansion and cancellation by monotonic seq + idempotency key.** Generalize the escrow WAL (CCEDF) pattern (Section 15): each `speculate()`/expand/prune/cancel gets a `seq` and an idempotency key `(run_id, node_id, attempt)`; replay reconstructs the tree in seq order regardless of original timing, and duplicate appends are harmless. The deterministic selection tuple (terminating in content sha1, *never* insertion order) already makes the *winner* order-independent — exploration order cannot change the result given the same accepted set.
3. **Remove wall-clock hedging from the deterministic path, or journal the cancel decision.** On the deterministic path, replace deadline-triggered hedging with budget/evidence triggers; if a wall-clock hedge is kept, the *cancel decision* is journaled so replay reproduces it from the log rather than re-deriving it from timing. Ban `Date.now`/`Math.random`-equivalents in orchestration (shared with R9).
4. **CI acceptance test: kill mid-run → identical resumed trajectory + identical winner.** `ci_r5_kill_resume` kills a speculative run at a random `seq`, resumes from the WAL, and asserts (a) the resumed trajectory matches the recorded one and (b) the selected winner is identical. This is the standard durable-execution acceptance test and is a gating check before the speculation layer ships.

#### Detectors

- `ci_r5_kill_resume` pass/fail (gating).
- Replay-divergence counter: any node whose replayed decision differs from its journaled decision.

#### Fallback

**Sequential, no speculation.** The speculation layer is an optional amplifier over the verified best-of-N floor; disabling it yields a fully deterministic, replayable run with no hedging or sampling in orchestration — exactly v1's determinism posture.

---

### 21.7 R6 — Reward-hacking / contamination amplified by heterogeneous fleets

**Likelihood: high. Capability-correlated: yes (this gets WORSE with stronger models).** Reward hacking scales with capability (GPT-5 exploits impossible tests 76% of the time, [ImpossibleBench](https://arxiv.org/html/2510.20270v1)) and a heterogeneous fleet fans out *more* attempts, multiplying the surface for false-positive passes. Contamination is the dominant validity threat: OpenAI retired SWE-bench Verified for verbatim gold-patch recall; ~32–33% of "successes" can be solution leakage; widespread harness-level cheating is documented across 9 benchmarks ([DebugML/Meerkat](https://debugml.github.io/cheating-agents/)). This is non-negotiable for a mixed-vendor fleet and is defended by v1's anti-cheat/fairness machinery, generalized.

#### Mitigations

1. **Upstream harness is the ONLY publishable number.** APEX-private rescoring is diagnostic and published as a delta (v1's `fairness_audit.json`, FLAG_THRESHOLD 0.02). The fleet's internal selection signals never become the headline.
2. **Hidden/read-only tests in an agent-unreachable grader process.** The grader runs in a sandbox the worker cannot read or write; default test files are hidden or read-only. This is the single most effective anti-hacking lever ([ImpossibleBench](https://arxiv.org/abs/2510.20270)). The grader process is never the agent's process.
3. **Scrub-at-load + leakage scrubber.** Gold fields are *discarded* (not masked) at load time (v1: the task dataclass has no field to hold the gold patch). The sandbox is scrubbed of git reflog/all-branches/origins/tags/future-commit messages (SWE-bench issue #465 loophole); Commit0 uses a true git-history flatten.
4. **Transcript auditing for hack signatures.** Automated flags for the four ImpossibleBench strategies plus mock fakery: `__eq__`/operator overloads, call-count hidden-state tracking, hardcoded test-input returns, test-file edits, exit-code-only "pass," over-mocking. v1's anti-hack ledger (structured fingerprint, false-negative bias, soft-downweight) is the substrate; cross-vendor it is vendor-blind.
5. **Canary impossible-tasks in production monitoring.** Seed a few tasks whose tests contradict the spec; any "pass" is a direct reward-hacking alarm, monitored continuously, not just at eval time.
6. **Private freshness-controlled eval only.** Capability claims are validated on post-training-cutoff issues from controlled repos (SWE-Rebench / SWE-bench Pro pattern), never on public pre-cutoff sets.

#### Detectors

- `anti_hack.signature_hits` per transcript; nonzero ⇒ quarantine the candidate (downgrade-only, per Cardinal Contract).
- `canary.impossible_pass_rate`; any pass ⇒ alarm.
- `fairness_audit.delta` between upstream and private scorers; > FLAG_THRESHOLD ⇒ investigate.

#### Fallback

**Upstream harness only + abstention.** With every soft/diagnostic signal stripped, the system still selects on execution-authoritative, hidden-test, upstream-harness evidence and abstains (first-class outcome) when there is no positive evidence — the floor where reward-hacking has the least leverage.

---

### 21.8 R7 — Harness-leak erases the cross-vendor diversity gain

**Likelihood: medium. Capability-correlated: no.** The whole cross-vendor diversity advantage (Section 3) is contingent on the executor faithfully realizing each vendor's flags/sandbox/schema. Terminal-Bench 2.0 shows 30–50pt same-model swings across harnesses; standardized scaffolds swing 15–21pt ([Terminal-Bench 2.0](https://www.tbench.ai/leaderboard/terminal-bench/2.0)). A single mismapped flag, wrong sandbox mode, or lossy schema delivery can *nullify* the diversity gain — turning a claimed advantage into a loss. The vendor-neutrality verdict is **sound_with_caveats**: feasibility is sound, but "without losing paradigm benefits" must be downgraded to *bounded, declared degradation*.

#### Mitigations

1. **Treat the executor as part of the harness.** The normalized Executor (Section 3) is versioned and tested as a first-class component, not glue. Capability differences are consolidated into one `CapabilityProfile` (internet, native_schema, sandbox_levels, effort, tool-interception, bidirectional_stream, mcp) with explicit graceful degradation to APEX's own floor (no native schema → embed + post-parse; no read-only sandbox → APEX worktree + fcntl lock).
2. **Per-vendor conformance tests asserting the sandbox/schema actually take effect.** `ci_r7_conformance` per vendor asserts that the mapped sandbox mode is genuinely enforced (e.g., a write outside the workspace is actually blocked) and that the declared schema delivery actually produces validated structured output. A leaky mapping fails the gate loudly, before it can silently erase diversity.
3. **Pin + record resolved CLI versions in the RunManifest.** `npm i -g pkg@X.Y.Z`; the resolved version is recorded so a drift that changes harness behavior surfaces in provenance (shared with R8).
4. **Capability-negotiation handshake with explicit declared support.** Model the ACP `initialize` pattern ([Agent Client Protocol](https://agentclientprotocol.com/get-started/introduction)): a vendor declares what it supports; unsupported capabilities degrade rather than crash, and the degradation is recorded so the diversity claim is measured against what actually ran.

#### Detectors

- `ci_r7_conformance` pass/fail per vendor (gating).
- `harness.schema_postparse_failure_rate` per vendor; a spike means schema delivery silently degraded.
- Cross-vendor `resolve_delta` audit: if a vendor's contribution to the fleet collapses after a CLI update, suspect harness leak.

#### Fallback

**Single-vendor floor.** If a vendor's conformance test fails, drop that vendor from the fleet for the run (failure-memory eviction, shared with R8) and run the remaining vendors or a single competent vendor. Cross-vendor diversity is an additive amplifier; its absence returns to a proven single-vendor best-of-N.

---

### 21.9 R8 — Vendor CLI drift / breaking changes

**Likelihood: high (fast-moving). Capability-correlated: no.** npm-distributed CLIs change fast and break the normalization layer silently (e.g., Codex profile semantics broke at 0.134.0; `--full-auto` deprecated). Unpinned `@latest` is a standing liability for any vendor-neutral orchestrator.

#### Mitigations

1. **Version pinning + recorded resolution.** `npm i -g pkg@X.Y.Z`; the resolved version goes into the RunManifest (digest-pinning discipline, v1 Section 17). No `@latest` in any production path.
2. **Two-tier failure memory + self-evicting BackendPortfolio.** v1's distinction between *call-failover* (current-stage reroute on 429/stall) and *backend-level global reroute* (auth/missing-binary/SDK breakage) means a broken backend is isolated, not propagated; the self-evicting portfolio drops a structurally-broken vendor for the run so a transient blip on one vendor cannot poison a healthy heterogeneous fleet.
3. **Adapters fail soft to other vendors.** A vendor that fails its capability handshake or returns malformed output degrades to another vendor or the OpenAI-compatible API path; the run continues. The executor `run_structured_prompt` analog (Section 3) never raises to the caller — every abnormal exit becomes a typed result.

#### Detectors

- `backend.global_reroute_count` per vendor; a spike after a version change ⇒ pin-and-investigate.
- `manifest.resolved_version` diff between runs (drift surfaces in provenance).

#### Fallback

**Other vendor / API path.** Vendor neutrality is itself the fallback: any single CLI breaking is survivable because the fleet and the API path remain. The worst case degrades to a single working backend, never to a dead run.

---

### 21.10 R9 — Engine determinism regression from model-authored control flow

**Likelihood: low-medium. Capability-correlated: no.** Orchestration-as-code (Section 2) lets a model (or a deterministic planner) author the workflow program. If *live, stochastic* model output becomes the un-journaled control flow, it injects nondeterminism into the exact layer v1 keeps pure (deterministic ranking, pure `assign_strategy`/`get_temperature`, bit-identical snapshot SHAs), undermining strict replay and the RunManifest reproducibility guarantee. The substrate verdict is **sound_with_caveats**: the engine win is real, but model-authored control flow must be frozen/journaled.

#### Mitigations

1. **Freeze-then-journal any model-emitted script.** If a model emits the workflow, snapshot + hash the emitted script into the RunManifest and journal it as a deterministic activity, so replay runs over a *frozen* script. Live model output is never un-journaled control flow. Prefer a **deterministic planner** authoring the workflow where possible; the model proposes, the engine freezes.
2. **Ban `Date.now`/`Math.random` equivalents in orchestration code.** A lint/CI rule (`ci_r9_no_nondeterminism`) statically rejects timestamps, RNG without a content seed, and network calls in the *workflow* layer (these belong only in *activity*/worker calls). This is the durable-execution determinism contract enforced as a build gate.
3. **Wire (or remove) v1's flagged `library_enabled` kill switch.** v1 carries a flagged `library_enabled` switch; it must be either fully wired (so model-authored extensions can be deterministically disabled) or removed (so it cannot become a silent nondeterminism path). Carrying a dead flag is itself a regression risk. *(Genuine uncertainty: the ingest flags `library_enabled` as a flagged switch but does not specify its current wiring state; the re-implementer must verify and resolve it.)*

#### Detectors

- `ci_r9_no_nondeterminism` lint pass/fail (gating).
- `manifest.workflow_script_hash`: a control-flow path not covered by a frozen, hashed script ⇒ flag.
- Shares `ci_r5_kill_resume`: a determinism regression in control flow surfaces as a resume divergence.

#### Fallback

**Deterministic planner-authored workflow.** If model-authored control flow cannot be made reliably deterministic, fall back to the IssuePlan-driven, deterministic-planner workflow (v1's existing shape lifted into the engine). The orchestration-as-code flexibility is then offered only over a frozen, hashed, journaled script — never live.

---

### 21.11 Cross-risk interactions and the unifying control

Several risks compound and must be reasoned about together; the register's detectors are designed so the compounding is caught, not masked:

- **R3 × R6:** cheap executors raise the false-positive accept rate, so the model economy makes reward-hacking detection *more* load-bearing. The cheap-first cascade and hidden-test grader (R6) are therefore prerequisites for enabling the economy (R3), not independent features.
- **R2 × R4 × R5:** the speculation layer (R2/R5) and the pruning layer (R4) both create paths to drop a correct candidate. The unifying brake is identical for all three: **execution evidence is authoritative; soft/static/search signals may only reorder or downweight, never exclude pre-execution; a full-suite/verified-best-of-N backstop always exists.** This single invariant, enforced structurally by the Cardinal Safety Contract's downgrade-only review (`accepted` flips True→False only) and the deterministic ranking tuple (every soft key strictly below every execution key), defends all three at once.
- **R1 × R7 × R8:** the controller (R1), harness conformance (R7), and CLI drift (R8) all concern the integrity of the vendor-neutral substrate. They share the RunManifest version-pinning and the failure-memory eviction machinery as common controls.

The single most important property of this register is that **every amplifier degrades to a proven floor.** Search collapses to verified best-of-N (R2); the economy collapses to frontier-everywhere (R3); pruning collapses to full regression prune (R4); speculation collapses to sequential deterministic execution (R5); the learned controller collapses to a bandit + heuristic router (R1); and the whole vendor fleet collapses to a single working backend (R7/R8). Because the floor is itself a defensible, SOTA-competitive system (execution-grounded verify-and-refute over isolated rollouts with Cardinal-Safety selection), APEX-Ω cannot fail *catastrophically* from any single risk firing — it can only fail *gracefully* back toward a system that already beats the naive baselines. That is the register's design contract, and it is what makes the bolder extensions safe to attempt.
