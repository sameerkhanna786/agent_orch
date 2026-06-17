## 13. Verification & Evidence-Grounded Selection (Verify-and-Refute)

> **Where this sits.** Verification is the load-bearing claim of APEX-Ω (see Section 1, Central Thesis): *capability beyond the base model comes from execution-grounded verify-and-refute, not from agent count.* Generation (Sections 8–12) produces a population of candidate diffs across vendors and budgets; this section is the machinery that converts that **coverage** (the fraction of issues with *some* correct candidate) into **trustworthy resolved issues**, and is the brake that keeps every speculative extension (Sections 9–12, 14) from ever shipping an unverified answer. Without it, the search/economy/blackboard amplifiers degrade into the inference-scaling false-positive trap, where a weak selector lets coverage saturate or *invert* into worse-than-single-shot outcomes ([Limits of Inference Scaling Through Resampling, arXiv:2411.17501](https://arxiv.org/abs/2411.17501)).

This section preserves APEX v1's verification kernel **verbatim** as the hardened substrate, then layers exactly two evidence-supported extensions on top — a swappable generative critic *for discrimination only*, and a single re-usable `verify_and_refute()` primitive callable at multiple granularities — both bound strictly by the Cardinal Safety Contract.

### 13.1 The Cardinal Safety Contract

> **Execution evidence is authoritative. A soft signal may re-rank *within* an execution-verified tier, or *downgrade* an already-accepted candidate — it may NEVER promote an unverified candidate, and it may never exclude a candidate before execution evidence exists.**

Enforced structurally (v1 §13.1): `_apply_evidence_bound_review` flips `accepted` only `True -> False`; `VerificationResult` has no `passed`/`status` field (`accepted` is the gate); the deterministic lexicographic ranking tuple `(combined_score, accepted, public_signal_score, critic_score, size, verification_score, eg_critic_tiebreak, perspective_score, len(changed_files), -cluster_id)` places every soft/learned/LLM key strictly below every execution+critic key, terminating in `-cluster_id` (never insertion order, killing slot-0 bias). Pre-execution scorers (Sections 9–10) set priors/budget only.

**Inverse-violation symmetry.** The contract forbids both directions of error: soft signals cannot promote (false positive), and pre-execution scorers cannot suppress a correct-but-unverified candidate (false negative). This is why static-AST CTDG-as-gate and plan-scoring-as-gate are **rejected** (Section 18): both decide a candidate's fate without execution evidence. Static-AST-as-gate also fails on merit (PyCG ~70% recall; reflection/monkeypatch/fixtures invisible; pytest set not statically enumerable).

**Abstention is first-class.** No positive execution evidence ⇒ `None`, not a least-bad guess (v1 §13.10): all-env-failed → `None`; `cross_candidate_voter` → `winner=None` when all `oracle_score==0`; xval returns `[]` (not a synthetic `0.5`) for singletons; `_selected_result_is_accepted` requires positive evidence; the heuristic fallback stamps `0.35` confidence so it is never mistaken for a real solution.

### 13.2 Cheap-first verification cascade (kept verbatim — never synthesizes a pass)

```
verify_patch(candidate, repo_ctx, baseline) -> VerificationResult:
    score = 0.0
    if not ast_compiles(candidate.changed_py_files):                # 1 SYNTAX
        return VerificationResult(accepted=False, score=0.0, reason="syntax")
    if flake8_available() and lint_clean(candidate): pass           # 2 LINT (gate only)
    if reproduction_passes(candidate): score += 0.35                # 3 REPRO
    prune = prune_by_regression(candidate, baseline, chunk=50)      # 4 REGRESSION
    if not prune.is_valid:                                          #   only baseline-passers,
        return VerificationResult(accepted=False, score=score, reason="regression")
    score += 0.35                                                   #   chunks of 50, in worktree
    xval = build_cross_validation_matrix(candidate, peers)          # 5 NxN ([] for singletons)
    if xval: score += 0.10 * mean(xval)
    score += 0.10 * candidate.pass_rate                             # 6 SCORE + ACCEPT
    return VerificationResult(accepted=positive_execution_evidence(candidate), score=score)
```

Never-synthesize-a-pass countermeasures, carried forward exactly:

| Pathology | Countermeasure | Rationale |
|---|---|---|
| Silent `rc==0` no-op | `errors=1`, never `passed=1` | A no-op must never read as pass |
| Timeout `rc==124` | separate `regression_inconclusive` axis, `+0.15` partial | Honest uncertainty, neither FP nor FN |
| Zero tests / singleton xval | abstain (`[]`), never synthetic `0.5`; `M[i][i]` excluded | No fabricated evidence |

**Test suite is a noisy signal, not an oracle.** SWE-bench has ~11.3% flaky problems; CodeContests has reference solutions failing their own tests ([arXiv:2407.21787](https://arxiv.org/abs/2407.21787)). v1 mitigations stay: NDFF flake firewall declares a flake only on positive evidence; the upstream Docker harness is the only publishable number (v1 §15.4).

### 13.3 Bounding the N×N matrix (cost discipline)

Cross-validation is `O(N²)` sandboxed test runs (v1 §12.6). Bound it **before** building, never by dropping candidates: **Bound 1** — AST two-pass clustering (exact `ast.dump` sha256 fingerprint, then single-linkage merge at `0.95`; `semantic_distance` pools `operator_kinds→control_flow`, `constant_signature→data_structures`) collapses behavioral duplicates; the matrix is built over **cluster representatives** (zero false-negative risk — identical patches can't disagree). **Bound 2** — CTDG test-impact selection (Section 10) skips cell `(i,j)` only when `j`'s diff provably can't affect `i`'s tests, with a **full-suite backstop**. Reordering + dynamic-coverage skip, never static gating.

```
build_cross_validation_matrix(candidates, cfg):
    reps = cluster_representatives(candidates)             # Bound 1
    M = {}
    for i in sorted(reps, key=cluster_id):
        impacted = dynamic_coverage_impact(i.tests)        # Bound 2
        for j in sorted(reps, key=cluster_id):
            if i is j: continue
            if cfg.test_impact_prune and disjoint(j.diff_files, impacted): continue  # cell SKIP
            M[(i,j)] = run_tests_sandboxed(tests=i.tests, worktree=j.worktree)       # 120s/cell
    return M
```

`xval.cluster_before_matrix=true`, `xval.test_impact_prune=true`, `xval.full_suite_backstop=true`, `xval.cell_timeout_s=120`, `xval.sandbox=true`.

### 13.4 Hybrid verification: execution anchors correctness, critic breaks ties

For code (distinct from math), execution-based and execution-free verifiers are **complementary**: execution is the reliable correctness anchor but has *low distinguishability* (cannot separate two patches passing the same tests); execution-free critics give *better discrimination* but are biased toward surface features and gameable. Each plateaus ~42–43% on SWE-bench Verified; the hybrid reaches **51.0%** at 26 rollouts ([R2E-Gym, arXiv:2504.07164](https://arxiv.org/abs/2504.07164)). *Do not choose; combine.* APEX-Ω keeps execution as the gate and adds a generative critic whose only job is **discrimination among equally-passing patches within the execution-verified tier**.

```
hybrid_select(candidates):
    verified = [c for c in candidates if verify_patch(c).accepted]   # execution gate
    if not verified: return None                                     # ABSTAIN
    tier = top_execution_tier(verified)
    if len(tier) == 1: return tier[0]
    scores = {c: generative_critic.score(c) for c in tier}           # within-tier ONLY
    return deterministic_rank(tier, critic=scores)[0]
```

**Best@K vs Pass@K gap is a first-class metric.** SWE-Gym's ORM: Best@16=32.0% vs Pass@16=42.8% — ~11pp, capturing only ~70% of headroom ([arXiv:2412.21139](https://arxiv.org/abs/2412.21139)); CodeMonkeys realized 57.4% vs a 69.8% ceiling ([arXiv:2501.14723](https://arxiv.org/abs/2501.14723)). APEX-Ω reports `best_at_k`, `pass_at_k` (oracle ceiling via held-out gold), and their gap per run — the honest measure of unrealized oracle headroom and the optimization target for Sections 14 and 17.

### 13.5 The swappable generative critic (vendor-neutral, fail-open)

Verifier *strength* is the binding constraint: weak/open critics can fail to beat the no-verifier baseline ([SWE-PRM, arXiv:2509.02360](https://arxiv.org/html/2509.02360v1): closed +5–11pp, open drop to 30–38.8%). The critic is pluggable and degrades to execution-only when weak.

```
@dataclass
class CriticVerdict:
    score: float;  rationale: str;  reliability: float;  abstain: bool   # score is NOT a pass
class Critic(Protocol):
    def score(self, candidate, peers, evidence) -> CriticVerdict: ...
@dataclass
class CriticConfig:
    backend="vendor_neutral"; style="generative"; min_reliability=0.5
    fail_open=True; sees_producer_context=False; timeout_s=1800
```

- **Generative/long-CoT over discriminative scalar PRMs** — robust + data-efficient OOD ([THINKPRM, arXiv:2504.16828](https://arxiv.org/abs/2504.16828): +4.5% LiveCodeBench on ~1K examples; discriminative PRMs fragile under shift).
- **Reliability-weighted, Youden-gated** — below `min_reliability` the critic is ignored (A-HMAD weighting; "SWE-finetuned models are not inherently reliable PRMs").
- **Execution-grounded self-critique over trained per-step PRMs** ([ORPS, arXiv:2412.15118](https://arxiv.org/html/2412.15118v1): implicit 59.9% vs trained 37.0% Pass@1). No bespoke per-token PRM.
- **Fail-open** — error/timeout ⇒ `accept=True`+abstain, fall back to deterministic ranking (v1 §13.8 default-off/fail-open).

> **Hard rule:** the critic never promotes an unverified candidate — `hybrid_select` scores only the already-verified `tier`, and the lexicographic tuple (13.1) keeps every critic key below every execution key.

### 13.6 `verify_and_refute()`: one primitive, many granularities

The capability multiplier is verify-and-refute ("stops saying done when half done"): orchestrator-worker + refutation beat single-agent Opus 4 by 90.2% ([Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)); debate reduces hallucination ([Du et al., arXiv:2305.14325](https://arxiv.org/abs/2305.14325)). v1's three forms (family-disjoint tool-call review, self-play tournament, VerificationAmplifier) lift into one primitive.

```
verify_and_refute(target, granularity, refuters, evidence, cfg) -> Verdict:
    verdict = Verdict(accepted=False, reason="default_refute")        # 1 default-to-refute
    if granularity == "candidate":
        ev = verify_patch(target)                                     # 2 executed evidence
        if not ev.accepted: return verdict                           #   no promotion path
        verdict = Verdict(accepted=True, evidence=ev)
    challenges = parallel([r.refute(target, evidence, redacted=True)  # 3 cross-vendor
                           for r in refuters])                       # 4 no producer ctx
    weighted = sum(c.broke * r.reliability for c,r in zip(challenges, refuters))  # 5 weighted
    if weighted >= cfg.refute_threshold: verdict = downgrade(verdict)   # downgrade-only
    return verdict
```

Invariants: (1) **default-to-refute** (broken until evidence shows otherwise); (2) **grounded in executed evidence** (gate is `verify_patch`); (3) **cross-vendor refutation is a strict diversity gain** over single-family review — different (vendor,model) families decorrelate blind spots (Devlo/TRAE pattern, Section 3); (4) **verifier never sees producer context** (`redacted=True`; family-disjoint enforced at config+hook-build, "a CLI may not be reviewed by its own family") — **reviewer fail-open caveat:** a reviewer error → no veto, so refutation can only strengthen the gate, never weaken it, but a silently-broken reviewer offers no protection (backstopped by reliability weighting + the execution gate); (5) **reliability-weighted, downgrade-only**.

| Granularity | Target | Use site | v1 lineage |
|---|---|---|---|
| `tool_call` | one tool invocation | mid-rollout, turn boundary | family-disjoint tool-call review |
| `finding` | a claim/localization | blackboard contribution (Section 11) | EpisodicMemoryBus |
| `candidate` | a complete patch | selection (13.4) | self-play tournament + VerificationAmplifier |

### 13.7 Anti-reward-hacking (a safety requirement)

Reward hacking scales with capability and generalizes beyond the hack: exploitable coding-RL produced `sys.exit(0)`/pytest-patching that generalized to 12% sabotage, 50% alignment-faking vs 0% baseline ([Anthropic, arXiv:2511.18397](https://arxiv.org/abs/2511.18397)); extensional-only verifiers "admit false positives" ([arXiv:2604.15149](https://arxiv.org/abs/2604.15149)). Since APEX-Ω may self-improve (Section 17), hack-resistance is a safety requirement.

- **Hidden/read-only tests in an agent-unreachable grader** — upstream Docker harness in an isolation tier the worker can't reach/edit; test files read-only, hash-checked; editing assertions/tests/CI forbidden. **Prefer held-out tests**; never benchmark with the same tests used as the selector's verifier.
- **Exploit detection** — `sys.exit(0)`, `__eq__`-override, pytest-monkeypatch/report-patching, exfil. v1 `patch_sanitizer`: `VENDORED_UPSTREAM_ARTIFACT` (stripped, soft signal) vs `GOLD_PROTECTED_TEST` (reject wins) — a protected test under a vendored `testing/` dir still rejects.
- **Transcript auditing** — anti-hack ledger grounds by structured fingerprint (`op+shape+repr`), biases to false-negatives, soft-downweights.
- **Gold-channel destruction, not neuron suppression** — gold fields discarded (not masked) at load; Commit0 true git-history flatten ("block the channel, never the neuron").
- **Verifier Youden gating** — learned verifier admitted only above a Youden-J threshold; TTS skill ≠ RL-reward fitness ([SWE-RM, arXiv:2512.21919](https://arxiv.org/abs/2512.21919)), so re-validate before any RL use.
- **Inoculation prompting** (defer, Section 17) — cut misalignment 75–90% despite >99% hack rate (Anthropic).

### 13.8 Failure taxonomy & abstention (infra ≠ model miss)

8-bucket `FailureClass` + `UNCLASSIFIED`, phase-ordered never-raising `classify()` (v1 §17.2): `charges_apex == {APEX_MISS, UNCLASSIFIED}` only; `HARNESS_BUG` excluded from both `is_environment` and `charges_apex`; `ENV_*`/`NON_DETERMINISTIC` retryable. Phase makes `ModuleNotFoundError` an install-failure during setup but an APEX-miss during test execution. Keeps abstention honest: env failure abstains+retries; real miss abstains, no retry.

### 13.9 Selection pipeline (end-to-end) and config

```
select(candidates, repo_ctx, baseline, cfg) -> Selection:
    verified  = [verify_patch(c, repo_ctx, baseline) for c in candidates]      # 1 cascade
    survivors = [c for c,v in zip(candidates, verified) if v.accepted]
    if not survivors: return Selection(winner=None)                            # ABSTAIN
    reps = cluster_representatives(survivors)                                   # 2 cluster
    M    = build_cross_validation_matrix(reps, cfg.xval)                        #   bounded NxN
    ranked = deterministic_rank(reps, xval=M); tier = top_execution_tier(ranked) # 3 exec-rank
    if cfg.critic.enabled and len(tier) > 1 and critic.reliability >= cfg.critic.min_reliability:
        ranked = reorder_within_tier(tier, critic.score)                       # 4 within-tier
    winner = ranked[0]
    if cfg.refute.enabled and verify_and_refute(winner,"candidate",cross_vendor_refuters,M,cfg.refute).accepted is False:
        ranked = ranked[1:]; winner = ranked[0] if ranked else None            # 5 refute
    winner = apply_evidence_bound_review(winner)                               # 6 True->False only
    record_metrics(best_at_k=..., pass_at_k=..., gap=...)                      #   13.4 metric
    return Selection(winner=winner)                                           #   may be None
```

| Config key | Default | Notes |
|---|---|---|
| `critic.enabled` | `false` | Default-off; downgrade-only ⇒ cannot move headline |
| `critic.min_reliability` | `0.5` | Youden gate; below ⇒ execution-only |
| `critic.backend` | `vendor_neutral` | Any worker family |
| `refute.enabled` | `true` | Default-to-refute |
| `refute.cross_vendor_only` | `true` | Strict diversity over single-family |
| `eg_critic.enabled` | `false` | Triple-gated (v1) |
| `final_acceptance.enabled` | `false` | Downgrade-only, fail-open |
| `abstain_on_no_evidence` | `true` | Cardinal Contract |
| `metrics.report_best_vs_pass_gap` | `true` | Oracle-headroom diagnostic |

### 13.10 Worker/Executor interface (vendor-neutral) and build notes

Critic and refuters are ordinary **workers** via the Normalized Executor (Section 3): opaque external CLIs (Claude Code, Codex, open-weight, mixed) observed via stdout turn-parsing + artifacts, never driven in-process. They consume a redacted prompt (`sees_producer_context=False`) and emit structured `CriticVerdict`/`Verdict` (embed-and-post-parse when no native schema). Evidence/verdicts flow to disk (filesystem-as-source-of-truth), never into an orchestrator context. To build on **Codex** or **Claude Code** you need only: (a) `verify_patch` over a worktree (Section 15), (b) the `Critic` protocol pointed at any worker, (c) `verify_and_refute` with cross-vendor refuters, (d) the deterministic ranking tuple — none vendor-specific.

**Honest limits.** Guarded, not certain: (1) the generative critic helps only with a strong backend, so it ships default-off + Youden-gated; (2) the Best@K–Pass@K gap means real oracle headroom is left unrealized — we report it, not hide it; (3) the test suite is noisy (flaky/contaminated), so every accept is "best available execution evidence," not proof of correctness. The final human gate (read the diff, run the upstream harness) is the ultimate authority — passing tests are not correctness.
