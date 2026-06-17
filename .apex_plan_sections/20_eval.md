## 20. Evaluation Plan & Experiment Matrix

This section specifies how APEX-Ω's claims are tested, what counts as evidence, and the exact baselines, ablations, metrics, and statistical procedures a coding agent (running on Codex or Claude Code workers, possibly mixed) must implement to produce a result a top-venue reviewer will accept. The governing rule, inherited from the Cardinal Safety Contract (see Section 13) and the v1 anti-cheat invariants (see Section 15), is unchanged here: **the upstream Docker harness is the only publishable number; every APEX-private signal is diagnostic-only and reported as a published delta.** The evaluation harness is itself orchestration-as-code (see Section 2) — it runs through the same `solve()` entry point used in production, so benchmark numbers come from the code path a library user exercises, distinguished only by `benchmark_metadata`.

The central claim we must defend (see Section 1 thesis; Section 19 novelty) is narrow and falsifiable: **a vendor-neutral, execution-authoritative dynamic-workflow orchestrator with an open-pool controller wins the cost-quality Pareto frontier on contamination-resistant, execution-verified repo-level SWE tasks, and generalizes to a model pool it never trained on, with no retraining.** Every design choice below exists to make that claim either provable or refutable on a cost-matched axis. We never make a beats-baseline claim off an uncontrolled-budget axis, never rely on a pre-cutoff public benchmark as a capability signal, and never accept exit-code-only passes.

### 20.1 Benchmarks

The benchmark landscape bifurcated in 2025-2026: [SWE-bench Verified](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) is deprecated as a capability signal (OpenAI, Feb 2026: 59.4% of audited hard problems had flawed test cases; all tested frontier models reproduced verbatim gold patches and release-note details; progress flattened 74.9% to 80.9% over six months). We therefore **never report SWE-bench Verified as a capability signal.** It may appear once, in an appendix, only as a contamination/leakage control witness (to demonstrate the gap between a contaminated public set and our contamination-resistant primaries), and is explicitly labeled non-capability.

| Tier | Benchmark | Role | Why | Notes |
|---|---|---|---|---|
| Primary | [SWE-bench Pro](https://arxiv.org/abs/2509.16941) public + commercial | Headline capability | Contamination-resistant (GPL-copyleft public + purchased commercial + held-out); 1,865 tasks/41 repos. Standardized-scaffold scores collapse ~35 pts vs Verified, exposing real orchestration headroom. | Always report public and commercial splits separately; commercial is the honest generalization signal (e.g., GPT-5 41.8% public to 14.9% commercial). |
| Primary | [SWE-bench-Live](https://github.com/microsoft/SWE-bench-Live) | Freshness capability | Monthly auto-updated, multi-language + Windows, post-cutoff tasks via RepoLaunch. Tracks any month's tasks ingested *after* the controller's most-recent training cutoff. | Report per-month cohorts; never pool a month that predates a retrain. |
| Secondary | [Terminal-Bench 2.0](https://www.tbench.ai/leaderboard/terminal-bench/2.0) | Harness-sensitivity sanity | 89 Docker tasks; confirms the Normalized Executor (see Section 3) does not regress shell/agentic fluency across vendors. | Only 89 tasks means wide CIs; 1-2 pt gaps are noise; never a headline. |
| Internal | Private rotating freshness-controlled set | Leakage-proof capability | Freshly authored tasks ingested strictly post-training-cutoff; rotated each cycle; never published task-by-task. | The decisive evidence against contamination skeptics. |

**Private rotating set + leakage scrubber.** We maintain a private, freshly-authored eval set authored after every model in the pool has its training cutoff, rotated each evaluation cycle (retire any task once any pool model's cutoff passes its authorship date). Construction follows the SWE-bench-Live/RepoLaunch model (LLM-driven containerized env provisioning) but with human-authored fail-to-pass tests. Each sandbox the agent sees passes through a **leakage scrubber** before any worker touches it:

```
scrub_sandbox(repo_dir):
  # Destroy every gold/future-state recovery channel; "block the channel, never the neuron".
  git -C repo_dir reflog expire --expire=now --all            # strip reflog
  remove all remotes (origin, upstream, ...)                  # no fetch-the-fix channel
  delete all branches except the working base                 # no future-commit branch
  delete all tags                                             # no release-tag leak
  rewrite/strip commit messages on HEAD..future to placeholder  # no future-commit message text
  for Commit0-style tasks: full git-history flatten           # v1 _flatten_repo_git_history (Section 15.2)
  assert: rev-list --all --count consistent with intended visible history
  assert: no path under repo contains gold_patch/test_patch bytes (structured fingerprint scan)
```

This generalizes v1's load-time gold-field discard (the `SWEBenchTask` dataclass literally has no field to hold `patch`/`test_patch`) and Commit0 history flatten. Grading happens **only** in an upstream-harness grader process the agent cannot reach: hidden/read-only fail-to-pass and pass-to-pass tests live in a separate container/process with no filesystem or network path back to any worker worktree (extends v1's per-rollout `fcntl`-locked isolation, Section 8.2). A worker that synthesizes a "pass" sees errors=1, never passed=1 (v1 silent-`rc==0` to errors=1 rule, Section 12.1).

### 20.2 Baselines

All baselines run under **identical standardized scaffolding** so the only moving variable is orchestration, not model or harness. We follow the Scale-AI standardized-scaffold discipline (the same SWE-Agent/mini-swe-agent shell exposed to every system) precisely because scaffold swings results 15-21 points on a fixed model -- a "beats baseline" claim that conflates scaffold with orchestration is marketing, not science.

| ID | Baseline | What it isolates | Implementation note |
|---|---|---|---|
| B0 | Strongest single vendor/model, single shot | The frontier-model counter-anchor: "is orchestration worth it over one great model?" | Pick per-split (models reshuffle: GPT leads Pro public, others elsewhere). The hardest bar to beat on quality-at-matched-cost. |
| B1 | Cost-equal best-of-N (single vendor) + execution-grounded selection | The inference-scaling floor (see Section 13). | N chosen so total $ approx APEX run $; uses the Cardinal-Contract selector so it is a *fair* best-of-N, not a strawman. This is the floor APEX must never do worse than. |
| B2 | APEX v1 as-shipped (full-cap-16, caps OFF) | The strong incumbent / cost-pathology witness. | Demonstrates adaptive low-K's cost win and that v1's default is the headline pathology (see Section 12 cost hotspots). |
| B3 | Re-trained published learned orchestrator on the *same* pool | Prior-art parity. | [Puppeteer](https://arxiv.org/abs/2505.19591) (NeurIPS'25 RL controller) and/or [AFlow](https://arxiv.org/abs/2410.10762) re-trained on our pool. Required: a reviewer treats learned-controller-that-prunes-for-cost as published prior art, so it must be a baseline we beat, not our headline. |
| B4 | Cross-vendor best-of-N (mixed pool) + selector, **static** routing | Isolates the *learned* controller's marginal value over a heterogeneous pool. | Same mixed pool, same selector, no learned policy -- so any delta over B4 is attributable to the controller, not to vendor diversity alone. |

B0 and B4 together pin down the two confounds reviewers probe hardest: B0 says "did you beat a single frontier model at equal cost," B4 says "is your win just from mixing vendors, or from *learning* how to route them." The minimum reviewer-demanded result (Section 20.6) is dominance over B0, B1, and B3 on a cost-matched Pareto plot **plus** the held-out-vendor split.

### 20.3 Ablations -- one per mechanism

Each ablation toggles exactly one accepted mechanism and **fails open to the heuristic baseline** when disabled, so flipping a switch can never silently move the headline (v1 default-off/fail-open discipline, Section 13.8). Negative controls are first-class: several ablations exist specifically to *demonstrate a safety violation* and confirm that the rejected variant degrades, which is itself a publishable result (cf. [ChromaFlow](https://arxiv.org/abs/2605.14102): more orchestration can hurt).

| # | Ablation | Arms | Hypothesis / expected | Maps to mechanism |
|---|---|---|---|---|
| A1 | Difficulty-adaptive low-K vs full-cap-16 | adaptive-K ON / full-cap-16 | Single biggest cost lever; optimal K often <10 ([Limits of Resampling](https://arxiv.org/abs/2411.17501)). Expect near-equal solve at large cost cut. | Difficulty-adaptive low-K allocation (adopt) |
| A2 | Bounded adaptive-branching | branching ON / collapse-to-best-of-N | Measures search's marginal value *and* the "when search hurts" regime; mandatory collapse below feedback-confidence floor. | Bounded adaptive-branching (adopt-modified) |
| A3 | Early localization-futility gate | gate ON / OFF | Kills the "15/16 doomed" waste; informs allocation, never suppresses a candidate without execution evidence. | Localization-futility gate (adopt) |
| A4 | CTDG test handling | prioritize+dynamic-prune / run-all / **static-CTDG-as-gate (neg. control)** | Reorder + dynamic-coverage prune approx safe with full-suite backstop; static-as-gate must show fault-revealing-test loss leading to solve-rate drop. | CTDG prioritizer (adopt-modified); static-AST gate (reject) |
| A5 | Blackboard | abstracted-negative phased / no-sharing / **raw share-all (neg. control)** | Abstracted negatives preserve diversity; share-all must degrade (~-3.7pp) and homogenize. | Blackboard 2.0 (adopt-modified); share-all (reject) |
| A6 | Model economy | sub-role cascade / heavy-everywhere / **thin-executor-everywhere (neg. control)** | Cheapen run/verify+narrow edits only; thin-executor-everywhere must show worst resolve-drop ([HyperAgent](https://arxiv.org/abs/2506.17208) navigation/multi-file finding). | Model economy (adopt-modified); thin-executor default (reject) |
| A7 | Hybrid verifier | execution+critic / execution-only / critic-only | Critic breaks ties only within execution-verified tier (R2E-Gym ~43% to 51%); critic-only must underperform and violate nothing it cannot promote. | Hybrid verifier (adopt) |
| A8 | **Vendor-mix** | single-vendor pool / heterogeneous pool, controller held constant | Cross-vendor diversity decorrelates hallucinations; expect coverage + selected-rate lift (CodeMonkeys "Barrel of Monkeys" 80.8% coverage). | (vendor,model) as diversity axis (adopt) |
| A9 | **Capability-profile vs one-hot** (H1 falsification) | learned capability/cost profile vectors / one-hot vendor IDs | Profiles are the open-pool enabler ([MoMA](https://arxiv.org/html/2509.07571v1)/[DAAO](https://arxiv.org/html/2509.11079v1)); one-hot cannot route an unseen vendor. H1 fails if one-hot matches profiles on the held-out split. | Open-pool controller (adopt-modified) |
| A10 | **Held-out-vendor generalization** | test-pool contains a vendor/model absent at train time, **no retraining** | The clearest unclaimed NeurIPS-grade gap; controller must route the unseen vendor via profile, fail-open to heuristic if profile is OOD. | Open-pool controller (adopt-modified) |
| A11 | **Cardinal-Contract relaxation** (H2 neg. control) | contract enforced / allow soft signals to **promote** | Must degrade: relaxing promote-ban ships LLM-preferred-but-unexecuted patches; expect solve-rate inversion (the [Inference Scaling Flaws](https://arxiv.org/abs/2411.17501) false-positive mode). | Cardinal Safety Contract (adopt) |

The three negative controls (A4 static-CTDG-gate, A5 raw share-all, A6 thin-executor-everywhere) plus A11 are the load-bearing honesty results: they show that the *rejected* dispositions in the Fusion Ledger (Section 18) are rejected for measured reasons, not taste. A11 specifically is the experiment that proves the Cardinal Contract earns its keep -- if relaxing the promote-ban did *not* degrade, the entire selection-trustworthiness argument would be unfounded.

### 20.4 Metrics

Every metric is reported per-split, per-vendor, and for the mixed fleet, with confidence intervals. The cost-matched axis is mandatory on every comparison plot.

| Metric | Definition | Why it matters |
|---|---|---|
| Solve rate | upstream-harness resolved % (fail-to-pass flips green AND pass-to-pass stays green) | The only publishable capability number; never exit-code-only. |
| Tokens / solve | total prompt+completion tokens / resolved tasks | Cost lever transparency; counts cache-read vs cache-creation separately. |
| Cost per verified-resolved (token yield) | $ (or tokens) / verified-resolved tasks | The honest efficiency number; the headline alongside solve rate. |
| Wall-clock p50 / p95 | per-task latency distribution | `pipeline()` (Section 9/16) should cut p95 from sum-of-slowest-per-stage toward slowest-single-chain. |
| Cost-quality Pareto frontier | resolved-rate vs $ and vs tokens | The reviewer-demanded plot; all dominance claims live here. |
| Best@K vs Pass@K gap | oracle coverage (any sample correct) minus realized selected-correct | Quantifies headroom the verifier leaves; CodeMonkeys 69.8% coverage vs 57.4% selected = 12.4pt gap to close. |
| Abstention rate + precision | fraction abstained; fraction of abstentions that were truly unsolvable-by-us | First-class outcome (Section 13.10); abstention beats a false-positive ship. |
| Futility-detection rate | fraction of doomed runs aborted before the patch loop | A3 gate efficacy; cf. [SWE-Effi](https://arxiv.org/abs/2509.09853) token-snowball avoidance. |
| Cache-hit rate (fleet SLO) | cache_read / (cache_read + cache_creation) tokens | Makes prefix-stable assembly's ~90%-off-cached-reads claim (Section 16) measurable; tracked as a fleet SLO. |

**Cost matching is the spine.** No comparison is permitted on a non-cost-matched axis. When two systems differ in spend, we either (a) match B1/B4's N so total $ equals APEX's, or (b) report only the Pareto frontier (resolved-rate vs $) and claim dominance only where one frontier strictly dominates another. "Beats B0 while spending 5x the tokens" is not a result; it is rejected at write-time by a harness assertion that refuses to emit a delta claim unless `abs(cost_a - cost_b)/max(cost_a,cost_b) <= cost_match_tol` (default 0.10) or the claim is explicitly framed as a Pareto-dominance claim.

### 20.5 Statistical rigor

```
config (eval/stats.yaml):
  cost_match_tol: 0.10              # |dcost|/maxcost ceiling for a point comparison
  flaky_prefilter_reruns: 50       # rerun each task to estimate flake rate
  flaky_youden_min: 0.0            # gate self-improvement on verifier Youden's J > 0 (strict >)
  ci_method: "bootstrap_paired"    # paired bootstrap on matched task sets
  ci_level: 0.95
  seeds_per_task: 3                # multiple seeds where engine allows; temp-0 not bitwise reproducible
  contamination_audit: required    # appendix gate; run fails to "publishable" without it
```

- **Cost-matched axis always.** Any uncontrolled-budget delta is rejected (see 20.4). This is non-negotiable and enforced in code.
- **Paired comparisons.** All A-vs-B deltas are computed on the *identical* task set; we report a paired bootstrap CI on the resolved-rate delta, not two independent CIs. Paired design removes per-task difficulty variance, which dominates SWE-bench-style variance.
- **Confidence intervals on deltas.** Every headline number ships with a 95% CI; Terminal-Bench 2.0's 89 tasks (+/-2-3pt) means 1-2pt gaps there are explicitly labeled "within noise."
- **Flaky pre-filter.** Before any task enters the scored set, rerun it ~50x under the unmodified base agent; tasks whose pass/fail is non-deterministic are quarantined (extends v1's NDFF flake firewall, which declares a flake only on positive evidence and never re-runs a real failure, Section 15). Any self-improvement loop (Section 17) is gated on the verifier's Youden's index J > 0 -- a verifier no better than chance may not steer learning.
- **Determinism via artifact replay, not token streams.** Temp-0 is **not** bitwise reproducible across hosted APIs (batch non-invariance); we do not claim bit-reproducible agent *output*. Instead we reproduce *artifacts*: the RunManifest (git sha, python/platform, seed, redacted `APEX_*` env, model ids, Docker digests, harness versions) plus Docker digest pinning plus `apex replay-deterministic --verify` re-apply the recorded diffs and re-run verification, asserting the *resolved/unresolved verdict* reproduces. Multiple seeds per task are used where the engine allows to bound run-to-run variance, but the determinism *claim* is scoped to environment + ordering + verdict, never to the model's token stream (Section 15).
- **Contamination-audit appendix (required).** A run is not "publishable" until the appendix witnesses: leakage-scrubber assertions passed on every sandbox; the verbatim-gold-recall probe (prompt each pool model with the issue and check for gold-patch recall) on every primary task; the private-set authorship-vs-cutoff dates; and the Verified-vs-primary gap as a contamination witness. This operationalizes the field consensus that contamination is the dominant validity threat.

### 20.6 The minimum reviewer-demanded result

A single composite figure + table, and nothing less, clears the bar:

1. **Cost-matched Pareto dominance** on a contamination-resistant benchmark (SWE-bench Pro public *and* commercial; corroborated on a SWE-bench-Live month and the private set): the open-pool controller's resolved-rate-vs-$ frontier **strictly dominates** B0 (strongest single model), B1 (cost-equal best-of-N), and B3 (re-trained published orchestrator on the same pool), with paired bootstrap CIs on the deltas.
2. **Held-out-vendor split, no retraining** (A10): at test time the pool contains a vendor/model absent at train time; the controller routes it via learned capability/cost profiles and still dominates B4 (static cross-vendor routing). This is the open-pool generalization claim -- the unclaimed gap in the literature (Section 19) -- and is what makes the contribution NeurIPS-grade rather than an engineering note.

If (1) holds but (2) fails, we have a strong systems result but **not** the novelty claim, and the paper is reframed accordingly (honest framing per Section 1: search/economy as bounded amplifiers; execution evidence as steering signal and brake; best-of-N as the floor). We never overclaim open-pool generalization off a same-pool result.

### 20.7 Standardized scaffolding & grader isolation

To isolate orchestration from model/harness, every system (B0-B4 and all APEX arms) runs through one **Normalized Executor** interface (see Section 3) exposing the same minimal worker contract:

```
WorkerSpec:
  vendor: str            # "codex" | "claude_code" | ... (mixed pools allowed in one run)
  model_id: str
  capability_profile: vec # learned; NOT a one-hot id (A9)
  cost_per_1k_in/out: float
  sandbox_caps: set      # negotiated; degrade-not-crash if a cap is absent

run_worker(spec, task, scaffold) -> WorkerResult:  # identical scaffold for all vendors
  # scaffold = same shell/tool surface for every vendor (Scale-AI standardized discipline)
  # results to disk, never re-flooded into an orchestrator context (filesystem-as-source-of-truth)
```

The grader is a **separate, agent-unreachable process**: hidden/read-only tests live outside every worker worktree, behind the per-rollout `fcntl` lock and network-default-none sandbox (Section 8/18). A worker can run *its own* tests (CTDG-prioritized, Section 10) but can never read or mutate the hidden fail-to-pass/pass-to-pass suite, and the upstream Docker harness -- pinned by digest -- produces the only published verdict. This closes the "benchmarking with the tests used as the verifier" pitfall (hold-out evaluation tests from selection tests) and the exit-code-only-pass pitfall (the harness asserts the specific fail-to-pass set flips and pass-to-pass holds; a bare `rc==0`, a `rc==124` timeout, or a zero-collected run never counts as a pass -- v1 counting precedence, Section 12).

### 20.8 Artifact replay & reproducibility package

Each published run ships a reproducibility bundle so reviewers can re-derive verdicts:

- **RunManifest** + Docker digest pins + redacted env + model/harness versions.
- **Per-`agent()`-call WAL** (the journaled, restart-survivable resume substrate, Section 15/17): the durable record of every worker dispatch, its inputs (input-hash keyed), and its emitted artifacts, doubling as off-policy credit substrate for the learned controller.
- **Escrow WAL** of confirmed-resolved candidates (idempotent, seq-ordered, latest-wins) so a preempted-but-correct run is never lost from the published coverage (the v1 Commit0 loss fix, Section 16.5).
- **`apex replay-deterministic --verify`**: replays recorded worker I/O and re-runs the harness, asserting the resolved/unresolved verdict reproduces. We state plainly that this guarantees *environment + ordering + verdict*, not the agent's token stream -- temp-0 is not bitwise reproducible across hosted APIs, so we reproduce artifacts (diffs + re-run verification), never token streams.

This package, together with the contamination-audit appendix (20.5) and the cost-matched Pareto + held-out-vendor result (20.6), is the complete evidentiary surface APEX-Omega stakes its central claim on: every dominance assertion is paired, cost-matched, CI-bounded, contamination-audited, and replayable -- and every rejected mechanism (Section 18) has a negative-control experiment showing it degrades.
