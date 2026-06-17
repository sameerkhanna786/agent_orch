## 12. The Vendor-Agnostic Model Economy

### 12.1 Scope, Thesis, and What This Section Does NOT Claim

The model economy is APEX-Ω's mechanism for spending less compute per *resolved* issue without surrendering the resolve rate that the execution-authoritative kernel buys us. It is an **amplifier**, not a load-bearing pillar: the three properties that actually expand capability — execution-grounded verify-and-refute (Section 13), context isolation (Section 15), and orchestration-as-code (Section 2) — work whether or not we cheapen a single worker. The economy rides on top of them.

The disposition in the canonical mechanism list is precise and we honor it verbatim: **"Model economy as sub-role, verification-gated cascade — adopt-modified."** Concretely:

- We **reject** the heavy-orchestrator + thin-executor shape *as the default execution shape on hard repo SWE*. The adversarial verdict on that claim is `partially_sound` at high confidence, and the model-economy SOTA verdict is blunt: "QUALIFIED NO (risky) for naive thin executors on hard multi-file repo SWE." The [HyperAgent ablation](https://arxiv.org/html/2409.16299v1) shows weakening the Navigator (codebase exploration) and Editor (multi-file editing) roles causes the *worst* resolve-rate drops; [SWE-bench scaling](https://benchmarkingagents.com/swe-bench/) is steep (<13B models score <5% on SWE-bench Verified). A thin executor handed navigation or large multi-file edits regresses toward the cheap-model baseline.
- We **adopt** the cheaper path only as (a) a *sub-role tiering* policy that confines cheap models to the empirically substitutable roles, and (b) a *verification-gated cascade* (try cheap → cheap-first verify on the diff → escalate to frontier on first failure, bounded by a rewrite-cycle cap), so that no cost decision can ever promote an unverified candidate. The cascade is calibration-first, not intuition-first: [FrugalGPT](https://arxiv.org/abs/2305.05176), [RouteLLM](https://arxiv.org/html/2406.18665v4), and the [code-specific self-test cascade](https://arxiv.org/html/2405.15842v1/) all show cheap-then-escalate preserves quality *only when the confidence signal is in-domain-calibrated and a frontier fallback is guaranteed*.

The economy is **opt-in and defaulted off-or-unbounded**, to respect v1's "optimize for SOTA, never for cost" directive and the hard invariant that *a cost cap can never abort a succeeding run*. Turning the economy on is a deliberate operator choice expressed through `budget{}` (Section 16) and the config keys in §12.9.

The unifying safety claim: **execution evidence is both the steering signal and the brake.** Cost shapes *budget allocation and routing priors*; the git diff plus tests decide *acceptance*. The contract and the executor's self-report are never the oracle.

---

### 12.2 The HyperAgent-Tiered Sub-Role Taxonomy

A single "executor" is the wrong unit. The evidence repeatedly separates roles that need sustained, long-context codebase interaction (dangerous to cheapen) from roles that are narrow, well-bounded, and verifiable (safe to cheapen). APEX-Ω's per-rollout stage graph (reproducer → localizer → patcher → test_writer, plus orchestration and selection) maps cleanly onto a tier table.

| Sub-role (APEX-Ω stage / function) | Tier default | Why | Cheapen? |
|---|---|---|---|
| **Orchestrator** (workflow program, FrontierSearch ranking/budget, contract authorship) | `frontier` | Decomposition + search control are where compute demonstrably pays off (Aider architect; Claude opusplan). Errors here compound across the whole tree. | No |
| **Navigator / localizer** (codebase exploration, file/symbol localization) | `frontier` | HyperAgent: weakening Navigator causes the worst resolve drops; long-context env interaction. | No (hard tasks) |
| **Patcher on multi-file / hard edits** | `frontier` or competent-mid (Sonnet-class) | HyperAgent: Editor role degradation is second-worst; the "almost-right trap" (3-4 retries) can cost more than one frontier pass. | No (hard); cautious mid (medium) |
| **Patcher on narrow single-file, well-localized edits** | `cheap` → cascade | Bounded, test-anchored; the contract is nearly complete; safe substitution zone. | Yes, with cascade |
| **Runner / verifier** (run tests, capture diagnostics, regression prune) | `cheap` | HyperAgent: Executor (run/verify) is the *most* substitutable role. v1 already runs verification deterministically; the model here mostly orchestrates shell + parses output. | Yes |
| **Reproducer** (write a failing test that reproduces the issue) | `cheap` → cascade | Narrow, test-anchored output; self-checkable (does it fail at baseline?). | Yes, with cascade |
| **Final-review gate** (frontier review checkpoint after cheap steps) | `frontier` | Mandatory; this is where compounding cheap-model errors are caught before they land. Non-optional (Claude Code guidance). | No |

`Tier` is a vendor-neutral abstraction, *not* a model id (see §12.3). Three logical tiers — `frontier`, `mid`, `cheap` — each resolve to a concrete `(vendor, model, effort)` at command-build time via a routing profile. The 15–60x tier price gaps ([Haiku $0.25/M vs Opus $15/M](https://www.mindstudio.ai/blog/best-ai-model-routers-multi-provider-llm-cost)) make the economy worth building even with imperfect routing.

```python
# Tier is the planner-facing knob; it never names a vendor.
class Tier(StrEnum):
    FRONTIER = "frontier"   # e.g. claude opus / gpt-5.x xhigh — navigation, multi-file, review, orchestration
    MID      = "mid"        # competent editor class (Sonnet-class) — medium patches
    CHEAP    = "cheap"      # Haiku / gpt-5-mini / flash class — run/verify, narrow edits, reproduction

@dataclass(frozen=True)
class SubRolePolicy:
    role: str                       # "localizer" | "patcher" | "verifier" | "reproducer" | "reviewer" | "orchestrator"
    base_tier: Tier
    allow_cheapen: bool             # gate: never True for navigation/multi-file-on-hard
    difficulty_floor: float | None  # if task difficulty >= floor, force base_tier (no cheapen)
    cascade: "CascadePolicy | None" # how to escalate when cheapened; None => single tier
```

The `difficulty_floor` wires the economy to the difficulty estimator (Section 5 / v1 `estimate_difficulty`): on a hard task, `allow_cheapen` is overridden to `False` for the patcher/localizer regardless of policy, collapsing those roles to `frontier`. This is the structural defense against "cheapen the wrong role."

> Honest hedge: there is no published study sweeping every frontier-planner × cheap-executor combination on full SWE-bench Verified; the per-role degradation numbers are *inferred* from the HyperAgent ablation and scaling curves, not measured end-to-end. The tiering above is therefore a calibrated default, and §12.8's evaluation plan is what turns it from default into measured policy.

---

### 12.3 Tier Resolution: Vendor-Neutral by Construction

Tiers resolve through the normalized Executor interface (Section 3), so the economy works on Codex, Claude Code, or a mixed fleet without special-casing. A routing profile maps `(Tier, capability requirements)` to a ranked list of concrete `(vendor, model, effort)` candidates; the existing `resolve_available_llm_config` failover ranking (v1 `llm_routing.py`) then picks the first healthy candidate, so a tier never hard-fails when one vendor is down.

```python
@dataclass(frozen=True)
class TierBinding:
    vendor: str          # "claude_cli" | "codex_cli" | "gemini_cli" | "opencode_cli" | "openai_api"
    model: str           # human alias, resolved to launcher id at command-build time (v1 resolved_cli_model)
    effort: str          # "low" | "medium" | "high" | "xhigh" | "max"
    est_price_in: float  # $/M input tokens — for budget accounting only, NOT for acceptance
    est_price_out: float # $/M output tokens

@dataclass
class RoutingProfile:
    # Each tier resolves to an ORDERED candidate list; failover picks first healthy.
    bindings: dict[Tier, list[TierBinding]]
    same_family_pairs_only: bool = True   # prefer standard<->mini/flash within a family (predictable quality)
```

Two cross-vendor economy levers, both expressed here:

1. **Cost arbitrage.** A heterogeneous fleet can put the heavy orchestrator on one vendor and cheap executors on another, exploiting the 15–60x tier gap. The fleet is also a strength for *diversity* (Section 13): cross-family errors decorrelate, widening coverage more than re-sampling one model ([Devlo 70.2% / TRAE 70.4% on SWE-bench Verified](https://arxiv.org/html/2506.17208v2) both used 3 distinct cross-vendor models + a selector).

2. **Token *yield*, not invoice.** [xRouter](https://arxiv.org/html/2510.08439v1) shows static "expensive-for-hard / cheap-for-easy" routing trees are brittle and do not transfer across providers; the "almost-right trap" means a cheap executor needing 3–4 retries + a frontier rewrite costs *more* than one frontier pass. The economy therefore measures **cost per verified-resolved task** (post-verification, including cross-validation cost), never gross executor tokens.

**Vendor accounting caveats that the cost model must encode** (these are real and break naive arithmetic):

- Claude subscription `-p` / Agent-SDK usage draws a **separate monthly Agent-SDK credit pool from 2026-06-15** ([headless docs](https://code.claude.com/docs/en/headless)). A mixed fleet's "cost" is not one currency; the budget ledger tracks a per-vendor sub-account.
- Token *units* differ across vendors (different tokenizers); the shared budget normalizes to estimated USD, not to a common token count.
- `prefer same-family tier pairs` (standard ↔ mini/flash) for predictable quality; avoid cross-architecture and closely-matched pairs (the pairing-studies caveat).

```python
@dataclass
class CostLedgerEntry:
    vendor: str
    tier: Tier
    credit_pool: str          # "api" | "claude_agent_sdk" | ... — separate sub-accounts
    input_tokens: int
    output_tokens: int
    est_usd: float            # normalized; from TierBinding prices
    role: str                 # which sub-role spent this
    rollout_id: int
    resolved_outcome: str | None  # filled at acceptance: "verified_resolved" | "abstained" | "failed"
# Yield metric = sum(est_usd) / count(distinct issues with resolved_outcome == "verified_resolved")
```

---

### 12.4 Test-Anchored Contracts: The Planner↔Executor Interface

The contract is the portable, vendor-neutral handoff that lets a tiered/heterogeneous fleet cooperate. It is **per-task, not whole-system** — this directly answers the Beck critique ("writing a full spec before implementation wrongly assumes you learn nothing during implementation") and the Fowler/SDD drift problem ("an outdated spec is as harmful as none"). [OpenAI Symphony](https://openai.com/index/open-source-codex-orchestration-symphony/) validates "software as a spec" portability (one spec → compliant TS/Go/Rust/Java/Python), but *notably defines no planner/executor model split*; we add the split and keep the spec scoped to one task.

We reuse v1's existing `core/contract_slice.py` rather than invent a new format. `build_contract_slice(repo_root, issue_plan, localization_artifact, max_files=8)` / `render_contract_slice` already produce a `# Contract Slice` prompt block (failing tests + relevant files + per-file exports/symbols/imports/stubs), and the `TaskBlackboard` already carries contract *obligations*. The economy promotes this from "prompt scaffolding" to "the typed interface between a frontier author and a cheaper satisfier."

```python
@dataclass(frozen=True)
class TaskContract:
    # Authored by the frontier orchestrator/localizer; satisfied by the (possibly cheaper) patcher.
    contract_id: str
    issue_ref: str
    anchor_tests: list[str]          # test ids the patch MUST make pass (the oracle; e.g. graph_target_test_ids)
    must_not_regress: list[str]      # baseline-passing tests that MUST stay green
    target_files: list[str]         # scoped edit surface (max_files=8), from contract_slice
    obligations: list[str]           # required exports/symbols that MUST survive (TaskBlackboard obligations)
    forbidden: list[str]             # anti-patterns (e.g. "do not stub the tested function")
    contract_slice_block: str        # rendered render_contract_slice() text for the worker prompt
    authored_by_tier: Tier
    rewrite_cycle: int = 0           # incremented each escalation; bounded by CascadePolicy.max_rewrite_cycles
```

Why test-anchored: the contract's acceptance criterion is *executable* (anchor tests pass, must_not_regress stays green). This is what makes the cheap path safe — the cheaper a worker is, the more the false-positive accept risk rises ([ImpossibleBench](https://arxiv.org/html/2510.20270v1): GPT-5 exploits tests 76% on impossible tasks), so the contract is checked by execution on the diff, *never* by the executor's claim that it satisfied the contract. Anchor tests are *self-generated* by the frontier author (reproducer/localizer stages) and are the in-domain confidence signal the cascade keys on (§12.5).

> Spec-drift defense, made concrete: because acceptance runs on the git diff against the anchor + regression tests (filesystem-as-source-of-truth, execution-evidence-authoritative), a stale contract cannot launder a wrong patch. If the contract diverges from reality, the diff fails verification and the candidate is rejected or escalated — drift becomes a *loud* failure, not a silent one. There is no separate "spec consistency in CI" burden because the per-task contract lives and dies inside one solve.

**Handoff hygiene** (attacks the ~37% context-loss failure class, [MAST](https://futureagi.substack.com/p/why-do-multi-agent-llm-systems-fail)): the contract is a *structured briefing* (objectives + constraints + prior decisions + evidence as typed fields above), schema-validated at the Executor boundary — never a raw context dump. This is the same schema-validated-message discipline used everywhere else in APEX-Ω; the economy just reuses it for the cheap-path handoff.

---

### 12.5 The Verification-Gated Cascade (Cascade, Not Blind Routing)

This is the core algorithm. It is deliberately a **cascade** (try cheap, escalate on verified failure) and not static up-front routing, because (a) static routing trees are brittle across providers (xRouter), and (b) the cascade reuses APEX's existing cheap-first verification ladder, so escalation is gated by *execution evidence*, not by a guessed confidence threshold.

```
ALGORITHM: execute_subrole_with_cascade(contract, policy: SubRolePolicy, difficulty, budget)
# Honors: never optimize for cost above correctness; never promote unverified;
#         a cap can never abort a succeeding run.

  # 0. Difficulty override: hard tasks collapse dangerous roles to frontier.
  if (not policy.allow_cheapen) or (policy.difficulty_floor is not None
                                    and difficulty >= policy.difficulty_floor):
      tier = policy.base_tier            # e.g. localizer/patcher -> FRONTIER on hard
  else:
      tier = Tier.CHEAP                  # start cheap on the substitutable / easy slice

  cascade = policy.cascade               # ordered escalation ladder, e.g. [CHEAP, MID, FRONTIER]
  cycle = 0
  while True:
      binding = routing_profile.resolve(tier, contract.required_capabilities)  # vendor-neutral
      # SPAWN a worker via the normalized Executor (Section 3). The worker satisfies the contract.
      exec_result = executor.run(worker_prompt(contract, binding), tier=tier, isolation="worktree")
      diff = executor.observe_diff()     # filesystem/git is the source of truth, vendor-blind
      budget.charge(exec_result.usage, binding, role=policy.role)  # cost accounting (separate credit pools)

      # CHEAP-FIRST VERIFY ON THE DIFF (v1 _build_patch_feedback_generator ladder, reused verbatim):
      #   AST syntax -> public-symbol-survival + stub-residue scan -> targeted cached pytest on anchor tests.
      # This NEVER synthesizes a pass: rc==0 => evidence of a pass; rc==124 => regression_inconclusive (NOT pass).
      verdict = cheap_first_verify_on_diff(diff, contract.anchor_tests, contract.must_not_regress)

      if verdict.passed_anchor_tests and not verdict.regressed:
          return Accepted(diff, tier, cycle, exec_result)     # success: stop, do NOT escalate

      # FAILURE: decide whether to escalate. Escalation is the ONLY response to a verified failure.
      cycle += 1
      next_tier = cascade.next_tier_after(tier)               # CHEAP -> MID -> FRONTIER
      if next_tier is None or cycle > cascade.max_rewrite_cycles:
          # No higher tier left, or rewrite-cap hit. Hand the FAILED candidate (+ failure evidence)
          # to the standard rollout failure path / FrontierSearch as a refuted hypothesis.
          return Refuted(diff, tier, cycle, verdict)
      if budget.remaining() <= 0 and not has_any_verified_success():
          # Budget exhausted, no success yet: a cap must NEVER drop a verified pass, but here there is none.
          return Refuted(diff, tier, cycle, verdict)          # loud failure, never a faked pass
      tier = next_tier
      contract = contract.with_failure_feedback(verdict)      # carry the failing diagnostics forward
      contract = replace(contract, rewrite_cycle=cycle, authored_by_tier=tier)
```

```python
@dataclass(frozen=True)
class CascadePolicy:
    ladder: list[Tier]            # ordered, e.g. [CHEAP, MID, FRONTIER]; last element is the HARD frontier fallback
    max_rewrite_cycles: int = 2   # hard cap on cheap->retry loops (the almost-right-trap brake)
    escalate_on: str = "verified_failure_only"  # never escalate on self-report; only on diff/test evidence
```

Five design commitments, each tied to evidence:

1. **Self-generated tests are the in-domain confidence signal.** The cascade escalates when the contract's anchor tests fail on the diff — functional correctness, the natural code-domain signal ([code-cascade paper](https://arxiv.org/html/2405.15842v1/)), not a model's verbal confidence. This sidesteps the calibration trap that sinks naive thresholds.

2. **Calibration is measured, not intuited; reject poorly-calibrated cheap models.** The threshold that *would* be used for any soft scoring (and the choice of which cheap model is admitted to a tier) is tuned on **labeled traces, not intuition** ([GATEKEEPER](https://arxiv.org/pdf/2502.19335); [Gemini-Lite cautionary tale](https://www.mindstudio.ai/blog/best-ai-model-routers-multi-provider-llm-cost) — a poorly-calibrated cheap model cannot close the accuracy gap at *any* threshold). A cheap model whose self-test pass-signal does not correlate with true resolution on held-out traces is **disqualified from the cheap tier** for that role (§12.8).

3. **Hard frontier fallback is guaranteed.** The last rung of `ladder` is always `frontier`. The hardest tier bypasses any soft gate. No task can get stuck at a tier below frontier solely because a cheap model felt confident.

4. **Rewrite-cycle cap is the almost-right-trap brake.** `max_rewrite_cycles` (default 2) bounds the cheap-retry loop; once hit, the candidate is refuted and budget is routed to the next tier or to surviving hypotheses (the early localization-futility gate, Section 9). This is the futility/budget control that [SWE-Effi](https://arxiv.org/pdf/2509.09853) shows is mandatory to avoid the "token snowball" (4x+ on off-track runs).

5. **The cascade never promotes an unverified candidate.** A cheap success is accepted *only* after diff verification; a cheap failure escalates. Soft signals (the generative critic of Section 13) re-rank within the execution-verified tier or downgrade — never promote. This is the Cardinal Safety Contract applied to the economy.

---

### 12.6 The Mandatory Frontier-Review Gate

Cheap models make more errors, and those errors *compound* (Claude Code guidance treats the review checkpoint as non-optional). After any cheap-executor step that produces an accepted-by-cheap-verification candidate, APEX-Ω inserts a **mandatory frontier-review gate** before the candidate is allowed into the selection pool as a verified peer.

```
ALGORITHM: frontier_review_gate(candidate)   # runs only when candidate was produced/verified by a cheap tier
  if candidate.produced_tier == Tier.FRONTIER:
      return candidate                        # frontier output skips the extra gate
  # 1. Full-scope execution re-verification at frontier rigor: run must_not_regress in full,
  #    plus the discriminating-test amplifier (Section 13) if the candidate is otherwise a tie.
  full = full_scope_execution_verify(candidate.diff)        # execution evidence, authoritative
  if not full.accepted:
      return downgrade(candidate, reason="frontier_gate_regression")   # never silently accept
  # 2. Frontier generative critic (discrimination-only): may DOWNGRADE on contradicting evidence,
  #    may RE-RANK within the verified tier, may NEVER PROMOTE an unverified candidate.
  critic = frontier_critic_review(candidate.diff, contract=candidate.contract)
  candidate = apply_soft_signal(candidate, critic)          # re-rank/downgrade only
  return candidate
```

Two notes:

- The gate is **execution-first**: step 1 is a real test run on the diff; the frontier *model* only enters in step 2 as a discrimination-only critic. This keeps the gate cheap-ish (one full verification + at most one frontier critic call) while still catching the compounding-error class.
- The gate's frontier critic obeys the Cardinal Safety Contract: it operates only *within* the execution-verified tier. It cannot rescue a candidate that failed execution; it can only downgrade a passing-but-suspect candidate or break a tie. This is why a cheap executor *raises* verification load (the N×N cross-validation cost hotspot grows) — the savings the economy delivers are **net of verification**, never gross.

---

### 12.7 Worked Control Flow: Easy vs. Hard Task

To make the economy concrete, here are the two regimes end-to-end. (Difficulty is the Section 5 estimate; `economy_enabled` is off by default per §12.9.)

**Easy / well-localized task (`difficulty < patcher.difficulty_floor`, economy on):**

```
plan -> difficulty=0.2
reproducer  : CHEAP   (writes failing test; self-checks it fails at baseline)        [cheap tier]
localizer   : CHEAP-allowed since difficulty < floor; small repo -> CHEAP            [cheap tier]
contract    : authored by orchestrator (frontier) from localization -> TaskContract
patcher     : CHEAP -> cheap_first_verify_on_diff -> PASS anchor tests, no regress   [cheap tier]
frontier-review-gate : full re-verify PASS + frontier critic no contradiction        [1 frontier call]
verifier    : CHEAP (run suite)                                                       [cheap tier]
=> accepted. Cost ~= mostly cheap tier + 1 frontier critic. Yield: 1 resolved at low $.
```

**Hard / multi-file task (`difficulty >= floor`, economy on):**

```
plan -> difficulty=0.8
localizer   : difficulty >= floor => FORCED FRONTIER (no cheapen; navigation is dangerous to cheapen)
contract    : frontier-authored, multi-file target surface
patcher     : multi-file edit on hard task => base_tier FRONTIER (allow_cheapen False for this slice)
              cheap_first_verify_on_diff -> if FAIL, this is already frontier; refute -> FrontierSearch
verifier    : CHEAP (run/verify is substitutable even on hard tasks)                  [cheap tier]
frontier-review-gate : N/A for frontier-produced patch (skips extra gate)
=> The economy here cheapens ONLY run/verify and reproduction. The expensive cognitive work
   (navigation, multi-file editing, review) stays frontier. This is the rejected-default made safe.
```

The contrast is the whole point: the economy's *savings concentrate on easy/bounded tasks and on the run/verify slice of every task*, exactly where the evidence says cheapening is Pareto-safe, and it *withdraws automatically* on the hard cognitive roles where cheapening regresses quality.

---

### 12.8 Calibration & Acceptance of a Cheap Model into a Tier

A cheap model is admitted to a `(role, Tier.CHEAP)` slot only by passing an offline calibration check on labeled traces — never by intuition. This is the gate that operationalizes "reject poorly-calibrated cheap models."

```
ALGORITHM: calibrate_cheap_binding(binding, role, labeled_traces)
  # labeled_traces: held-out (contract, gold_resolved?) pairs for this role, contamination-resistant split.
  for trace in labeled_traces:
      diff = run_worker(binding, trace.contract)
      cheap_signal = cheap_first_verify_on_diff(diff, trace.contract.anchor_tests, ...)  # the in-domain signal
      true_resolved = oracle_resolves(diff, trace.hidden_tests)                          # ground truth
      record(cheap_signal.passed, true_resolved)
  # Calibration quality = how well the cheap self-test signal predicts true resolution.
  precision = P(true_resolved | cheap_signal.passed)
  if precision < MIN_CHEAP_PRECISION:        # e.g. 0.85 — tune per benchmark, NOT a global constant
      DISQUALIFY binding from (role, CHEAP)   # poorly-calibrated -> excluded; route stays frontier/mid
  else:
      ADMIT binding; record the threshold used in the RunManifest for replay
```

- **Use contamination-resistant, standardized-scaffold splits** for calibration and for any "economy preserves quality" claim — SWE-bench Pro public/commercial, Terminal-Bench 2.0, SWE-bench Live, or a private freshly-authored set. *Never* SWE-bench Verified ([OpenAI deprecated it](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/): 59.4% flawed tests in the audited hard subset, verbatim gold-patch recall). A cheap model's pass rate on a contaminated benchmark is partly memorization, so its calibration would be a mirage.
- **Per-benchmark thresholds.** v1 already shows abstention thresholds vary widely across benchmarks; `MIN_CHEAP_PRECISION` and `difficulty_floor` are tuned per eval, recorded in the RunManifest, and replayed deterministically (Section 15).
- The economy's evaluation in Section 20 must report **cost-per-resolved-task net of verification** and the **token-yield** delta versus frontier-everywhere, on a standardized scaffold (harness-dominates-model: [Terminal-Bench 2.0 shows 30–50pt same-model swings across harnesses](https://www.tbench.ai/leaderboard/terminal-bench/2.0)), or the comparison is meaningless.

---

### 12.9 Configuration, Defaults, and Invariants

The economy is governed by explicit config keys. All default to the SOTA-not-cost stance: the economy is **off**, and `budget{}` is **unbounded** (Section 16), so nothing here can ever silently make APEX cheaper-but-worse or abort a succeeding run.

| Key | Type | Default | Effect |
|---|---|---|---|
| `economy_enabled` | bool | `False` | Master switch. Off ⇒ every role runs at its frontier-equivalent tier (v1 behavior). |
| `routing_profile` | RoutingProfile | frontier-only | Maps tiers → ordered `(vendor, model, effort)` candidates. |
| `subrole_policies` | dict[str, SubRolePolicy] | per §12.2 table | Per-role base tier, `allow_cheapen`, `difficulty_floor`, cascade. |
| `cascade.max_rewrite_cycles` | int | `2` | Almost-right-trap brake on cheap-retry loops. |
| `cascade.ladder` | list[Tier] | `[CHEAP, MID, FRONTIER]` | Escalation order; last rung is always the hard frontier fallback. |
| `min_cheap_precision` | float | `0.85` (tune per bench) | Disqualify cheap bindings below this calibrated precision. |
| `frontier_review_gate.enabled` | bool | `True` *(when economy on)* | Mandatory review after cheap steps; cannot be disabled while `economy_enabled`. |
| `cross_vendor_arbitrage` | bool | `False` | Allow heavy-orchestrator-one-vendor + cheap-executor-another fleets. |
| `same_family_pairs_only` | bool | `True` | Prefer standard↔mini/flash tier pairs for predictable quality. |
| `budget` | Budget\|None | `None` (unbounded) | Shared cost ceiling (Section 16); a cap can never abort a succeeding run. |

**Hard invariants the economy must never violate** (these are non-negotiable and cross-cut the whole plan):

1. **A cost decision never promotes an unverified candidate.** Cheapness sets *priors and budget share*; the git diff + tests decide acceptance (Cardinal Safety Contract, Section 13).
2. **A budget cap never aborts a succeeding run.** Caps fire only when *no* verified success exists (v1's `_cumulative_token_cap_exceeded` invariant), and never drop a candidate already in the escrow WAL (Section 15).
3. **Hard tasks keep frontier on navigation, multi-file editing, and final review.** `difficulty_floor` enforces this structurally; the thin-executor-default is rejected.
4. **Frontier-review gate after cheap steps is mandatory** while the economy is on — it cannot be configured away.
5. **Acceptance is vendor-blind.** The diff is the oracle regardless of which vendor/tier produced it; the contract and the executor's self-report are never trusted (filesystem-as-source-of-truth).
6. **Measure yield, not invoice.** All economy reporting is cost-per-verified-resolved-task, net of verification, with separate per-vendor credit-pool accounting (notably Claude's Agent-SDK pool from 2026-06-15).

---

### 12.10 Why Thin-Executor-Default Is Rejected (and What We Keep Instead)

For the record, since this is the most tempting and most dangerous economy shape:

- **The almost-right trap.** A thin executor that produces a plausible-but-wrong patch costs the full verification pass *plus* the rewrite cycles *plus* the eventual frontier escalation — frequently more than one clean frontier pass. xRouter and the model-economy SOTA both warn: measure yield, not invoice.
- **The small-model cliff.** [<13B models score <5% on SWE-bench Verified](https://benchmarkingagents.com/swe-bench/); the resolve-rate curve is steep and frontier-dominated. A thin executor on the cognitive roles sits in the collapse zone for hard tasks.
- **The wrong-role ablation.** HyperAgent shows Navigator/Editor are the *worst* roles to cheapen; only run/verify is safely substitutable. A uniform "cheap executor" ignores this and regresses toward the cheap-model baseline.
- **The false-positive amplifier.** A cheaper executor raises reward-hacking and benchmark-passing-but-wrong rates ([ImpossibleBench](https://arxiv.org/html/2510.20270v1)), stressing verification harder and eating the savings.

What we keep is the *defensible* version the evidence supports: a frontier planner + frontier reviewer spine, cheapening confined to run/verify and narrow well-localized edits, a calibrated verification-gated cascade with a guaranteed frontier fallback and a rewrite-cycle cap, test-anchored per-task contracts as the portable interface, and execution-evidence-authoritative acceptance that neutralizes both spec drift and the cheap executor's added error rate. This is the [Aider architect/editor](https://aider.chat/2024/09/26/architect.html) Pareto win (improved *every* model over its solo baseline, polyglot SOTA at ~14x lower cost) generalized across vendors and confined to the regime where it actually holds.

See Section 13 for the verification/selection machinery this economy depends on, Section 14 for the active controller that learns the routing/budget priors over journaled traces, Section 15 for the determinism + escrow-WAL guarantees that make "a cap can never drop a verified pass" true across restart, and Section 16 for the `budget{}` primitive and broader cost engineering.
