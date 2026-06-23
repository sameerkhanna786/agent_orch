# Eval backend switch: Codex → MetaCode/Avocado?

**Question (user):** Can we run the commit0 eval on MetaCode/Avocado (cheaper Meta-internal)
instead of Codex *without hurting our ability to build/eval experiments* — and does it help
the multi-model publishability goal? Switch only if it does not hurt the eval.

**Verdict: `pilot_first_then_decide`.** Expected landing zone if the pilot passes:
**hybrid** — cheap Avocado for high-volume dev iteration + Codex/GPT-5.5 kept as the paper
anchor + Avocado added as a *disclosed* 2nd backend column. Do **not** switch fully, and do
**not** add it to the paper panel, until a ~18-attempt pilot clears its gate.

Source: workflow `wqbdgbvcn` (9 agents, 3 adversarial reviewers). The reviews materially
**corrected** the discovery/decision phases — the corrected facts below are what stands.

---

## What MetaCode/Avocado are (corrected, high confidence)

- **Not two competitors — a harness + a model.** MetaCode = Meta's agentic coding CLI (fork
  of OpenCode, a Claude Code reimplementation). Avocado (ext. "Muse Spark") = the model that
  powers it. "MetaCode running Avocado" is the default.
- **Both are genuinely agentic** (edit files, run commands/tests, iterate, commit) — not
  one-shot completion. Drivable headless: `metacode run --format json --yolo --model <m>`.
- **Already first-class in our repo.** `apex/core/config.py` has `LLMBackend.METACODE_CLI`;
  `cli_backend.py` builds `metacode run …` and parses it (`_parse_opencode_result`); it is in
  `_OPENCODE_FAMILY_BACKENDS`. So `WorkerSpec(vendor="metacode_cli", model=…)` routes
  end-to-end through `V1Executor` **today** — no new executor class needed.

## Capability (REBASED — the decision phase was stale)

- The decision pinned the floor-fear to **45.2%** (April `avocado-tester`, MSBL-RL/Crucible).
  **That is the old checkpoint, and it was partly a rate-limit artifact** (95%+ of requests
  429'd on a shared 30 RPM Plugboard bucket under ~644 concurrent workers — author Zi Wang
  confirms in-thread).
- **May checkpoint `avocado-staging` = 69.9%**, statistically **TIED with Opus 4.7**
  (5.2pp gap, McNemar χ²=1.8, n.s.), ~11pp behind GPT-5.5 (significant).
- Net: Avocado is **capable-but-slightly-weaker**, *not* a weakling. That is the **good**
  regime for our eval — a weaker-but-real model lands more often in the partial / near-solve
  band where our orchestration knobs (decompose, SARP, REPAIR_EXCERPTS_LOOP) have the most
  headroom, and our continuous scoring banks that as a gradient. More headroom than Codex,
  and ~cheaper.

## The real risk is NOT capability-floor — it's high-QPS rate-limit collapse

- Our OMEGA/converge/hybrid arms run **UNBOUNDED** (`run_ladder.py` `_OMEGA_MAX=1000`, wave
  doubling) — a **high-QPS** workload. That is *exactly* the regime that 429'd Avocado to a
  fake 45.2%. If our fan-out hits the same Plugboard limiter, every Avocado dispatch degrades
  to `infra_nonresult` and the governor books `cut:nonresult-streak` — looks like "too weak,"
  is actually "we DDoS'd ourselves." **This is the single biggest silent signal-destroyer and
  must be probed before any budget.**

## Our harness is partially RESILIENT to the May failure mode (good news)

- May's dominant Avocado failure = "can't close the test loop," but the source posts say that
  specifically = **can't find the right buck test target** in fbsource (a Meta-monorepo
  tooling gap). **commit0 hands the pytest command to the agent** (`commit0_autogen.py:413`),
  so it may not transfer.
- More importantly: our `score_fn` (`commit0_autogen.py:521-569`) **re-runs the gold suite on
  the worktree diff regardless of whether the agent ran tests**. So "made correct edits but
  never verified" still scores a real partial `pass_rate>0` — NOT a hard zero. The genuine
  hard-zero is the **empty-diff / analysis-mode** case (no edits at all). So the kill-switch
  must key on **non-empty fs_diff** and **harness-scored pass_rate**, not on whether the agent
  self-ran pytest.

## Adapter feasibility: LOW effort, with caveats

- Genuinely **vendor-blind**: authoritative artifact is the git worktree diff
  (`v1_executor.py` `_git_diff`); scoring never reads vendor JSON/finalization. The one
  codex-specific line (`.codex` mkdir) is vendor-gated. Effort ≈ hours (Path A: just run with
  a metacode worker spec) — **only for the CLI path**.
- **Caveats:** (1) finalization telemetry is codex/claude-shaped → metacode soft-failures
  collapse to `infra_nonresult`, blunting governor diagnostics (does NOT affect scoring);
  (2) token-usage input/output/cache split is lost; (3) `native_schema=False` → schema is
  prompt-embedded + post-parsed (more JSON-nudge churn); (4) **API-only Avocado would break
  the adapter** — there is no API executor (CLI-only); the "model is just a field" claim holds
  only for the `metacode` CLI binary.

## REAL CODE BUGS to fix before any pilot (verified)

1. **Model-default inconsistency.** `apex_omega/executor/auth_env.py:37`
   `DEFAULT_MODELS["metacode_cli"] = "meta/avocado-tester"` (the weak, 429-poisoned April
   variant w/ the 100%-failure `validate_changes` bug), while `apex/core/config.py:124,132`
   default to `meta/avocado-code-latest`. **The omega path pulls auth_env's default** → an
   unpinned pilot tests the WRONG broken model → falsely pessimistic floor. Repoint to a
   post-May checkpoint AND log the resolved `--model` in the spawned argv.
2. **Confirm `--yolo`** (== `allow_edits` == bypassPermissions) reaches the argv
   (`cli_backend.py:11893-11894`) — the analysis-mode countermeasure. Assert it in argv.

## Access / cost constraints

- **ALLOWLIST_ONLY**, locked to 2 API keys (`D103036522`); routes Plugboard v2, **not** AI
  Gateway → cost-tracking differs. Avocado is `$0` marginal in-tool (Meta 1P, no external API
  bill) but real cost = amortized **BYOC GPU** (Muse Spark: no free capacity).
- Cost numbers are **UNVERIFIED**: $4.18/1M Codex vs $0.95/1M Claude is self-flagged
  medium-confidence/possibly-misattributed; **no Avocado per-token contract rate exists** in
  the knowledge base. Our dominant cost driver is **dispatch-count × token-volume** (unbounded
  arms), so cheaper-per-token can be erased by floor-thrash. **Report cost per-SOLVE, never
  per-token.** Confirm the rate with the Coding Acceleration V-Team (group `acdogfooding`)
  before sizing real budget.

## Publishability (corrected)

- **n=2 (one challenger) does NOT establish "model-agnostic."** A reviewer rejects "agnostic"
  on n=2. Defensible n=2 claim: *"deltas replicate on a second, weaker, different-vendor
  backend."* For a real generality claim, add a **second diverse challenger** (Gemini-3.1-pro,
  already first-class in `config.py`) → anchor(Codex/GPT-5.5) + 2 challengers, require
  **consistent-sign deltas** across all three.
- **Mandatory rigor:** (1) **stratify every delta by reached-regime, report per-model, never
  pool** (weak vs frontier occupy different regions of the scoring space); (2) **asymmetric
  seeds** — challenger n≥4-5 vs anchor n=3, given 73.2% run-to-run determinism (1-in-4 flip);
  (3) **disclose the off-native-harness confound** (MetaCode is −7.2pp vs Claude Code on the
  same model; APEX is a 3rd harness).
- **Hybrid-cheap-dev is fine for engineering iteration speed, NOT as the scientific
  instrument.** Pre-register the orchestration knob set before seeing paper-model numbers;
  re-freeze + re-validate any Avocado-tuned knob on the paper model; **prohibit Avocado-
  specific harness hacks** (test-target hints, edit-forcing prompts) from the published config
  unless they also help the frontier model (frozen-knob firewall).

---

## Pilot plan (~18 attempts, < one full Codex ladder cell)

**Prereqs:** fix the model-default (log resolved `--model` = post-May checkpoint); assert
`--yolo` in argv.

- **Stage 0 — edit-reliability smoke (kill-switch, ~6 attempts).** 1-shot on 2 easy repos
  (voluptuous, jinja) × 3 seeds. Gate on **harness-scored** metrics: fraction non-empty
  fs_diff and fraction reaching pass_rate>0. STOP if >30% empty diffs. *Designed to FALSIFY
  the "can't close the loop transfers to commit0" hypothesis.*
- **Stage 0.5 — rate-limit / high-QPS probe (NEW, the dominant risk).** Run ONE converge cell
  at real fan-out width on avocado-staging; watch for `too_many_requests` / `infra_nonresult`
  spikes. If we 429 ourselves, fix concurrency/limiter before anything else.
- **Stage 1 — regime reach (8 cells).** `converge` on 1 easy + 1 hard/multi-module repo
  (mimesis or babel), ≥4 seeds on the hard repo. Gate: hard repo reaches PARTIAL (>0, <1)
  and/or NEAR-SOLVE-then-plateau (≥0.95, not accepted). Ensure ≥1 repo lands **partial on BOTH
  arms** or deltas are 0 by construction.
- **Stage 2 — orchestration-delta sensitivity (the go/no-go, ~4 cells).** Same near-solve
  repo: `hybrid-diag` (excerpts off) vs `hybrid-diag-excerpts`
  (`APEX_OMEGA_REPAIR_EXCERPTS_LOOP=1`), ≥4 seeds. **GO** requires the excerpts-ON > OFF delta
  to **replicate with the same sign** as on Codex, with multi-seed agreement. Reuse
  `scripts/analyze_phase_ab.py`.

**GO → hybrid** (cheap Avocado dev + Codex anchor + Avocado disclosed 2nd column, ideally with
Gemini as a 3rd). **NO-GO → keep Codex** (don't burn full-ladder budget on flat deltas).
Cost of *deciding* is trivial vs a wrong full switch.

**Confirm before full budget:** (i) Avocado per-token contract rate (Coding Acceleration
V-Team); (ii) which checkpoint `avocado-code-latest` resolves to + that `D101555948`
validate_changes fix is live; (iii) off-native −7.2pp penalty acceptable for a paper column.
