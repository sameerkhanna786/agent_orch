# Stop-Policy Design: Hybrid — SPFG++ primary + LLM governor as a VETO-before-kill

**Design name:** `hybrid-veto`
**Author role:** Principal designer
**Binding principle (user):** kill ONLY when the orchestrator is GENUINELY STUCK / not making
progress. NEVER kill for tokens or wall-clock. Many agents / long wall-clock is FINE while real
progress is happening. The whole decision rests on a CORRECT, hard-to-fool PROGRESS signal.

---

## 0. One-paragraph thesis

Keep the cheap deterministic SPFG+ frontier governor as the **primary, sole authority that can ever
say "cut"** — but make it strictly more conservative by widening its progress vocabulary (SPFG++)
and then interposing a **single journaled LLM governor call that holds a one-shot VETO** at the exact
moment the metric is about to convert a `cut:*` verdict into a halt. The LLM can only answer one
bounded question — *"is there latent real progress the metric missed (e.g. 45/95 files implemented
but not yet collecting)?"* — and its only legal effect is to **DOWNGRADE a kill to one more grace
window of CONTINUE**. It can never force a kill, never accept a candidate, never raise the frontier.
Conservative-by-construction: the system is never *more* aggressive than the metric (the LLM only
ever adds CONTINUEs), it banks acceptance only through execution-grounded gold-pytest, and it still
kills a genuinely dead arm because the veto is rate-limited and itself execution-grounded, so a
truly sterile arm exhausts its vetoes and dies.

---

## 1. Architecture: two cooperating tiers, one direction of override

```
                    every wave boundary (after ctx.parallel barrier)
                                     │
                                     ▼
            ┌──────────────────────────────────────────┐
            │  SPFG++  (cheap, deterministic, PRIMARY)  │   governor.verdict(state)
            │  - all current SPFG+ cut arms             │
            │  - PLUS widened progress vocabulary       │
            │    (failing-set churn, distance-to-pass,  │
            │     diff-size growth) that RESETS patience │
            └──────────────────────────────────────────┘
                                     │
            verdict == "continue" ───┼───► CONTINUE  (no LLM call, ~0 cost)
                                     │
            verdict == "cut:*"  ─────┘
                                     │  (proposed kill — NOT yet a halt)
                                     ▼
            ┌──────────────────────────────────────────┐
            │  VETO GATE (deterministic pre-checks)     │
            │  - veto budget remaining for this arm?    │
            │  - is the cut reason veto-ELIGIBLE?       │
            │  - is there a non-trivial evidence delta  │
            │    since the last veto? (cheap guard)     │
            └──────────────────────────────────────────┘
              no ──► HALT (book the cut:* reason)        yes
                                     │
                                     ▼
            ┌──────────────────────────────────────────┐
            │  LLM GOVERNOR (journaled ctx.ask, ONE     │
            │  veto). Sees an EXECUTION-GROUNDED        │
            │  evidence pack, returns strict schema:    │
            │  {latent_progress: bool, ...}             │
            └──────────────────────────────────────────┘
              latent_progress=false / null ──► HALT (book cut:* reason; tag llm_veto:declined)
              latent_progress=true ──► CONTINUE for a bounded GRACE window (tag llm_veto:granted)
```

The override is **one-directional**: the LLM can only move `cut:* → continue`. There is no path by
which the LLM produces a kill that the metric did not already propose, and no path by which it marks
acceptance. This is the Cardinal-Contract-preserving shape.

---

## 2. The EXACT kill criterion (progress-only)

An arm is KILLED at wave boundary *t* iff **both** hold:

**(A) SPFG++ proposes a cut** — `governor.verdict(state_t)` returns a reason in the cut set
`{cut:no-progress, cut:sterile-diff-streak, cut:nonresult-streak, cut:harness-stall}`. (`stop:*`
honest-no-headroom outcomes and `stop:agent-ceiling` are NOT kills in the "stuck" sense and are not
veto-eligible; they are the explicit opt-in budget/ceiling stops which are off by default.)

**AND**

**(B) the veto does not rescue it** — one of:
  - the cut reason is **not veto-eligible** (see §3.2), OR
  - the **veto budget for this arm is exhausted** (`vetoes_used >= VETO_BUDGET`), OR
  - the **deterministic evidence-delta guard says nothing changed** since the last veto (cheap, no
    LLM call — see §3.3), OR
  - the LLM governor returns `latent_progress == false` (or `null` / schema-miss / infra-fail —
    fail-CLOSED-to-the-metric: an unavailable veto behaves exactly like the metric alone).

Restated as the hard rule the code enforces:

```
KILL  ⇔  SPFG++.verdict ∈ cut:*    AND  NOT veto_rescues(reason, arm_state, evidence_pack)
```

Crucially **(A) is itself 100% progress-only**: every SPFG++ cut arm resets on a *frontier rise*,
where "frontier" is now the **widened** progress set of §4 — gold-pass COUNT, pass_rate tie-break,
collection-error shrink, failing-nodeid-set shrink-or-churn, distance-to-first-pass shrink, and
cumulative implementation-diff growth. None of `tokens`, `wall-clock`, `agents-used`, or
`time` ever appears in (A). The opt-in token floor and agent ceiling remain OFF by default and are
not part of the stuck-kill path.

---

## 3. The LLM-governor VETO: budget, eligibility, grounding

### 3.1 Veto budget (per arm/cell, not per wave)

```
VETO_BUDGET = 2          # APEX_VETO_BUDGET; total LLM vetoes granted per arm over its whole life
GRACE_WINDOW_MEAS = w_meas_effective   # one extra VALID-measurement window per granted veto
GRACE_WINDOW_WALL = w_time             # one extra journaled VALID-measurement wall per granted veto
```

A **granted** veto does NOT reset the frontier (the metric stays honest). Instead it grants a
**single bounded grace window**: the patience arms must elapse *again* (one full
`w_meas_effective` of valid measurements AND one full `w_time` of journaled valid-measurement wall)
before SPFG++ can re-propose the same cut. If the frontier genuinely rises during the grace window,
the run continues normally and the granted veto is *refunded* (it was correct — real progress
appeared, so it does not count against the budget). If the grace window elapses with the frontier
still flat AND no veto budget remains, the arm is killed with reason
`cut:no-progress (post-veto)` — this is the "still kills a dead arm" guarantee (§6).

Why a *budget*, not unlimited: an unlimited LLM veto is exactly the AgentPRM reward-hacking failure
mode (VERIFIED RESEARCH, AgentPRM: "success fell 82%→70% while the PRM's own validation reward kept
RISING") — a soft judge that can always say "keep going" runs forever on a confident-but-broken
arm. The budget caps total LLM-granted life-extension at `VETO_BUDGET × (one grace window)` so the
worst case is bounded and the dead arm provably terminates. The grace window is generous (a full
patience window each) so a real slow-burn winner (jinja-class, the 63-agents-then-solve case) is
rescued; the budget is small (2) so a confident liar gets at most two reprieves.

### 3.2 Veto eligibility by cut reason

| cut reason | veto-eligible? | rationale |
|---|---|---|
| `cut:no-progress` | **YES** | the false-kill surface: a single-load-bearing-bug large repo (pydantic/babel) sits at frontier=0 while really implementing. This is the exact "45/95 files, not yet collecting" case the brief names. |
| `cut:sterile-diff-streak` | **YES** | frozen-diff-but-working: a hard-killed-at-wall rollout reports an unchanged carry diff while real edits happened (tokens=0 artifact). The veto can confirm the underlying worktree is in fact growing. |
| `cut:nonresult-streak` | **CONDITIONAL** | eligible ONLY if the evidence pack shows a non-empty cumulative implementation diff (real bytes on disk). A literal all-`None` / all-empty streak with zero disk bytes is objectively dead and NOT veto-eligible — no latent progress is physically possible. |
| `cut:harness-stall` | **NO** | by construction the arm was *never validly measured* (a wall of collection/harness failures). There is no execution evidence to ground a "latent progress" claim, and an LLM reading harness tracebacks is exactly the transcript-only judge the project's deepest lesson forbids. Harness-stall stays a deterministic metric-only cut. (A SEPARATE remediation path — restart the sandbox / fix env — is the right response, not a veto.) |

### 3.3 Deterministic evidence-delta guard (free pre-filter before any LLM call)

Before spending an LLM call, a pure deterministic check: has the **execution evidence materially
changed** since the last veto for this arm? Compute a cheap evidence fingerprint:

```
evidence_sig = (
    cumulative_diff_bytes_bucketed,      # bucketed size of union of all attempt fs_diffs
    changed_files_count,                 # |union of changed file paths|
    min_errors_so_far,                   # best (lowest) collection-error count
    failing_nodeids_setlen,              # |union/last failing-nodeid set|
    failing_nodeids_set_hash,            # hash of the failing-id SET (detects churn)
    distance_to_first_pass,              # min(errors+failed) over valid measurements
)
```

If `evidence_sig == evidence_sig_at_last_veto`, **skip the LLM call and decline the veto
deterministically** (nothing the LLM could newly discover; the arm is in exactly the state it was
when we last asked, and we already answered). This collapses the worst case to "at most one LLM call
per genuinely-new evidence state per veto-budget slot," which is what bounds cost (§5) and also
prevents the LLM from rubber-stamping a frozen state twice.

### 3.4 What the LLM governor is GROUNDED on (execution evidence, never transcript)

The veto prompt is built from `VetoEvidencePack` — derived ENTIRELY from execution artifacts
already in candidate meta / VerificationResult, never from the agent's self-reported chat. Fields:

- `gold_passed`, `gold_total` → "0/740 gold ids pass" (distance framing, normalized).
- `min_collection_errors` and its trajectory across waves (5091 → 4200 → 4000) — the
  collection-collapse progress signal.
- `distance_to_first_pass` = `min(errors + failed)` trajectory (is the wall to the first pass
  shrinking?).
- `failing_nodeids` set: current size, and the **diff vs. the prior wave's set** (shrinking =
  progress; churning = real code change even when count is flat — the frozen-but-working tell).
- `cumulative_diff_bytes` and `changed_files_count` trajectory (112KB/30 files → 263KB/45 files =
  real work even with frontier=0; directly addresses the tokens=0 / 263KB-diff artifact).
- `failure_excerpts` **delta**: the top import/collection error of THIS wave vs. the prior wave —
  "the error moved deeper" (conftest ImportError advancing 4 layers) is execution-grounded latent
  progress.
- `taxonomy` / `finalization_status` histogram across the streak (how many were timeouts/guard-
  aborts that the tokens=0 artifact made *look* sterile but actually carried a real diff).
- the **cut reason** and which patience arm tripped.

The LLM is explicitly instructed: judge ONLY whether these execution numbers describe a system
mid-implementation (trending toward collect/pass) vs. genuinely frozen; it has NO access to and must
NOT rely on any agent narrative. This is the "intermediate observability layer" the Observability-
Gap research prescribes (surface partial-execution state, runtime signals) rather than the top-layer
transcript that the research shows mis-rates ~18% of the time.

### 3.5 Veto response schema (strict, journaled)

```json
{
  "latent_progress": false,
  "confidence": 0.0,
  "evidence_cited": ["min_collection_errors fell 5091->4000 over 6 waves"],
  "signal": "collection_shrinking | failing_set_churn | diff_growth | error_moved_deeper | none",
  "expected_next_milestone": "first gold collect within ~N more waves | none"
}
```

`latent_progress=true` AND a non-`none` `signal` AND at least one `evidence_cited` item that
references a REAL numeric trend present in the evidence pack → veto GRANTED. Any other shape (false,
null, schema-miss, empty evidence, `signal=none`) → veto DECLINED → HALT. `confidence` is telemetry
only — it never gates acceptance and never adjusts a frontier (per the AgentPRM lesson that a soft
score must never bank progress).

---

## 4. SPFG++ widened progress vocabulary (the primary detector, strengthened)

The audit (GOVERNOR_AUDIT.md:127-142) lists rich execution signals already computed in meta /
VerificationResult but never read by the progress logic. SPFG++ wires the highest-leverage ones in
as **additional frontier-rise resets** (each makes the metric strictly LESS likely to cut — never
more), so the cheap deterministic tier catches most false-kills *without* an LLM call:

1. **distance-to-first-pass shrink**: `min(errors + failed)` strictly drops → frontier rise. Catches
   repos that DO collect but haven't flipped a gold id (GOVERNOR_AUDIT.md:136 #3).
2. **failing-nodeid-set shrink OR churn**: `len(failing_set)` drops, OR the set membership changes
   by ≥ `CHURN_MIN` ids with the same count → counts as a *content-progress* reset of the sterile
   streak only (NOT a frontier rise that banks anything). Different ids failing across waves is
   execution-grounded proof of real code change — the frozen-but-working / single-load-bearing-bug
   tell. (Churn resets sterile/nonresult patience; it does NOT reset the no-progress frontier wall,
   so an arm that only ever *churns* without shrinking still eventually hits `cut:no-progress` →
   veto → death. This keeps churn from being a free perpetual-life exploit.)
3. **cumulative-diff growth**: union changed-files count or bucketed diff bytes strictly grows with a
   non-cheating diff → resets the sterile streak. Distinguishes a frozen-but-working timeout
   (263KB/45 files cumulative) from a truly sterile repeat (GOVERNOR_AUDIT.md:63 + tokens=0 case).
4. **collection-error shrink**: already landed as Fix-1 (`best_min_errors`); retained.
5. **gold_total-scaled patience floor**: `w_meas_effective` and the streak cuts scale with
   `log2(gold_total)` so a large gold universe (740 ids) gets proportionally more attempts before
   any sterile/no-progress cut (GOVERNOR_AUDIT.md:194-200, Fix 4). Repo-AGNOSTIC (function of
   `gold_total`, not repo name) — preserves the pre-registered-rule property.

The division of labor: **SPFG++ resolves the false-kills it can prove cheaply and deterministically**
(error-shrink, set-churn, diff-growth are all integers/sets it already has). The **LLM veto is the
last-resort rescue for the residual** — cases where NONE of those integers moved yet the system is
genuinely mid-implementation (e.g. a giant non-cheating diff that hasn't yet changed any collection
error because the import chain isn't complete). This keeps LLM calls rare.

---

## 5. Cost

- **Common path (frontier climbing or within window):** the LLM governor is NEVER called. SPFG++ is
  a handful of integer comparisons per wave (`governor.verdict`). Cost ≈ **0** added. This is the
  overwhelmingly common case — a healthy run pays nothing.
- **Veto path:** an LLM call fires ONLY when SPFG++ is about to cut AND the cut reason is
  veto-eligible AND the deterministic evidence-delta guard says evidence changed AND veto budget
  remains. Upper bound per arm = `VETO_BUDGET = 2` LLM calls *per distinct evidence state*. With the
  evidence-delta guard (§3.3) a frozen arm spends at most ~1-2 calls total before it stops changing
  and is declined deterministically. So **lifetime LLM cost per arm ≤ ~2-3 short structured calls**,
  each a single `ctx.ask` with a compact numeric evidence pack (no repo upload, read-only, cheap
  model permitted). Negligible against an unbounded coding run that may dispatch dozens of agents.
- **Replay cost:** zero. A journaled veto is a cache HIT on resume (§6) — no re-call.
- **Knob to disable:** `APEX_VETO_BUDGET=0` reduces `hybrid-veto` exactly to today's SPFG++ (pure
  deterministic), so the LLM tier is fully opt-out and A/B-able.

---

## 6. Determinism / replay story

**SPFG++ tier:** unchanged from today — `governor.verdict` is a pure function of journaled inputs and
`_wave_verdict` already records/replays the `(continue, reason)` verdict by POSITION via
`resume_or_run_json` (context.py:695-721). The added widened signals (§4) are all pure functions of
journaled candidate meta, so they replay for free.

**LLM-veto tier:** the veto is a **journaled agent call**, exactly the `ctx.ask` mechanism that the
engine already records + replays deterministically (context.py:1014+; the design constraint
explicitly notes "an LLM-agent decision is ALSO replayable IF its call is journaled like ctx.ask").
Concretely:

1. The veto decision is wrapped in `resume_or_run_json` keyed by `{kind:"veto", wave:n, arm:ns}` —
   the SAME pattern as `_wave_verdict`. First run: build the evidence pack (pure function of
   journaled meta), call the LLM via a forced read-only `ctx.ask` with the strict schema, journal
   `{granted: bool, signal, evidence_sig}`. Resume: replay the journaled record — no LLM re-call,
   identical branch.
2. The **evidence pack is a pure function of journaled artifacts** (candidate meta is already in the
   WAL), so even the *input* to the LLM is reconstructed identically on replay — there is no live
   clock, no volatile counter, no nondeterministic read in the veto path. The wall arm reuses the
   existing journaled `_valid_wall_accum` scalar (no live clock).
3. The grace-window bookkeeping (`vetoes_used`, `grace_until_meas`, `grace_until_wall`) is journaled
   scalar state advanced deterministically per wave, identical to the existing patience counters.

Net: a replay re-emits the exact same CONTINUE/KILL sequence and the exact same booked reason,
whether or not the LLM is re-invoked — the cardinal determinism/replay constraint holds.

---

## 7. False-kill robustness (the three named artifacts + the guard-abort case)

### 7.1 tokens=0 telemetry artifact (codex hard-killed at wall, usage all-zeros, empty final_message, but 263KB/45-file real diff)
- **SPFG++ defense (primary):** the in-cell scorer ALWAYS re-scores the worktree diff regardless of
  `finalization_status` (the worktree is scored even on a timeout/guard-abort), so a hard-killed-but-
  real-diff agent still earns a REAL pytest measurement — `tokens=0` never reaches the progress logic
  as a sterile signal. PLUS the new **cumulative-diff-growth** reset (§4.3): the 263KB/45-file diff
  is a strict growth → resets the sterile streak deterministically, no LLM needed.
- **LLM-veto defense (backstop):** if (and only if) the diff growth somehow didn't reset (e.g. the
  carry diff was frozen so no NEW bytes this wave but the cumulative is large), the evidence pack
  surfaces `cumulative_diff_bytes=263KB, changed_files=45, finalization_status histogram = 8×timeout`
  and the LLM grants a veto on `signal=diff_growth`. tokens=0 is never even shown to the LLM.

### 7.2 frozen-diff-but-working (committed candidate diff looks unchanged while real edits happened)
- **SPFG++ defense:** **failing-nodeid-set churn** (§4.2) — even when the candidate sha is frozen and
  the count is flat, a DIFFERENT set of node ids failing across waves is execution proof of real
  change and resets the sterile streak.
- **LLM-veto defense:** evidence pack carries the failing-set diff and the `failure_excerpts` delta
  ("error moved deeper"); the LLM grants on `signal=failing_set_churn` or `error_moved_deeper`.

### 7.3 collection-collapse (pydantic/babel: gold suite can't COLLECT until much is implemented; frontier pinned at 0 while 45/95 files done)
- **SPFG++ defense:** the **collection-error-shrink** secondary frontier (Fix-1, retained) +
  **distance-to-first-pass shrink** (§4.1) reset the patience arms while the import chain is being
  built, so a run shrinking 5091→4000 errors is never cut.
- **LLM-veto defense (the headline case in the brief):** the residual where the error count is
  *temporarily* flat (a hard import gate not yet cleared) but 45/95 files are implemented — the
  evidence pack shows `gold_total=740, gold_passed=0, changed_files=45/95, min_errors trending down
  over the run, last error moved from module A to module D`. The LLM answers
  `latent_progress=true, signal=collection_shrinking, expected_next_milestone="first collect within
  ~N waves"` → veto GRANTED → one grace window. This is precisely "45/95 files implemented but not
  yet collecting."

### 7.4 guard/policy-abort rollouts (tokens=0 + frozen diff that looked sterile)
- Same as 7.1: the worktree is scored regardless of `finalization_status`, the diff-growth reset
  catches real bytes, and a policy_violation attempt is recorded as telemetry, never punished. The
  veto's `finalization_status` histogram lets the LLM see "these 8 'sterile' attempts were
  guard-aborts that each produced a non-empty diff," exactly the case that produced the false
  no-progress read when the governor cut pydantic at sterile-streak=8.

**Why the veto cannot be fooled into a false CONTINUE forever** (robustness in the OTHER direction):
the veto is rate-limited (`VETO_BUDGET=2`), grace-windowed (each grant buys only one bounded patience
window, not a frontier reset), evidence-delta-gated (a frozen state can't be vetoed twice), and
execution-grounded (the LLM sees numbers, not a confident summary). A confident-but-broken arm
(AgentPRM reward-hack analog) gets at most 2 reprieves of bounded length and then dies.

---

## 8. How it STILL kills a genuinely dead arm

A genuinely dead arm has, by definition, no widened-progress signal: gold frontier flat at 0,
collection errors flat, failing set neither shrinking nor churning, cumulative diff not growing (or
empty), distance-to-first-pass flat. Trace through the policy:

1. SPFG++ proposes `cut:no-progress` (or sterile/nonresult) — the patience window AND journaled wall
   both elapsed with NO frontier rise of any widened kind. (Progress-only: it took as long as the
   arm needed, never a clock cut.)
2. Evidence-delta guard: on the FIRST cut, evidence may have a residual delta → one LLM call. The LLM
   sees a frozen evidence pack (all trends flat, diff not growing) → `latent_progress=false` →
   DECLINE → HALT. **Done in one call.**
3. Even in the adversarial case where the LLM wrongly grants: a granted veto buys exactly one grace
   window. The arm stays dead through it (no frontier rise), the granted veto is NOT refunded (no
   real progress appeared), `vetoes_used` increments. After `VETO_BUDGET=2` grants the evidence-delta
   guard also short-circuits (a frozen arm's `evidence_sig` stops changing → declined without a
   call). The arm is killed `cut:no-progress (post-veto)`.

**Termination bound:** worst-case extra life = `VETO_BUDGET × (w_meas_effective valid measurements
AND w_time journaled wall)` beyond the metric's own cut — finite, bounded, and reached only if the
LLM is wrong twice. A dead arm provably terminates. `cut:harness-stall` is non-eligible so an
all-harness-fail arm dies immediately on the metric with no veto at all.

---

## 9. ctx / governor API mapping (concrete methods/signals + new ones)

### 9.1 Existing signals consumed unchanged (no new computation)
- `governor.verdict(state) -> (continue, reason)` — `apex_omega/engine/governor.py:93`. PRIMARY
  authority; the veto only intercepts its `cut:*` returns.
- `ctx._wave_state()` → `apex_omega/autogen/context.py:670`. Already emits
  `valid_measurements_since_improvement`, `seconds_since_frontier_improved`, `indeterminate_streak`,
  `sterile_streak`, `nonresult_streak`, `attempts_since_improvement`.
- `ctx._wave_verdict(state)` → `context.py:695`. The journaled record/replay point. The veto is
  inserted HERE, between `governor.verdict` and `self._halted = True` (context.py:717-720).
- `ctx.ask(prompt, schema=..., strict=False)` → `context.py:1014`. The journaled, replay-
  deterministic, forced-read-only LLM call used to implement the veto.
- Candidate `meta` fields already written (context.py:958-969): `gold_passed`, `gold_total`,
  `errors`, `empty_diff`, `failing_nodeids`, `failure_excerpts`, `finalization_status`, `ok`,
  `indeterminate`, `pass_rate`.
- `VerificationResult` fields (scoring.py:150+): `passed`, `failed`, `errors`, `total`,
  `missing_expected`, `taxonomy`, `failing_nodeids`, `failure_excerpts`.
- `res.fs_diff` (ExecResult) — currently read only for emptiness (context.py:968); §4.3 reads its
  SIZE / changed-files too.

### 9.2 New SPFG++ signals (additional frontier-rise resets — make the metric MORE conservative)
- In `FrontierTracker.ingest` (frontier.py:178) and `ctx._observe` (context.py:~538-668), add:
  - `best_min_dist_to_pass` = `min(errors + failed)` over valid measurements; strict drop → rise.
  - `failing_set_prev` / `failing_set_hash`: detect set shrink (rise) and set churn (sterile reset
    only). New meta read: `failing_nodeids` (already written, not yet read as progress).
  - `cum_changed_files` / `cum_diff_bytes_bucket`: union over attempt fs_diffs; strict growth →
    sterile reset. New meta field to write: `changed_files_len`, `diff_bytes` (from `res.fs_diff`).
  - `gold_total`-scaled patience: in the `RunGovernor(...)` construction (context.py where governor
    is built), scale `plateau_patience_meas`, `sterile_streak_cut`, `nonresult_streak_cut` by
    `max(1, ceil(log2(max(2, gold_total))))`.
- `governor.verdict` reads no new state keys for these (they only ever RESET existing patience
  counters upstream in `_observe`), so the governor signature is unchanged and back-compatible.

### 9.3 New veto API (small, additive)
- `RunGovernor.verdict` unchanged. Add a thin orchestration method on the context:
  ```python
  def _veto_gate(self, state, reason) -> tuple[bool, str]:
      """Return (rescued, tag). Pure-deterministic pre-checks, then a journaled LLM veto.
      rescued=True downgrades the kill to a grace-window CONTINUE."""
  ```
  Called inside `_wave_verdict` ONLY when `not cont and reason.startswith('cut:')`.
- New journaled record kind `{"kind": "veto", "scoped_inputs": {"wave": n, "arm": ns}}` via
  `resume_or_run_json` (mirrors the `wave` record), storing
  `{granted: bool, signal: str, evidence_sig: tuple, confidence: float}`.
- New journaled scalar arm-state on the context (advanced deterministically per wave, like the
  patience counters): `_vetoes_used`, `_veto_evidence_sig_at_last`, `_grace_until_meas`,
  `_grace_until_wall`, `_veto_refundable`.
- New helper `VetoEvidencePack.from_meta(all_candidates, frontier_history)` — pure function building
  the §3.4 numeric pack from already-journaled candidate meta (no new execution, no transcript).
- New cut reasons emitted (telemetry-distinct, so eval can audit veto behavior):
  `cut:no-progress (post-veto)`, and verdict tags `llm_veto:granted` / `llm_veto:declined` /
  `llm_veto:ineligible(<reason>)` / `llm_veto:budget-exhausted` / `llm_veto:evidence-frozen`.
- Env knobs: `APEX_VETO_BUDGET` (default 2; 0 = pure SPFG++), `APEX_VETO_MODEL` (cheap model for the
  veto call), `APEX_VETO_ELIGIBLE_REASONS` (default
  `cut:no-progress,cut:sterile-diff-streak,cut:nonresult-streak`).

### 9.4 Cardinal-Contract enforcement points
- The veto method returns only `(rescued: bool, tag)` — it has **no** path to set
  `cand.accepted` or call `_checkpoint_accepted`. Acceptance remains execution-grounded gold-pytest
  via `candidate_from_verification` / `verification_from_commit0_evaluation` (scoring.py), untouched.
- Acceptance-checkpointing of the rising PARTIAL frontier (context.py:630-636) is unchanged, so any
  verified solve is banked the instant it passes — a later kill (or a wrongly-declined veto) can
  never lose a real solve.

---

## 10. Risks and mitigations

1. **LLM grants a veto on a hallucinated trend.** Mitigated by: evidence pack is execution numbers
   only; schema requires `evidence_cited` to reference a real numeric trend present in the pack (a
   post-hoc deterministic check can reject a granted veto whose cited trend isn't in the pack);
   grace window is bounded; budget is 2; non-refund on a flat grace window. Worst case = 2 bounded
   reprieves, then death.
2. **Veto cost on a flaky-but-eligible arm that thrashes evidence states.** Mitigated by the
   per-arm `VETO_BUDGET` (not per evidence-state-without-bound) — total grants ≤ 2 regardless of how
   many evidence states appear; the evidence-delta guard only suppresses *redundant* calls, it does
   not grant extra budget.
3. **Reward-hacking drift if the veto model is ever trained/tuned on its own outcomes.** Forbidden by
   design: the veto is a frozen prompt over execution evidence, never a learned PRM in the accept
   loop (the AgentPRM lesson). It can only DOWNGRADE a kill, never bank acceptance, so a drifting
   veto's worst effect is bounded extra compute, never a false solve.
4. **Harness-stall arms that are actually progressing under the hood.** Deliberately NOT veto-
   eligible (§3.2) — the correct fix is env/sandbox remediation, not an LLM reading harness
   tracebacks. Flagged as a separate remediation path, not folded into the stop policy.
5. **Determinism regression if the evidence pack ever reads a live clock or unjournaled state.**
   Mitigated: the pack is asserted to be a pure function of journaled candidate meta + the journaled
   wall scalar; a unit test reconstructs the pack twice from the same journal and asserts equality.
