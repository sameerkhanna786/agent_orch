# Stop-Policy Design — LLM Governor Agent

**Design axis:** *A read-only LLM governor AGENT that periodically (or at a plateau-candidate)
inspects the arm's REAL execution evidence and judges "is this making reasonable progress toward
the goal? CONTINUE / KILL." It is journaled like `ctx.ask` (replay-deterministic), execution-
grounded (not transcript-only), cheap (gated, infrequent), and can NEVER accept.*

**Name:** `llm-governor-agent`

**One-line thesis:** Keep the deterministic SPFG+ frontier governor as the **floor and ceiling** of
the decision; insert the LLM only in the **narrow, expensive-to-decide middle band** where the
deterministic signals are *ambiguous* (a candidate plateau the frontier cannot resolve), and let the
LLM **only ever VETO a deterministic kill** ("this looks like real progress, keep going") — never
*originate* a kill on its own soft judgment, and never accept. This makes the LLM strictly a
**false-kill-reducer**, which is exactly the residual problem the audits identified, while the
genuinely-dead-arm guarantee stays anchored in execution-grounded deterministic backstops.

---

## 0. Why this shape (and not "LLM decides everything")

The VERIFIED RESEARCH gives two hard constraints that dictate the architecture:

1. **A soft/learned progress score can climb while real solving falls, and ensemble-disagreement
   detectors FAIL to catch it** (AgentPRM: success 82%→70% while the PRM's own validation reward kept
   *rising*; arxiv 2502.10325). → An LLM judge must **never be the sole authority that ends a run on
   a soft "no progress" read**, and must **never bank acceptance** — that is already the Cardinal
   Contract, and this design hard-honors it.

2. **A transcript/diff-only judge mis-rates ~18% of the time** (execution-free code critics: ~1 in 5
   build-status judgments wrong; openreview gDWkImLIKd) and **the Observability Gap** proves
   output-only feedback is symptom-correcting, not cause-identifying. → The LLM must be fed
   **intermediate execution-state evidence** (pytest counts, real diffs, frontier history, failing-id
   deltas), NOT the agent's own conversation/self-report.

3. **Mature harnesses INTERVENE before they KILL** (OpenDev doom-loop detection + system-reminder
   nudge are *separate layers* from the hard iteration cap; arxiv 2603.05344). → The LLM verdict
   space is **CONTINUE / NUDGE / KILL**, with KILL reserved, and the deterministic backstop is the
   true cap.

So the LLM governor is positioned as a **veto-and-nudge layer over the deterministic plateau cut**,
not a replacement. It can *delay* a deterministic kill (bounded number of times) when it sees
execution-grounded progress the integer frontier missed; it can *escalate* a nudge; it can *confirm*
a kill. It can never **lower** the genuinely-dead-arm guarantee below the deterministic backstop.

---

## 1. The exact KILL criterion (progress-only)

The arm STOPS with reason `cut:no-progress-llm-confirmed` **iff ALL** of the following hold at a
*plateau-candidate* evaluation point (defined in §3):

> **D (deterministic trigger):** the existing SPFG+ deterministic plateau cut would fire — i.e.
> `governor.verdict(state)` returns a `cut:no-progress` OR `cut:sterile-diff-streak` verdict
> (computed exactly as today over `attempts_since_improvement`, the frontier dual-AND, and the
> sterile/nonresult streaks). The LLM is **only ever consulted when D is already True.** A climbing
> frontier never reaches the LLM at all (the deterministic governor returns `continue` and we never
> spend the call).
>
> **E (execution-grounded ambiguity gate is exhausted):** the *deterministic secondary progress
> probe* (§4 — failing-id churn / errors-shrink / collected-rise / diff-growth over the patience
> window) shows **no** strict improvement either. If E shows progress, we DON'T even call the LLM —
> the deterministic Fix-1-extended secondary frontier already resets the clocks. The LLM is reserved
> for the residual case where *every cheap deterministic signal is flat* but we still want a
> sapient second opinion before paying the cut.
>
> **L (LLM veto budget exhausted OR LLM confirms dead):** the LLM governor, given the execution
> evidence packet (§5), either (a) returns `verdict="kill"` with a grounded rationale that passes
> the anti-over-claim check (§6), OR (b) returns `verdict="continue"` but the **per-arm veto budget
> `V` (default 2) is already spent** — a vetoed plateau that re-plateaus more than `V` times is cut
> regardless of the LLM, so a confidently-wrong "keep going forever" LLM can never run the arm
> forever.

In words: **kill only when (1) the deterministic frontier is flat, AND (2) every cheap deterministic
secondary progress signal is also flat, AND (3) an execution-grounded LLM either agrees the arm is
dead or has already used up its limited benefit-of-the-doubt vetoes.** Nothing in this criterion
reads tokens or wall-time as a *reason to kill* — they appear only as the deterministic dual-AND
*wall* arm, which (per the existing design) is a **journaled VALID-measurement nominal increment, not
a live clock**, and is purely a debounce so we don't ask the LLM after a single flat sample.

### The genuinely-dead-arm guarantee (so we never run forever)

Two hard deterministic backstops are **NOT vetoable by the LLM**, so a dead arm always terminates:

- `cut:harness-stall` (indeterminate_streak ≥ INDET_CEIL 24): a wall of harness/scorer failures =
  the arm was never measured. The LLM has no execution evidence to reason over here (there are no
  valid measurements), so it is **not consulted**; this cut fires deterministically.
- **Veto-budget exhaustion** (the `V`-cap above) and the **legacy attempts-since-improvement backstop
  (64, raised to a `gold_total`-scaled floor per §7)**: a confidently-wrong LLM that keeps vetoing
  gets overruled after `V` re-plateaus, and even with `V` vetoes the arm cannot exceed the
  attempts backstop without a real frontier rise. So the *worst case* added lifetime of a truly dead
  arm is bounded: `V × (one patience window)` extra attempts beyond today's behavior, then a forced
  cut. With `V=2` and the default window that is a small, bounded, fully-deterministic ceiling.

---

## 2. Determinism / replay story

The LLM governor call is **mechanically identical to `ctx.ask`** (`context.py:1014`), which is
already journaled and replay-deterministic through `self._engine.agent(...)` →
`resume_or_run_exec`. Concretely:

- The governor call is dispatched through a new `ctx._governor_judge(...)` that wraps a **forced
  read-only** `self._engine.agent(ScopedTask(..., sandbox="read-only", schema=GOVERNOR_SCHEMA), ...)`
  with a **stable node id** `f"{self._node_ns}gov{plateau_index}"` and **scoped_inputs keyed on the
  exact execution-evidence packet hash** (see below). The engine already journals + replays agent
  calls by node id, so on resume the recorded JSON verdict is replayed as a **cache HIT** — the LLM
  is *not* re-invoked, and the control-flow branch (continue/kill) is byte-identical.
- The verdict is then **double-journaled** the same way the deterministic verdict is today: it flows
  into `_wave_verdict`'s `resume_or_run_json` envelope (`context.py:707`) under the existing
  `{"kind":"wave", ...}` record, so the *final* `(continue, reason)` decision is journaled by
  position **independently** of the agent call. Even if the cached agent reply were ever lost, the
  wave verdict that consumed it is still replayed verbatim. Belt and suspenders.
- **No live clock, no RNG, no volatile counter** enters the LLM prompt. The evidence packet is built
  **purely from journaled inputs** (WAL score records, candidate meta, frontier history,
  `_wave_state()` — all of which are already deterministic-on-replay) plus the deterministic
  `seconds_since_frontier_improved` *nominal* scalar (journaled VALID-measurement increment, not
  epoch time). Two replays produce the identical packet → identical scoped_inputs hash → cache HIT.
- **Plateau index, not time, is the cadence key.** The governor is invoked at the `k`-th
  plateau-candidate (a monotone counter `self._gov_counter`, journaled like `self._wave_counter`), so
  the *sequence* of governor calls is a pure function of the journaled attempt/measurement sequence.

This gives the project's required property verbatim: *"an LLM-agent decision is ALSO replayable IF its
call is journaled like ctx.ask — so an LLM governor CAN be deterministic-on-replay, but adds cost +
must be cached."* Here it is journaled like `ctx.ask` **and** the consuming verdict is journaled like
the wave decision, so it is cached at two layers.

---

## 3. Cadence (when the LLM is consulted)

**Never on a healthy run, and never on a clearly-dead run.** The LLM is gated behind two cheap
deterministic predicates so it fires only in the ambiguous middle:

1. **Healthy → skip.** Each wave, `governor.verdict(state)` runs first (free, pure). If it returns
   `continue`, the LLM is never called. A climbing frontier resets the patience arms and never
   reaches a plateau-candidate. **Cost on a winning run = 0 governor calls.**
2. **Plateau-candidate → maybe.** When `governor.verdict` returns a `cut:no-progress` /
   `cut:sterile-diff-streak` token, we treat it as a *candidate*, not a kill. We then run the **cheap
   deterministic secondary probe** (§4). If the probe shows ANY strict secondary progress, we
   *reset the clocks deterministically and continue* — still no LLM call. **Cost on a slow-but-real
   run with a detectable secondary signal = 0 governor calls.**
3. **Ambiguous plateau → one LLM call.** Only when D is True *and* the secondary probe is flat do we
   spend **one** journaled governor call. After a continue-verdict, the LLM is **debounced**: it will
   not be consulted again until at least `w_meas_effective` more VALID measurements have accrued
   (one fresh patience window), so a single plateau costs at most one call per window.
4. **Hard floors are never consulted** (harness-stall, attempts-backstop, agent-ceiling) — those cut
   deterministically.

Net cadence: **≤ `V+1` governor calls per arm over its entire lifetime** (one per re-plateau up to
the veto budget, plus the final confirming call). On the 86-cell ladder that is a handful of
read-only calls per cell *only on cells that plateau*, i.e. a negligible fraction of total compute.

---

## 4. The deterministic secondary progress probe (the cheap gate before the LLM)

This is the highest-leverage piece and it is **deterministic** — it wires in the rich execution
signals the audit found are computed-but-unread (`gold_total`, `failed`, `missing_expected`,
`failing_nodeids`, `fs_diff` size, `failure_excerpts`). It runs over the patience window (the
attempts since the frontier last rose) and reports a single bool `secondary_progress`:

- **errors-shrink** (already partly wired as Fix-1): `min(errors)` over the window strictly below the
  established baseline → progress (collection-collapse repos like pydantic/babel).
- **collected-rise:** `gold_total` is fixed, but `failed+errors+passed` (tests the suite managed to
  *enumerate*) rising → the suite is starting to collect.
- **distance-to-first-pass shrink:** `(failed + errors)` strictly shrinking with `gold_total` fixed
  → fewer failing/erroring gold ids, real movement toward the first green even before count flips.
- **missing_expected shrink:** fewer gold ids missing/uncollected → more of the gold universe now
  exists.
- **failing-id SET churn/shrink:** `len(failing_nodeids)` dropping, OR the *set* of failing ids
  differing across attempts (Jaccard < 1) → the code genuinely changed even when the count is flat
  (directly defeats the frozen-but-working + single-load-bearing-bug false-kill).
- **implementation-diff growth:** cumulative changed-files / diff-byte size strictly growing across
  attempts (112KB→263KB / 45 files) → real work even when no integer moved and `tokens=0`.

If **any** of these is strictly true over the window, `governor.verdict` is overridden to `continue`
and **both patience arms reset** — *without the LLM*, and *without banking any solve*. This is the
deterministic backbone; the LLM only sees plateaus that are flat on **every** one of these too.

---

## 5. The evidence packaged for the LLM (execution-grounded, never transcript)

The packet is a small JSON object built purely from journaled execution artifacts — **the agent's
conversation/self-report is deliberately EXCLUDED** (Observability-Gap + 18%-mis-rate lessons). It
contains:

```json
{
  "goal": "<the task contract: implement repo so gold expected-ids pass; n gold ids total>",
  "gold_total": 5091,
  "frontier": {
    "best_gold_passed": 0,
    "best_pass_rate": 0.0,
    "best_min_errors": 5091,
    "history": [[valid_idx, gold_count], ...],          // strict frontier rises only
    "valid_measurements": 9,
    "valid_measurements_since_improvement": 9,
    "nominal_measurement_units_since_improvement": 9,    // NOT wall-clock
    "indeterminate_total": 0
  },
  "window_measurements": [                               // the last K VALID measurements, per attempt
    {"attempt": 710003, "gold_passed": 0, "failed": 0, "errors": 5091,
     "collected": 0, "missing_expected": 5091, "pass_rate": 0.0,
     "diff_bytes": 112395, "changed_files": 45,
     "failing_nodeids_count": 50, "failing_nodeids_sample": ["...", "..."],
     "failure_excerpt_tail": "ImportError: cannot import name X from _discriminated_union",
     "finalization_status": "timeout", "indeterminate": false}
  ],
  "deltas": {                                            // PRE-COMPUTED deterministic deltas
    "errors_first_to_last": 0,                           // 5091 -> 5091 (flat)
    "diff_bytes_growth": 0,                              // frozen
    "failing_id_jaccard_vs_prev": 0.0,                   // SAME ids each time => no churn
    "failure_excerpt_changed": true,                     // the ImportError moved 4 layers deeper
    "collected_growth": 0
  },
  "secondary_probe_result": "flat-on-all-deterministic-signals",
  "deterministic_verdict": "cut:sterile-diff-streak",
  "veto_budget_remaining": 1
}
```

Key properties:
- Every field is from `vr` / candidate meta / frontier — **execution evidence**, not the transcript.
- The packet is **bounded** (last `K` measurements, failing-id *sample* + *count* not the full list,
  excerpt *tail* capped at ~2KB) so the call is cheap and the scoped_inputs hash is stable.
- The **`deltas` block is pre-computed deterministically** so the LLM's job is *interpretation*, not
  arithmetic it could fumble — and so a hallucinated number can be cross-checked against it (§6).
- `failure_excerpt_changed` / `failing_id_jaccard` are the *exact* "the error moved deeper / the
  failing set churned" signals the audit named as the missing frozen-but-working evidence.

---

## 6. Anti-over-claim grounding (so the LLM can't fabricate progress)

The LLM is read-only and its reply is **schema-validated and cross-checked against the deterministic
deltas before it can affect the decision** — it cannot simply *assert* progress:

1. **Schema.** The reply must match `GOVERNOR_SCHEMA`:
   `{"verdict": "continue"|"kill", "primary_signal": <enum of the named deterministic signals>,
   "evidence_field": <a key path into the packet>, "claimed_value": <number>, "rationale": <string>}`.
   A `continue` verdict **must cite a specific `evidence_field` and `claimed_value`**.
2. **Cross-check (the firewall).** Before honoring a `continue` veto, the host verifies the cited
   `claimed_value` **matches the packet's actual value at `evidence_field`** (exact for ints, ε for
   floats). If the LLM claims "errors shrank to 4000" but the packet says `errors_first_to_last=0`,
   the veto is **rejected** and treated as `kill` — a fabricated-progress veto is structurally
   impossible to act on. This is the *generated-reward-can-climb-while-solving-falls* mitigation
   applied at the decision boundary: the soft judgment is only allowed to *agree with* an
   execution-grounded number that already exists in the packet.
3. **Whitelisted progress reasons.** A `continue` veto is honored **only** if `primary_signal` is one
   of the §4 deterministic signal names *and* the cross-check passes — i.e. the LLM can only veto on
   *the same evidence the deterministic probe checks*, but with the latitude to spot a real signal
   the strict thresholds missed (e.g. "the ImportError advanced 4 layers across 3 attempts: the
   import chain is moving, this is pre-collection progress" — grounded in `failure_excerpt_changed`).
4. **Veto budget.** Each honored `continue` veto decrements `veto_budget_remaining`. At 0, a
   subsequent plateau is cut regardless. So even a *correct-looking but ultimately-stuck* arm is
   bounded.
5. **Cannot accept.** The verdict enum is literally `{"continue","kill"}` — there is no `accept`
   token, and the call goes through the read-only `ctx.ask`-class path that *cannot* produce a
   Candidate or touch `accepted`. Acceptance remains execution-grounded gold-pytest only, and
   acceptance-checkpointing (`_checkpoint_phase` / `_checkpoint_accepted`) banks any real solve the
   instant it passes, so even a wrong `kill` never discards a verified solve (Cardinal Contract C7).

---

## 7. False-kill robustness vs the named artifacts

| Artifact | How this design avoids the false kill |
|---|---|
| **tokens=0 telemetry artifact** (hard-killed rollout reports usage=all-zeros + empty final_message but produced a 263KB/45-file diff) | The packet **never reads tokens or final_message**. It reads `diff_bytes`/`changed_files` from `res.fs_diff` (the in-cell scorer always re-scores the worktree regardless of `finalization_status`). The §4 **diff-growth** signal sees 112KB→263KB and the deterministic probe alone resets the clocks — the LLM is never even needed. If diff is frozen too, the LLM still sees `diff_bytes=263000` and `failure_excerpt_changed=true` and can veto. tokens=0 is invisible to the whole pipeline. |
| **Frozen-diff-but-working** (carry frozen while real edits happened) | §4 **failing-id Jaccard churn** + **failure_excerpt_changed** catch "the code changed even though the committed diff sha repeated." If the deterministic thresholds are too strict, the LLM sees the churn/excerpt-move in the packet and vetoes on the whitelisted `failing_id_churn` / `failure_excerpt_moved` signal (cross-checked). |
| **Collection-collapse** (pydantic/babel gold suite can't COLLECT; frontier sits at 0 for 45/95 files) | §4 **errors-shrink / collected-rise / distance-to-first-pass / missing_expected-shrink** all credit pre-collect progress deterministically. The LLM sees the full per-attempt `errors`/`collected`/`missing_expected` series and can veto on "the suite is starting to collect" even when no single threshold tripped. The §5 `failure_excerpt` showing the ImportError advancing layers is grounded evidence the LLM can cite. |
| **Guard/policy-abort rollouts** (tokens=0 + frozen diff that looked sterile) | Same as tokens=0: diff is re-scored from the worktree; if a guard genuinely blocked all work the diff *and* every secondary signal *and* the LLM cross-check are all flat → a fair kill. If the guard blocked only *some* waves while others did real work, the diff/error/excerpt series shows it and the LLM vetoes. The design does not punish the abort — it judges the *execution evidence*, which is the audit's prescription. |
| **Sterile-diff-streak cutting real work** (the pydantic 8-streak where several were guard-aborts/time-kills doing real work) | This is the *exact* case the LLM veto targets: `cut:sterile-diff-streak` becomes a *candidate*, the §4 probe runs, and if flat the LLM gets the diff-size/error/excerpt/churn packet and can veto a genuine-but-not-yet-integer-moving streak. The 8-streak can no longer be a silent false kill — it must survive the deterministic probe AND the LLM cross-check. |

The unifying principle: **every false-kill artifact is a case where a CHEAP execution signal existed
but the integer frontier didn't read it.** This design wires those signals in deterministically
first, and uses the LLM only as a sapient backstop over the *same* execution evidence — so the LLM
adds robustness without becoming a new fabrication surface.

---

## 8. How it STILL kills a genuinely dead arm

A truly dead arm (identical empty diffs, errors flat, no churn, excerpt unchanged, no collect) is
cut, on a bounded schedule:

1. Frontier flat → deterministic plateau candidate fires (as today).
2. §4 secondary probe is flat on **every** signal (errors flat, diff frozen, no churn, no collect,
   excerpt unchanged) — no deterministic reset.
3. The LLM is given a packet where **every delta is 0/unchanged**. With nothing real to cite, a
   `continue` veto **cannot pass the cross-check** (there's no nonzero progress value to cite), so the
   honest outcome is `kill`. Even a hallucinated veto is rejected by the firewall and treated as kill.
4. If a (wrongly) honored veto ever slips through, the **veto budget `V`** (default 2) caps re-vetoes,
   and the **`gold_total`-scaled attempts backstop** (e.g. `max(64, ceil(8·log2(gold_total)))`,
   repo-agnostic) guarantees termination regardless of the LLM.
5. `cut:harness-stall` (no valid measurement ever) is **never routed to the LLM** and cuts
   deterministically.

So the *added* lifetime of a dead arm over today's behavior is **bounded by `V` extra patience
windows**, after which a forced deterministic cut fires. We never run forever.

---

## 9. ctx / governor API mapping (concrete methods + signals + new ones)

**Reuses (no change):**
- `RunGovernor.verdict(state)` (`governor.py:93`) — runs FIRST; its `cut:no-progress` /
  `cut:sterile-diff-streak` becomes the *candidate* trigger D.
- `ctx._wave_verdict` / `ctx.should_continue_waves` (`context.py:695,723`) — the journaled
  control-flow seam where the final `(continue, reason)` is recorded by position.
- `ctx.ask` machinery (`context.py:1014`) — the journaled read-only agent-call pattern the governor
  call is modeled on (`self._engine.agent(ScopedTask(sandbox="read-only", schema=...))`).
- `FrontierTracker.state()` / `frontier_history` (`frontier.py:212`) — frontier + history for the
  packet.
- `_checkpoint_phase` / `_checkpoint_accepted` (`context.py:467`) — acceptance-checkpointing, so a
  kill never loses a real or partial solve.

**New deterministic signals to surface into attempt meta** (all already computed in `vr`, just
written to meta and read by the probe):
- `meta["failed"]` ← `vr.failed`; `meta["missing_expected"]` ← `vr.missing_expected`;
  `meta["collected"]` ← `vr.passed + vr.failed + vr.errors`;
  `meta["diff_bytes"]` ← `len(res.fs_diff or "")`; `meta["changed_files"]` ←
  `changed-files count from fs_diff`. (`gold_total`, `errors`, `failing_nodeids`, `failure_excerpts`
  are *already* in meta — `context.py:965-969`.)

**New ctx internals:**
- `ctx._secondary_progress_probe(window) -> (bool, dict)` — the §4 deterministic probe over the
  window of attempt metas; returns `secondary_progress` + the pre-computed `deltas` block.
  Called inside `_observe`/`_wave_state` accounting; on True it resets `_valid_measurements_at_best`
  / `_valid_wall_at_best` exactly like a frontier rise (extends the existing Fix-1 path at
  `context.py:617-643`).
- `ctx._build_governor_packet() -> dict` — assembles §5 packet from journaled artifacts only.
- `ctx._governor_judge(packet) -> dict` — the journaled LLM call:
  `self._engine.agent(ScopedTask(prompt=GOVERNOR_PROMPT+packet_json, schema=GOVERNOR_SCHEMA,
  sandbox="read-only", ...), node_id=f"{ns}gov{self._gov_counter()}", agent_type="governor")`,
  then `_validate_and_crosscheck(reply, packet)` → `{"verdict","honored"}`.
- `ctx._gov_counter` — a journaled monotone counter (mirrors `self._wave_counter`,
  `context.py:392`) so the governor-call sequence is replay-stable.
- `ctx._veto_budget` — per-arm int (default `APEX_OMEGA_GOV_VETO_BUDGET=2`), decremented on each
  honored `continue` veto; journaled in `_wave_state`.

**New `RunGovernor` wiring:** `verdict()` gains an optional injected
`llm_hook: Optional[Callable[[dict], dict]] = None` and `veto_budget`. When a `cut:no-progress` /
`cut:sterile-diff-streak` would fire AND `llm_hook` is set AND `secondary_progress` is False, it
calls `llm_hook(packet)`; an honored `continue` (cross-check passed, budget>0) returns
`(True, "continue:llm-veto")` and decrements budget; otherwise it returns
`(False, "cut:no-progress-llm-confirmed")`. The LLM hook is **off by default** (env flag
`APEX_OMEGA_GOVERNOR_LLM=1`), so the deterministic SPFG+ behavior is the unchanged default and the
LLM is an opt-in robustness layer.

**New cut/continue reasons:** `cut:no-progress-llm-confirmed`, `cut:sterile-llm-confirmed`,
`continue:llm-veto` (telemetry: how often the LLM saved a false kill, for offline evaluation against
the deterministic-only governor).

---

## 10. Risks & mitigations

- **R1: LLM cost on a ladder of many plateauing cells.** Mitigated by the cadence (≤ `V+1` calls per
  arm, only on cells that plateau AND are flat on the deterministic probe). Empirically a small
  fraction of cells reach this. Hard-cap with `V`.
- **R2: LLM reward-hacking (continue forever).** Mitigated by the cross-check firewall (§6.2) +
  veto budget (§6.4) + attempts backstop (§8.4). The LLM can only *delay*, never *override*, the
  dead-arm guarantee.
- **R3: Determinism break if the agent call isn't perfectly journaled.** Mitigated by double-
  journaling (agent call cached by node id AND the consuming wave verdict cached by position) + a
  CI test that runs an arm, kills mid-flight, resumes, and asserts the governor verdict sequence is
  byte-identical (extends the existing wave-verdict replay test).
- **R4: Packet leaks transcript / self-report.** Mitigated structurally: `_build_governor_packet`
  reads ONLY `vr`/meta/frontier fields; an assertion forbids any `final_message`/conversation field
  in the packet. A unit test snapshots the packet keys.
- **R5: Cross-check too strict (rejects a legitimate veto whose cited number is approximate).**
  Mitigated by ε-tolerance on floats and by allowing the `primary_signal=failure_excerpt_moved`
  veto to cite a boolean (`failure_excerpt_changed`) rather than a number, so qualitative-but-real
  "the error moved" progress is honorable without a fabricated count.
- **R6: External validity** — thresholds (`V=2`, `K`, scaled backstop) are reasoned, not fit on a
  held-out repo; report them as the pre-registered rule, evaluated offline against the
  deterministic-only governor on the archived runs (`/tmp/omega_phase_ab_n3max`, `runs/*`) by
  counting `continue:llm-veto` events that were later followed by a real frontier rise (true saves)
  vs. ones that weren't (wasted vetoes).

---

## 11. Implementability

**4/5.** The journaling, read-only agent call, schema validation/nudge, plateau seam, frontier, and
acceptance-checkpointing **all already exist** and are reused verbatim. The genuinely new code is:
(a) the deterministic secondary probe (~80 lines, pure, the same shape as the existing Fix-1 path);
(b) the packet builder + cross-check (~120 lines, pure); (c) the journaled governor-call wrapper
(~40 lines, a near-clone of `ctx.ask`); (d) wiring `llm_hook` into `RunGovernor.verdict`. The only
moderate risk is getting the double-journaling replay-exact (covered by R3's CI test). The LLM layer
is **opt-in** behind an env flag, so it ships behind the unchanged deterministic default and can be
A/B-evaluated against it offline before becoming the default.
