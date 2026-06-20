# Stop-Policy Design — "CLPV-Governor": Composite Lexicographic Progress-Vector Frontier

**Design lane:** Best / novel. **Name:** `novel-best`.
**Author:** principal designer (stop-policy task #10).
**One-line:** Replace the single COUNT frontier with a *monotone composite progress vector* (a
lexicographically-ordered tuple of execution-grounded distance-to-solve signals). Kill ONLY when
that vector strictly fails to advance across a dual-AND patience window. Every signal is read from
real execution evidence; the verdict is a pure function of journaled inputs; acceptance stays
gold-pytest-only.

---

## 0. The problem in one paragraph

The current SPFG+ governor is already wall/token-independent and is robust to the harness-stall and
tokens=0 *re-scoring* artifacts (it always re-scores the worktree diff regardless of
`finalization_status`). Its remaining weakness is structural, not philosophical: **its progress
metric is too COARSE.** The frontier is `best gold_passed COUNT` (+ a pass_rate tiebreak + a
min-collection-errors secondary). On a single-load-bearing-bug large repo (pydantic / babel class)
every valid measurement is `gold_passed=0`, so a run implementing 45/95 files, advancing the import
chain four layers deep, and shrinking `failed` looks *byte-for-byte identical to a dead run* — and
gets cut by `sterile-diff-streak=8` or `no-progress`. Symmetrically, the FALSE-CONTINUE surface is
that ANY new non-empty diff sha resets the sterile streak forever, so useless churn keeps a dead arm
alive until the very loose 64-attempt backstop. Both failure modes are the *same* root cause: the
governor reads ~3 integers when the candidate meta + `VerificationResult` already carry ~8
execution-grounded progress signals that are computed and then thrown away. The fix is to make the
progress signal *rich and monotone* so a genuinely-progressing arm always advances it and a dead arm
never does.

---

## 1. The progress object: the Composite Lexicographic Progress Vector (CLPV)

Define, per VALID measurement, a tuple `P` of execution-grounded coordinates ordered from
"closest to a real solve" to "earliest sign of implementation life". The frontier is the
**lexicographic max-so-far** of `P` over all valid measurements (BEST-not-LAST). A measurement is
"progress" iff its `P` is **lexicographically greater** than the running frontier `P*`.

```
P = ( gold_passed,                      # [0] PRIMARY: gold expected-ids green (the ONLY accept axis)
      pass_rate_q,                       # [1] quantized public pass-rate (tiebreak within a gold tier)
     -failed,                            # [2] FEWER gold failures = closer to first pass   (NEW)
     -missing_expected,                  # [3] FEWER missing gold ids (more now collect)    (NEW)
     -errors,                            # [4] FEWER collection errors (suite collecting)  (Fix-1, kept)
      collected,                         # [5] MORE gold tests the suite managed to collect (NEW)
      novelty_depth )                    # [6] structural-change depth: import-chain / failing-set churn (NEW)
```

Coordinate semantics (all from `VerificationResult` / candidate meta — see §6):

- `[0] gold_passed` — `vr.passed` under `commit0_test_ids` gold scoring. The *only* axis that can
  ever mean "accept", and acceptance still goes through the unchanged gold-pytest gate (§3). A rise
  here is a real solve-progress step.
- `[1] pass_rate_q = floor(vr.pass_rate * 1000)` — quantized to 0.1% buckets so floating drift in
  the collected-count denominator can't fake a rise (the deepest cause of false-continue churn). A
  rise here within the same gold tier is a real public-suite improvement.
- `[2] -failed` — `vr.failed`. The single most important NEW coordinate for the collection-collapse
  / single-load-bearing-bug regime: it is the **distance-to-first-pass**. When the suite collects but
  no gold id has flipped yet, a run fixing real bugs drives `failed` down (e.g. 5091 → 4800 → ...)
  long before `gold_passed` leaves 0. This is exactly the missing signal GOVERNOR_AUDIT.md:136 #3
  called out.
- `[3] -missing_expected` — `vr.missing_expected`. A drop = more gold ids now *exist / collect* =
  real progress on a repo that couldn't even surface the gold ids yet.
- `[4] -errors` — collection-error count (the already-landed Fix-1 secondary, folded in as a vector
  coordinate so it composes with the rest instead of being a special-case OR).
- `[5] collected = passed + failed` (gold ids that produced a real outcome) — captures "the suite
  went from not-collecting to collecting" even when `errors` is reported oddly. Monotone proxy for
  "the gold suite is becoming runnable."
- `[6] novelty_depth` — a bounded, execution-grounded structural-change counter (the anti-churn
  *and* anti-frozen-diff coordinate). It is NOT "a new diff sha". It increments only when the run
  produces evidence that *code behavior actually changed* even though no integer above moved:
  - the **failing-nodeid SET churns** (the set of `vr.failing_nodeids` differs from the last valid
    measurement's set by ≥1 id — different tests failing means real code changed), OR
  - the **failure_excerpts head-error advances** (the first failing import/error line differs from
    last time — "the conftest ImportError moved 4 layers deeper" IS progress, GOVERNOR_AUDIT.md:64),
    OR
  - the **cumulative changed-files set grows** (a file edited this run was never edited in any prior
    valid measurement — real new implementation surface, not a re-paste of the carry diff).
  `novelty_depth` is a count of *distinct structural-change events*, capped per-window so churn can
  bump it a bounded number of times, not infinitely (see §5 false-continue defense).

**Lexicographic, not weighted.** Coordinates are compared in order; coordinate `k` is only
consulted when all of `0..k-1` tie. This is deliberate: it makes the metric *un-gameable by trading
a real axis for a cheap one*. A run can NEVER lower `gold_passed` and pretend progress by bumping
`novelty_depth` — a drop in a higher coordinate makes `P` lexicographically smaller, so it is a dry
sample (BEST-not-LAST: a dip below `P*` never advances the frontier and never resets patience). And
because `gold_passed` is the lexicographic head, no soft/structural coordinate can ever *bank a
solve* — only push back the kill clock.

---

## 2. THE EXACT KILL CRITERION (progress-only)

Let `P*` = running lexicographic-max CLPV over all VALID measurements. Maintain, exactly as today
but keyed on the CLPV rise instead of the count rise:

- `valid_since_adv` = VALID measurements since `P*` last strictly advanced.
- `wall_since_adv`  = JOURNALED valid-measurement nominal wall-seconds since `P*` last advanced
  (a deterministic per-measurement increment accumulated into a journaled scalar — NOT a live clock).
- `indet_streak`    = consecutive INDETERMINATE (harness/scorer-failed) measurements.

A measurement is **VALID** iff it is a real gold test outcome (the existing
`vr.indeterminate==False` filter: excludes `infra_nonresult` / harness_failure / parser_error /
environment_failure / native-crash / scoring-timeout / non-gold-source downgrade / Mode-A
`total==0 & rc∉{0,1}`). Indeterminate measurements are **neutral** to `P*` and to both patience
clocks and feed only `indet_streak`.

**KILL the arm iff ANY of:**

```
(K1) cut:no-progress      P* has NOT strictly advanced  AND  valid_since_adv >= W_meas_eff
                          AND  wall_since_adv >= W_time.
                          (dual-AND; ANY CLPV advance resets BOTH clocks)

(K2) cut:harness-stall    indet_streak >= INDET_CEIL (24)  AND  P* never advanced in this streak.
                          (a wall of measurements that NEVER produced a real test outcome — the
                           arm was never measured, so it is a DISTINCT non-result, not no-progress.)

(K3) cut:dead-floor       attempts_since_advance >= DEAD_FLOOR  (legacy backstop, retained).
                          DEAD_FLOOR = max(64, gold_total_patience)  — see §4 (large-repo aware).
```

That is the WHOLE kill rule. Note what is GONE versus today:

- **`cut:sterile-diff-streak` is RETIRED as an independent killer.** "No new useful diff" is now just
  one of several ways `novelty_depth` (coordinate 6) fails to rise — and it can only matter when
  EVERY higher coordinate is also flat. A frozen diff on a run whose `failed`/`errors` are still
  dropping does NOT trip any clock, because coordinates [2]/[4] advanced `P*`. This single change
  kills the #1 documented false-kill (GOVERNOR_AUDIT.md §"When the governor IS unfair").
- **`cut:nonresult-streak` is absorbed** into K2: an all-nonresult wave is indeterminate and feeds
  `indet_streak`; a run producing zero usable work for INDET_CEIL measurements is `cut:harness-stall`
  (a never-measured arm), which is the honest classification.

**No coordinate is wall-clock or tokens.** `W_time` is a *journaled nominal* increment per valid
measurement (deterministic), used ONLY as the second leg of the dual-AND so the kill needs both
"enough real measurements" and "enough nominal valid-work elapsed since the last advance". Tokens and
real wall-clock never enter any branch. Spinning 1000 agents over 24h is fine *as long as `P*` keeps
advancing* — and `P*` is rich enough that a genuinely-working arm keeps advancing it.

---

## 3. Cardinal Contract (acceptance untouched)

The CLPV-Governor returns ONLY `(CONTINUE, "continue")` or `(STOP, "cut:*"/"stop:*")`. It can never
mark a candidate accepted. Acceptance is unchanged: `vr.accepted` comes from v1's
`decide_evaluation` under `commit0_test_ids` gold scoring (`scoring.py:77-81,121-124`), i.e.
execution-grounded gold pytest only. Coordinates [1]-[6] are SOLELY kill-clock inputs; none can lift
`accepted`. Acceptance-checkpointing is unchanged and EXTENDED: today a strict `gold_passed` rise
checkpoints the partial frontier (`context.py:628-636`). Under CLPV we checkpoint the
frontier candidate on any rise in coordinate [0] **only** (the partial-gold milestone) — the lower
coordinates advancing the kill-clock must NOT write a "phase" checkpoint, because a shrinking
`failed` is not a banked partial solve, just a reason to keep going. So a kill still never discards a
real solve, and a soft-coordinate advance never fabricates one.

This directly honors the AgentPRM reward-hacking lesson (VERIFIED RESEARCH, arXiv:2502.10325):
*"success fell 82%→70% while the PRM's own validation reward kept RISING."* The lexicographic head
being the only accept axis means a climbing lower coordinate can buy time but can NEVER be cashed as
a solve. The worst a hacked lower coordinate can do is *delay a kill* (waste compute) — never *fake a
result*. That is the correct, conservative failure direction for a stop-policy whose binding
principle is "never kill a progressing arm".

---

## 4. Large-repo-aware patience (repo-AGNOSTIC, pre-registered)

The single tuning change beyond the vector: scale the patience FLOORS by the gold universe so a
95-file repo gets proportionally more attempts before any cut, WITHOUT reading repo identity.

```
gold_total_patience = clamp( ceil( BASE * log2(max(2, gold_total)) ), 64, 512 )
W_meas_eff          = w_meas_effective(global_w_meas, arm_attempt_budget)   # unchanged fairness fn
DEAD_FLOOR          = max(64, gold_total_patience)
```

`gold_total` is already in candidate meta (`context.py:965`) and is a pure function of the repo's
gold contract, not its name — so the pre-registered-rule / apples-to-apples property is preserved
(governor.py:40-45). `BASE` is frozen (e.g. 16: `log2(5663)≈12.5 → ~200` attempts for pydantic-class
repos; `log2(230)≈7.8 → 64` floor for minitorch). This is GOVERNOR_AUDIT Fix-4, made concrete and
clamped so it can never blow up unbounded or shrink below the small-repo floor.

`W_time` and `W_meas` keep their frozen SPFG+ defaults (7200s nominal / 12 valid meas), env-
overridable via the existing `APEX_FRONTIER_*` / `LADDER_*` vars. The wall leg is the cross-mode
equalizer for 1-shot/best-of-N arms (frontier.py:135-146, unchanged).

---

## 5. False-KILL robustness (the four named artifacts)

**Artifact 1 — tokens=0 timeout telemetry (263KB diff, usage=all-zeros, empty final_message).**
The in-cell governor ALREADY re-scores the worktree diff regardless of `finalization_status`
(`context.py:957-973` runs `self._scored(wt, res)` on every attempt). So a hard-killed-but-real-diff
agent still produces a real `VerificationResult` and a real CLPV. Under CLPV that 263KB diff that
moved the import chain advances coordinate [6] (changed-files set grew) and very likely [2]/[4]
(failed/errors dropped), so it RESETS the clocks. The naive "usage=0/empty message ⇒ sterile" read
is structurally impossible: NO coordinate reads usage or final_message — every coordinate is read
from the re-scored execution result. The artifact is neutralized by construction.

**Artifact 2 — frozen-diff-but-working (carry diff looks unchanged while real edits happen).**
Today this trips `sterile-diff-streak` because the content_sha repeats. Under CLPV a repeated
content_sha is NOT a killer — it only fails to bump coordinate [6]'s changed-files sub-signal. If the
real edits moved `failed`/`errors`/`missing_expected` (they did, by hypothesis), coordinates [2]-[4]
advance `P*` and reset patience. Even if the integers are momentarily flat, the failing-nodeid-set
churn and the head-error-line advance (the other two `novelty_depth` triggers) fire on a working run
whose visible diff is frozen — "different tests failing / the error moved" is exactly the
frozen-but-working fingerprint. The arm survives.

**Artifact 3 — collection-collapse (pydantic/babel: frontier pinned at gold_passed=0 for a long
time).** This is the headline fix. While `gold_passed=0`, the run is graded on coordinates [2]-[6]:
`failed`↓, `missing_expected`↓, `errors`↓ (5091→4800→...), `collected`↑, and `novelty_depth`↑
(import chain advancing four layers = head-error-line advances four times = four resets). A
genuinely-implementing collection-collapse run advances `P*` on nearly every valid measurement and
is NEVER cut by K1. A dead collection-collapse run (errors flat 5091→5091, same head error, no new
files, same failing set — the exact pydantic infra-blocked case in GOVERNOR_AUDIT §4) advances NO
coordinate and IS correctly cut. The metric finally distinguishes the two regimes that were
previously identical.

**Artifact 4 — guard/policy-abort rollouts (tokens=0 + frozen diff that looked sterile).** Same as
Artifacts 1+2: re-scored worktree → real CLPV. If the abort happened AFTER real edits landed in the
worktree, those edits score and advance `P*`. If the abort happened with genuinely no edits (the
sandbox-blocked case), the attempt is an empty diff → indeterminate-ish nonresult → feeds
`indet_streak`, and a WALL of them is `cut:harness-stall` (K2, the honest "never measured"
classification), NOT `cut:no-progress`. The workspace-guard fixes (WORKSPACE_GUARD_ANALYSIS.md) are
orthogonal and complementary: they reduce how often this artifact occurs; CLPV ensures that when it
does occur it is classified correctly.

**False-CONTINUE defense (the opposite failure — keeping a dead arm alive on churn).** Today ANY new
diff sha resets the sterile streak forever. Under CLPV, a fresh-but-useless diff each round can only
bump coordinate [6] via the changed-files sub-signal — and only for *genuinely new* files (a set, so
re-editing the same file repeatedly does NOT grow it). To keep advancing `P*` purely on
`novelty_depth`, an arm would have to touch a brand-new file or churn the failing-set or move the
head-error EVERY window — and `novelty_depth` is **capped at `NOVELTY_CAP` (e.g. 3) advances per
patience window**: after 3 structural-only advances with no integer-coordinate movement, coordinate
[6] is frozen for the rest of the window, so a pure-churn arm stalls `P*` and is cut by K1. This
closes the loose 64-backstop-only path: churn buys at most `NOVELTY_CAP` extra windows, then dies.

---

## 6. How it STILL kills a genuinely dead arm

A dead arm by definition produces, across `W_meas_eff` valid measurements AND `W_time` nominal
valid-seconds, NO advance in ANY of: gold_passed, pass_rate(0.1%), failed↓, missing_expected↓,
errors↓, collected↑, or `NOVELTY_CAP` distinct structural-change events. Concretely:

- **Pydantic infra-blocked (real dead case, GOVERNOR_AUDIT §4):** 8 repair waves, errors flat
  5091→5091, same head ImportError, same (empty) failing set, no new files, tokens=0. Every
  coordinate flat → `P*` never advances → K1 fires at `valid_since_adv >= W_meas_eff &
  wall_since_adv >= W_time`. KILLED. (Classified `cut:no-progress` if the waves were VALID
  `gold_passed=0` measurements; `cut:harness-stall` if they were indeterminate nonresults — either
  way it dies, with the honest reason.)
- **Spin-the-wheel identical rollout:** same diff sha, same score, no churn → no coordinate moves →
  K1. KILLED.
- **Pure-churn arm (new useless diff each round):** bumps [6] up to `NOVELTY_CAP` times, then [6]
  freezes; `P*` stalls; K1 fires one window later. KILLED (just `NOVELTY_CAP` windows slower — the
  deliberate, bounded cost of being generous to maybe-progress).
- **All-harness-fail arm:** never a valid measurement → `indet_streak` climbs → K2 at INDET_CEIL.
  KILLED as harness-stall.

So the policy is strictly *more permissive* than today on progressing arms and *no less terminal* on
dead ones: it removes only the false-positive kill paths (sterile-on-frozen-diff, count-blind
collection-collapse) while keeping every true-positive kill path (flat-everything, all-indeterminate,
dead-floor backstop).

---

## 7. Determinism / replay story

Every input is execution-grounded and journaled; the verdict is a pure function. Specifically:

- **CLPV coordinates** are all derived from `VerificationResult` fields, which are already journaled
  via the `score` WAL record (`to_dict`/`from_dict`, verify.py:38-66) and replayed by
  `resume_or_run_json` (`_scored`). `failed`, `missing_expected`, `failing_nodeids[:50]`,
  `failure_excerpts[:3000]` are ALL already in `to_dict()` — no new journaled field is required for
  coordinates [2]-[5]. (`failed`/`missing_expected` must additionally be copied into candidate
  `meta` so `_observe` can read them; see §8 — that meta is itself part of the journaled candidate.)
- **`novelty_depth`** is computed from journaled, lossy-but-deterministic inputs:
  `failing_nodeids[:50]` (set churn), `failure_excerpts` head line (error advance), and the
  cumulative changed-files set. The changed-files set must be derived deterministically from the
  journaled diff (parse the unified-diff `+++ ` headers from `res.fs_diff`, which IS journaled) — NOT
  from a live filesystem walk. Because `failing_nodeids` is truncated to 50 and `failure_excerpts` to
  3000 chars in the journal, the set-churn/head-error signals are computed on the SAME truncated
  values on replay, so they replay bit-identically. (The truncation is the one knowingly-lossy edge;
  it is identical on first-run and replay, so determinism holds — it just means the signal is
  computed on the truncated view in both passes.)
- **Patience scalars** (`valid_since_adv`, `wall_since_adv`, `indet_streak`) are accumulated from
  journaled per-measurement records exactly as today (frontier.py:178-210, context.py:578-690). No
  live clock is read; `W_time` is a nominal journaled increment.
- **The verdict itself** is journaled by wave position via `_wave_verdict` /`resume_or_run_json`
  (`context.py:695-719`) — first run computes + journals `(continue, reason)`; resume replays the
  cached verdict. So even if a future refactor made a coordinate non-pure, the *decision* still
  replays identically (cache HIT). The CLPV is a pure function on top of that, so it replays for free.

A learned/LLM progress critic is explicitly NOT used here (it would add cost and a journaled-cache
requirement, and — per the AgentPRM and execution-free-critic VERIFIED RESEARCH — mis-rates ~18% and
can reward-hack). CLPV is a *learned-free heuristic value*: cheap, transparent, execution-grounded,
and impossible to cash as a solve. If an LLM signal is ever wanted, it should sit STRICTLY BELOW
coordinate [6] as a final tiebreak that can only delay a kill, never advance the accept axis — kept
optional and off by default.

---

## 8. ctx / governor API mapping (concrete)

**No new journaled record types.** Reuses the existing `score` WAL + candidate meta + `wave` verdict
journal. Changes are localized.

### 8.1 New: a `progress_vector` module-level helper (in `apex_omega/engine/frontier.py`)
```python
def clpv(*, gold_passed:int, pass_rate:float, failed:int, missing_expected:int,
         errors:int, collected:int, novelty_depth:int) -> tuple:
    return (int(gold_passed), int(pass_rate*1000), -int(failed), -int(missing_expected),
            -int(errors), int(collected), int(novelty_depth))
```
Pure, deterministic, comparable by Python tuple `<`. Used identically by the live tracker and the
disk reconstruction.

### 8.2 `FrontierTracker` (frontier.py:153-222) — generalize `best`/`best_rate`/`best_min_errors`
into one `self.best_p: tuple` (the lexicographic-max CLPV) plus the cumulative state needed for
`novelty_depth` (`self._changed_files:set`, `self._last_failing_set:frozenset`,
`self._last_head_err:str`, `self._novelty_in_window:int`). `ingest(...)` gains params
`failed, missing_expected, collected, failing_nodeids, head_err, changed_files`; computes
`novelty_depth`, builds `P=clpv(...)`, sets `improved = P > self.best_p`, and on improve does
`self.best_p = P; reset both patience arms; reset _novelty_in_window`. `state()` exposes the same
keys it does today plus `best_clpv` and `best_gold_passed = self.best_p[0]` (so existing readers and
the partial-gold checkpoint still see the gold count).

### 8.3 `RunGovernor.verdict` (governor.py:93-141) — three branches K1/K2/K3
Replace the two count-based plateau branches and DROP the standalone `sterile-diff-streak` /
`nonresult-streak` branches. `state` keys consumed:
`valid_measurements_since_improvement` (now = `valid_since_adv` on the CLPV),
`seconds_since_frontier_improved`, `indeterminate_streak`, `attempts_since_improvement`,
plus `dead_floor` (= §4 `DEAD_FLOOR`, computed once from `gold_total`). Order: K2 (harness-stall) →
K1 (no-progress dual-AND) → K3 (dead-floor) → `can_start()` ceiling. The opt-in token floor stays
exactly as-is (inactive by default).

### 8.4 `Context._observe` (context.py:546-668) — build CLPV per attempt
- Read `failed`, `missing_expected`, `collected=passed+failed` from candidate meta (NEW meta keys —
  copy `vr.failed` / `vr.missing_expected` at meta-build time, context.py:959-969).
- Compute the three `novelty_depth` sub-signals from journaled inputs:
  `set(m["failing_nodeids"])` churn vs `self._last_failing_set`; first line of
  `m["failure_excerpts"]` vs `self._last_head_err`; changed-files parsed from `res.fs_diff` `+++ `
  headers vs `self._changed_files`. Increment `novelty_depth` (capped at `NOVELTY_CAP` per window).
- Replace the `improved = (round_gold>best) or (round_pass>...) or secondary_improved` block with
  `improved = clpv(round...) > self._best_p`. On improve: store `self._best_p`, append to
  `_frontier_history` **only when coordinate [0] rose**, acceptance-checkpoint **only when
  coordinate [0] rose** (preserve `context.py:628-636` semantics), reset both patience arms +
  `_novelty_in_window`.
- DELETE the standalone `_sterile_streak` / `_nonresult_streak` accumulation and their `_wave_state`
  emission (or keep them as TELEMETRY-only, not consumed by `verdict`).

### 8.5 `Context._wave_state` (context.py:670-692) — emit `dead_floor`
Add `"dead_floor"` and the CLPV-derived `valid_measurements_since_improvement` /
`seconds_since_frontier_improved` (already emitted; now reset on a CLPV rise instead of a count
rise). Drop `nonresult_streak` / `sterile_streak` from the consumed set (keep as telemetry).

### 8.6 Ladder-tier reconstruction (frontier.py:325-495)
`frontier_from_wal` and `frontier_from_rollouts` build the SAME CLPV from the WAL `value`
(`passed`, `pass_rate`, `failed`, `errors`, `missing`/`missing_expected`, and `failing_nodeids` /
`failure_excerpts` if journaled) and the rollout records. The dual-AND + INDET_CEIL + DEAD_FLOOR
verdict (`plateau_verdict`, frontier.py:225-243) is generalized to compare `best_clpv`. Mode-A's
`frontier_from_rollouts` gains the `failed`/`errors` reads it currently lacks (it presently only
reads `verification_passed` + pass_rate — extend `_rollout_valid` to also return
`failed`/`errors`/`selected_test_count` so Mode-A gets the [2]/[4]/[5] coordinates too; head-error /
changed-files may be unavailable in Mode-A rollouts → `novelty_depth` degrades gracefully to 0
there, which is conservative).

### 8.7 New frozen params (frontier.py `FrontierParams`, env-overridable)
`novelty_cap=3` (`APEX_FRONTIER_NOVELTY_CAP`), `dead_floor_base=16`
(`APEX_FRONTIER_DEADFLOOR_BASE`), `dead_floor_min=64`, `dead_floor_max=512`,
`pass_rate_quantum=1000`. All repo-agnostic, pre-registered, single-sourced like the existing knobs.

---

## 9. Cost

Negligible. CLPV is a 7-tuple comparison plus three set/string diffs per attempt, all on data
already computed by the verifier. No extra agent calls, no extra pytest runs, no LLM. The set-churn
and head-error diffs operate on the already-truncated `failing_nodeids[:50]` / `failure_excerpts[:3000]`
(bounded), and the changed-files parse is a single pass over the diff that was already produced. Per
attempt: O(50) set ops + O(diff size) header scan (the diff already exists in memory). Zero new I/O
on the hot path; the journal records are unchanged in shape (a couple of new meta scalars).

---

## 10. Risks

- **R1 — `novelty_depth` over-generous keeps a near-dead arm alive `NOVELTY_CAP` extra windows.**
  Bounded by design (`NOVELTY_CAP=3`) and clamped by `DEAD_FLOOR` (K3). The cost is wasted compute on
  a maybe-progress arm, never a lost solve — the correct failure direction for this policy's binding
  principle. Tune `NOVELTY_CAP` down if archives show churn-survival.
- **R2 — truncated `failing_nodeids[:50]` / `failure_excerpts[:3000]` make set-churn/head-error
  signals lossy.** They are identically lossy on first-run and replay (determinism holds). On a
  >50-failing-id suite the churn signal sees only the first 50 ids; mitigated because [2]`-failed`
  (the COUNT) is un-truncated and is the dominant collection-collapse signal — `novelty_depth` is the
  *tertiary* backstop, not the primary.
- **R3 — Mode-A lacks head-error / changed-files → `novelty_depth=0` there.** Conservative (Mode-A
  just relies on [0]-[5], which it now reads). No false-kill risk — only slightly less generosity for
  Mode-A frozen-diff cases, which are rarer (Mode-A is best-of-N, not loop-until-dry).
- **R4 — lexicographic ordering could mask a large lower-coordinate gain behind a 1-unit higher-
  coordinate dip.** Intended: a `gold_passed` or `pass_rate` *drop* is a real regression-ish dry
  sample (BEST-not-LAST), so it correctly does not advance `P*`. The frontier is the MAX, so the run
  is still credited for its best-ever `P*` and only the *patience clock* (not the accept) is at
  stake. Acceptable and aligned with the existing best-not-last convention.
- **R5 — `collected = passed+failed` double-counts with `errors` on some harnesses.** They are
  independent coordinates compared lexicographically, so a correlated move just advances `P*` once;
  no double-credit risk because advancement is boolean (`P>P*`), not additive.
- **R6 — retiring `sterile-diff-streak` could, in principle, let a fast-churning arm waste budget.**
  Covered by the `NOVELTY_CAP` window cap + `DEAD_FLOOR`; net, the policy is *more* terminal on pure
  churn than the old "any-new-sha-resets-forever" rule, not less.

---

## 11. Why this is the BEST option (vs the surveyed alternatives)

- **vs trajectory-anomaly classifier (Pathak 16-feature):** that signal is the SHAPE of the action
  graph (tool_count/cycles/drift) — execution-grounded but about the *agent's behavior*, not the
  *task's solve-distance*. It would flag a slow-but-working pydantic run as "looping" (it IS
  re-running the same tools) → exactly the false-kill we must avoid. CLPV reads the *task outcome*,
  which is the right target. (The anomaly classifier is a good orthogonal TELEMETRY add, not the
  stop-gate.)
- **vs Process-Reward-Model / iStar learned step-value:** powerful but learned, costly, needs a
  journaled cache, and DEMONSTRABLY reward-hacks (success↓ while reward↑). A learned value as the
  governor violates "never bank a soft score". CLPV gets the dense-progress benefit with a
  learned-FREE, un-cashable heuristic.
- **vs self-predicted-marginal-improvement / adaptive-compute stop:** a self-report → fails the
  execution-grounded-over-transcript constraint (the project's deepest lesson). Could gate
  *exploration breadth*, never the kill.
- **vs relative bandit allocation across arms:** valuable for *compute allocation* (re-route budget
  to the arm with the steepest `P*` slope) and is fully COMPATIBLE — CLPV gives the per-arm progress
  rate a bandit needs. But allocation is a different decision than the kill; the kill must be an
  absolute no-progress judgment so a slow-but-only-arm is never killed merely for being slower than a
  sibling. CLPV is the absolute signal; a bandit can sit on top later.

CLPV keeps everything SPFG+ already got right (wall/token-independence, indeterminate-awareness,
dual-AND patience, journaled determinism, Cardinal-safety) and fixes the one thing it got coarse: it
makes the progress signal *rich enough that a genuinely-progressing arm is never mistaken for a dead
one* — which is precisely the binding principle.
```
