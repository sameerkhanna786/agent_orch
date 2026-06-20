# SPFG++ — Richer Multi-Signal Execution-Grounded Frontier (no LLM)

Principal-designer stop-policy spec. Extends SPFG+ (`apex_omega/engine/{governor,frontier}.py`,
`apex_omega/autogen/context.py`) from a **scalar** gold-pass-count frontier to a **VECTOR** of
execution-grounded progress signals. Every signal is derived from REAL execution evidence
(pytest counts, on-disk diff bytes/files) — never from the agent's self-report or token/wall
telemetry. The stop decision stays a pure, journaled, replay-deterministic function of those
signals. The binding principle holds verbatim: **cut ONLY on a genuine no-progress plateau,
NEVER on tokens or wall-clock.**

---

## 0. One-paragraph thesis

The SPFG+ frontier is a single integer `best_gold_passed`. On a single-load-bearing-bug large
repo (pydantic/babel collection-collapse) that integer is pinned at 0 for the entire run even
while the agent implements 45/95 files, advances the import chain 4 layers, and grows a 263KB
diff — so a genuinely-progressing run is cut identically to a dead one (GOVERNOR_AUDIT.md §"When
the governor IS unfair"). SPFG+ already added ONE secondary signal (min collection-errors).
SPFG++ generalizes that one fix into a **principled vector**: the frontier becomes a tuple
`F = (gold_passed, collected, neg_errors, nongold_passed, neg_failures, neg_import_depth,
neg_failing_set, diff_bytes_useful)` ordered most-meaningful-first. **Plateau = EVERY component
flat across the patience window.** ANY single component strictly improving is progress and resets
both patience arms. Only `gold_passed` (component 0) is ever an acceptance/solve number; the other
seven are *patience-resetters only* — they keep a progressing run alive, they can NEVER bank a
solve (Cardinal Contract preserved).

---

## 1. The progress vector (exact components, exact source)

Each component is an integer (or scaled integer) where **larger = more progress**, so the whole
vector is monotone-by-component and a "rise" is well-defined per component. All are read from
fields ALREADY produced by execution (most already in `VerificationResult.to_dict()` /
candidate `meta`; the two new ones are cheap on-disk reads).

| # | Component | Definition (larger=better) | Source field (existing or NEW) | Catches |
|---|---|---|---|---|
| 0 | `gold_passed` | gold expected-ids green (COUNT) | `vr.passed` → meta `gold_passed` (READ today) | the real solve gradient; ONLY acceptance signal |
| 1 | `collected` | gold ids that COLLECT = `gold_total - errors` (clamped ≥0) | derived from `vr.total` (`gold_total`) + `vr.errors` (both in meta today, NOT read) | suite going from not-collecting → collecting (FM-2 / collection-collapse) |
| 2 | `neg_errors` | `-errors` (collection errors SHRINKING) | `vr.errors` → meta `errors` (READ today as SPFG+ Fix 1) | distance-to-first-collect shrinking (5091→4000) |
| 3 | `nongold_passed` | visible-suite passes ≈ `round(pass_rate * collected)` | derived from `vr.pass_rate` + `collected` | real behavior change before any gold id flips |
| 4 | `neg_failures` | `-(failed + errors)` = distance-to-first-pass | `vr.failed` (in to_dict, NEVER read) + `vr.errors` | repo that DOES collect but hasn't flipped a gold id yet |
| 5 | `neg_import_depth` | `-import_chain_depth` (the conftest ImportError moving DEEPER = fewer unresolved layers remaining; encoded as a small monotone integer) | NEW `import_depth` parsed from `vr.failure_excerpts` (see §4) | "the error moved" progress (pydantic conftest advanced 4 layers) |
| 6 | `neg_failing_set` | `-len(failing_nodeids_union_churn)` — see §3 for the exact set rule | `vr.failing_nodeids` (in to_dict, used only for repair, NEVER as progress) | failing-set SHRINKING or CHURNING on a single-bug repo |
| 7 | `diff_useful` | cumulative NEW-useful on-disk diff bytes (and a tie-break on changed-files count) | NEW `diff_bytes` / `changed_files` from `res.fs_diff` (string on disk, NOT usage tokens) | tokens=0 timeout that still wrote a 263KB diff (FM artifact #1/#2) |

**Ordering rationale.** Components are listed by how directly they imply distance-to-solve, but
the plateau test is symmetric (ALL must be flat), so the order matters only for telemetry and for
the optional component-weighted "frontier history" ledger. `gold_passed` is first because it is
the only one the SELECT step and acceptance ever read.

**Why not pass_rate as a primary?** Unchanged from SPFG+: collected counts drift, so raw
pass_rate is a SECONDARY tie-break (component 3 is a derived COUNT, not the raw rate, precisely so
a shrinking-denominator artifact can't fake a rise).

---

## 2. The EXACT kill criterion (progress-only)

Replace the scalar improvement test with a **vector improvement test**, keep the dual-AND patience
window and the indeterminate routing exactly as SPFG+ has them.

```
# Per VALID measurement m (indeterminate measurements are skipped entirely — see §5):
V(m) = (gold_passed, collected, neg_errors, nongold_passed,
        neg_failures, neg_import_depth, neg_failing_set, diff_useful)

# Best-so-far vector, component-wise max with a per-component established baseline:
for i in range(8):
    if best[i] is None:               # FIRST valid measurement establishes the baseline
        best[i] = V[i]                 # (a baseline is NOT itself an improvement)
    elif V[i] > best[i]:
        best[i] = V[i]
        rose = True                    # ANY component strictly above its best = progress

improved = rose                        # vector rise == at least one component rose
```

A measurement is **progress** iff `improved` is True. On progress: reset BOTH patience arms
(`valid_measurements_since_improvement → 0`, journaled wall arm `wall_at_best → wall_accum`).
On a non-progress valid measurement: advance both arms by exactly one valid measurement and one
journaled wall increment.

**`cut:no-progress` fires iff (and only iff):**

```
valid_measurements_since_improvement >= W_meas_effective   # default 12, mode-scaled (unchanged)
AND seconds_since_frontier_improved   >= W_TIME            # default 7200 journaled VALID-meas secs
```

i.e. EVERY one of the 8 components has been flat for BOTH a full valid-measurement window AND a
full journaled-wall window. A single component rising in either window resets both. This is the
*entire* no-progress rule — it reads no clock and no token count.

The other four cut reasons are **unchanged** but now strictly subordinate to the vector:

- `cut:sterile-diff-streak` (8): consecutive attempts with empty/repeated content_sha AND no
  vector rise. **Hardened (§6):** the streak now resets on ANY vector-component rise, not just on a
  new diff sha — so a frozen-diff-but-collecting run (errors shrinking, nothing else) no longer
  trips it.
- `cut:nonresult-streak` (8): consecutive attempts producing zero usable work (None / all-empty).
- `cut:harness-stall` (`indeterminate_streak >= INDET_CEIL` 24): a wall of harness/scorer-failed
  measurements — a DISTINCT non-result, never `cut:no-progress`.
- `stop:agent-ceiling`: honest "no headroom" (not a failure).

Legacy `attempts_since_improvement >= plateau_patience` (64) backstop is **retained** but now
keyed on the vector improvement (it resets when any component rises), so it can never cut earlier
than the vector says is plateaued.

---

## 3. The failing-nodeid-set rule (component 6, the frozen-but-working killer-feature)

This is the single highest-leverage addition for the single-load-bearing-bug regime, where COUNTS
stay flat but the CODE is changing. Maintain a per-cell rolling history of the last `K=4` valid
measurements' `failing_nodeids` (already in meta, capped at 50; we compare on the cap as a stable
sample). Define progress two ways, EITHER of which counts:

1. **SHRINK**: `len(failing_set_t) < best_min_failing_len` → component 6 rises (a gold id stopped
   failing — strictly good).
2. **CHURN**: `failing_set_t != failing_set_{t-1}` with `len` flat (a DIFFERENT set of ids fails
   now) → the code demonstrably changed behavior even though the integer is flat. Churn does NOT
   advance component 6's *best* (it's not monotone), but it counts as a **soft reset of the sterile
   streak only** (it proves a non-sterile diff landed), NOT a reset of the no-progress patience
   arms (churn alone, forever, must still eventually plateau-cut a dead-but-flailing run).

So: SHRINK = full progress (resets patience). CHURN = sterile-streak reset only (keeps the run off
the hard sterile cut, but the slow no-progress arm still governs it). This precisely separates
"the agent is changing real behavior" (don't hard-cut as sterile) from "the agent is genuinely
stuck and will never converge" (the dual-AND no-progress arm eventually fires).

Determinism note: `failing_nodeids` is the advisory, knowingly-lossy field (cap 50, possibly
ordering-unstable). We therefore (a) sort + dedup before hashing, (b) compare on `frozenset`, and
(c) **journal the resulting per-measurement `failing_len` and `failing_set_sha` into the score WAL
value** so the churn/shrink decision replays from journaled scalars, never from a re-derived
unstable list (see §7).

---

## 4. The import-chain-depth signal (component 5, "the error moved")

On collection-collapse repos the only visible movement for a long time is the conftest ImportError
advancing deeper (pydantic: `errors.py → config.py → _config.py:88 → _typing_extra.py:207 →
_discriminated_union`, GOVERNOR_AUDIT.md §3). Encode it cheaply and deterministically:

- Parse `vr.failure_excerpts` (the last 3000 chars of failing output, already captured) for the
  traceback's deepest in-repo frame: extract the ordered list of in-repo module files in the import
  chain (`re.findall(r'([a-zA-Z0-9_/]+\.py)', excerpt)` filtered to repo-relative paths), take its
  length as `import_chain_depth`. A DEEPER chain (more layers resolved before the next failure)
  means earlier imports now succeed → fewer unresolved layers remaining.
- `neg_import_depth = import_chain_depth` (we want it to GROW as resolution advances; larger=better,
  so it is already "negated" in spirit — keep the column name for symmetry but store the raw depth).
- Guard against noise: only credit a STRICT increase over the established baseline, and cap the
  depth at a small constant (e.g. 64) so a pathological traceback can't manufacture unbounded
  progress. A decrease is NOT a regression cut (BEST-not-LAST: we keep the max depth seen).

This is a heuristic proxy (an excerpt parse, not a count), so it is deliberately placed LOW in the
vector and is **reset-only** (it can keep a run alive; it can never bank anything). If the excerpt
is empty or unparseable, depth = baseline (no credit, no penalty) — fail-neutral.

---

## 5. How a timed-out-but-working rollout is NOT counted sterile (the artifact defenses)

The three named artifacts (tokens=0 telemetry, frozen committed diff, collection-collapse) are
each defeated by reading EXECUTION not telemetry:

**Artifact 1 — tokens=0 timeout that did real work.** The in-cell governor ALREADY re-scores the
on-disk worktree diff regardless of `finalization_status` (`_scored` keys on `res.fs_diff`, runs
pytest on the worktree). SPFG++ goes further: component 7 `diff_useful` reads `len(res.fs_diff)`
and the changed-files count from the **on-disk diff string**, NOT from `usage.output_tokens`. So a
rollout hard-killed at the per-agent wall with `usage=all-zeros + empty final_message` but a 263KB
/ 45-file diff produces a LARGE `diff_useful` rise → progress → patience reset. The token floor is
opt-in and default-inactive; SPFG++ never adds a token read.

**Artifact 2 — frozen committed diff but real edits.** If the *committed candidate* diff carry is
frozen (`content_sha` repeats) but pytest on the worktree shows errors shrinking / collected
rising / failing-set churning, components 1/2/4/6 rise (or churn-reset the sterile streak), so the
run is not sterile-cut. The sterile streak is no longer "no new content_sha" — it is "no new
content_sha AND no vector rise AND no failing-set churn" (§6). Frozen sha + moving execution =
NOT sterile, by construction.

**Artifact 3 — collection-collapse (gold frozen at 0).** While `gold_passed=0`, components 1
(collected), 2 (neg_errors), 4 (neg_failures), 5 (import depth), 6 (failing churn), and 7 (diff
bytes) can ALL still rise. A run shrinking errors 5091→4000, or advancing the import chain, or
growing a real implementation diff, keeps resetting the patience arms even though the gold count is
0. Only when the gold count is 0 AND every other component is flat for the full dual-AND window is
it cut — which is exactly a dead collection-collapse run.

**Indeterminate stays neutral (unchanged).** A measurement that is `indeterminate` (harness fail,
parser error, native crash, total==0 with rc∉{0,1}, non-gold scoring-source downgrade) contributes
to NEITHER the frontier NOR the patience clocks; it only feeds `indeterminate_streak →
cut:harness-stall`. So a wall of harness failures never reads as no-progress and never as progress.

---

## 6. The EXACT reset rule (single source of truth)

```
# After each VALID measurement m with vector V(m):
rose          = any(V[i] > best[i] for i in range(8) if best[i] is not None)
shrink_or_new = (V.diff_useful > best.diff_useful) or (failing_set_shrank)
churn         = (failing_set(m) != failing_set(m-1)) and not failing_set_shrank
new_sha       = (content_sha not seen before) and (diff not empty)

# (A) NO-PROGRESS PATIENCE ARMS (the dual-AND clocks): reset iff a vector component rose.
if rose:
    valid_measurements_at_best = valid_measurements      # → since_improvement = 0
    wall_at_best               = wall_accum               # → seconds_since = 0
    attempts_at_best           = agents_used              # legacy backstop arm
    # frontier history append only on a STRICT gold_passed rise; phase-checkpoint the partial.

# (B) STERILE STREAK: reset on a vector rise OR a new useful diff OR failing-set CHURN.
if rose or new_sha or churn:
    sterile_streak = 0
else:
    sterile_streak += max(1, n_empty_or_repeated_diff_attempts)

# (C) NONRESULT STREAK: unchanged (reset on any usable work in the wave).

# Indeterminate measurement: NONE of the above; only indeterminate_streak += 1.
```

Key invariants:
- A baseline (first valid value of a component) is established without crediting it (same
  convention SPFG+ uses for gold and errors). So an already-collecting small repo (errors==0,
  full collect from attempt 1) is *wholly unaffected* — components 1/2/4 are flat at their max
  immediately and only gold_passed (and pass_rate) drive it, identical to SPFG+ today.
- BEST-not-LAST per component: a dip below a component's best is a dry sample, never a regression
  cut. The vector is monotone non-decreasing in `best`.
- A frontier (gold) rise still acceptance-checkpoints the partial the instant it appears, so a
  later kill never discards verified work (unchanged `_checkpoint_phase`).

---

## 7. Determinism / replay story

SPFG++ is a **pure function of journaled scalars** — it replays for free, no LLM, no live clock.

1. **All 8 components are journaled at the source.** Six already are or are trivially derived from
   fields in `VerificationResult.to_dict()` (which is the score WAL value via `_scored`):
   `passed, failed, errors, total, missing_expected, pass_rate`. Two NEW scalars are added to
   `to_dict()` so they persist in the WAL: `diff_bytes` (`len(fs_diff)`), `changed_files`
   (count of `^diff --git` / `^+++ ` headers), `import_depth` (parsed int), `failing_len`
   (`len(failing_nodeids)`), and `failing_set_sha` (sha1 of the sorted-deduped failing ids). These
   are plain integers/strings — lossless across `to_dict`/`from_dict`.
2. **The decision is journaled by position.** `_wave_verdict` already records `(continue, reason)`
   per wave via `resume_or_run_json` and replays the cached verdict on resume. The vector logic
   runs inside `_observe` BEFORE the verdict and is reconstructed from the journaled per-attempt
   meta (FRESH or CACHED candidates both carry the same meta), so the reconstructed `_wave_state`
   is identical on replay → the cached verdict is consistent with it.
3. **No live wall-clock.** The wall arm uses the SAME journaled `_valid_wall_accum` scalar
   (incremented by a fixed `_valid_wall_increment` per valid measurement), never `time.time()`.
   Unchanged from SPFG+.
4. **Unstable lists are reduced to journaled scalars before any decision.** `failing_nodeids` (the
   one lossy field) is never compared as a live list across attempts; only its journaled
   `failing_len` + `failing_set_sha` drive shrink/churn, so an ordering wobble or the 50-cap can't
   flip a replay. The excerpt parse for `import_depth` runs once at score time and journals the
   resulting int.
5. **Ladder tier (separate process) reconstructs the same vector from the WAL.**
   `frontier_from_wal` already reads `val.get('passed'/'pass_rate'/'errors')`; it gains
   `val.get('failed'/'total'/'missing_expected'/'diff_bytes'/'changed_files'/'import_depth'/
   'failing_len'/'failing_set_sha')` and applies the identical component-wise `improved` rule. The
   `FrontierState.as_state()` keys consumed by `plateau_verdict` are unchanged (the vector collapses
   to the same `valid_measurements_since_improvement` / `seconds_since_frontier_improved`), so
   `plateau_verdict` and `relaunch_decision` need NO change — only the reconstruction's `improved`
   predicate widens. Mode-A `frontier_from_rollouts` gains the same widened predicate over the
   per-candidate scorecard fields it already parses.

Net: identical replay, identical resume, single-sourced rule across all three tiers.

---

## 8. Cost

Negligible and bounded:
- 6 of 8 components are arithmetic on counts already computed by the scorer (zero extra cost).
- `diff_bytes`/`changed_files`: `len()` of a string already in memory + one cheap regex count over
  the diff text (already produced). O(diff size), once per attempt.
- `import_depth`: one bounded regex over the ≤3000-char `failure_excerpts` already captured. O(1).
- `failing_set_sha`: sha1 over ≤50 sorted ids. O(1).
- WAL grows by ~5 small scalar fields per score record. Sub-kilobyte.
- NO extra pytest runs, NO LLM calls, NO network. The progress vector is computed inside the
  existing `_scored` / `_observe` path. Pure deterministic, so resume re-reads it for free.

Compared to an LLM governor (the rejected alternative): SPFG++ adds zero model tokens and zero
latency, and avoids the documented PRM/LLM-judge reward-hacking failure (AgentPRM: validation
reward rose while real success FELL; execution-free critics mis-rate ~18%). Keeping the judge
execution-grounded and arithmetic is the whole point.

---

## 9. False-kill robustness (per artifact, summarized)

| Artifact / regime | SPFG+ behavior | SPFG++ behavior |
|---|---|---|
| tokens=0 hard-kill, 263KB/45-file diff | could read as sterile (no new sha if carry frozen) | component 7 `diff_useful` rises from on-disk bytes → progress, patience reset |
| frozen committed diff, real edits | sterile streak could fire on repeated sha | sterile resets on churn/vector rise; errors-shrink/collect-rise = progress |
| collection-collapse, gold pinned 0 | only the min-errors secondary saved it (narrow) | 6 independent non-gold components can rise; cut only if ALL flat |
| import chain advancing 4 layers | invisible (no integer moved) | component 5 credits the depth increase |
| collects but no gold id flipped yet | invisible until first flip | component 4 (neg_failures) + 1 (collected) credit the climb |
| visible behavior change pre-gold | invisible | component 3 (nongold_passed) credits it |
| failing set shrinking/churning | invisible | component 6 shrink=progress, churn=sterile-reset |
| guard/policy-abort, tokens=0, frozen diff | could feed a false sterile read | the abort is indeterminate (neutral) OR the worktree re-score credits real diff bytes |
| harness wall (collection can't even run) | cut:harness-stall (correct) | unchanged — neutral to all 8 components |

**The dominant residual false-CONTINUE surface SPFG++ also tightens:** SPFG+ resets the sterile
streak on ANY new non-empty diff sha forever (useless churn keeps the run alive to the loose 64
backstop). SPFG++ keeps the run alive on USELESS-churn only via the *sterile* arm; the no-progress
dual-AND arm is unmoved by churn (only a real component rise resets it), so a run emitting fresh
but useless diffs each round now plateau-cuts at the dual-AND window instead of running to 64.

---

## 10. How it STILL kills a genuinely dead arm

A truly stuck arm has, by definition, ALL of: gold_passed flat, no new collection (errors flat),
no failing-set shrink, no behavior change (pass_rate flat), no import-chain advance, and no NEW
useful diff bytes — because if ANY of those moved, real code changed and we WANT to keep going.
When every one of the 8 components is flat for a full valid-measurement window AND a full journaled
wall window, `cut:no-progress` fires. The pydantic dead-run case (errors flat 5091→5091, diff
frozen, 8 empty repair waves) trips `cut:sterile-diff-streak` exactly as before. A
harness-only-failing arm trips `cut:harness-stall`. So:
- dead + empty diffs → `cut:sterile-diff-streak` (8 attempts)
- dead + harness can't run → `cut:harness-stall` (24 indeterminate)
- dead + producing useless non-empty diffs that move NO component → `cut:no-progress` (dual-AND)
- explored ceiling, agent budget exhausted → `stop:agent-ceiling`

No infinite run, no clock, no token read. The arm dies precisely when execution evidence shows
nothing real is moving.

---

## 11. ctx / governor API mapping (concrete; existing + NEW)

### Signals consumed by `governor.verdict(state)` — UNCHANGED
The governor stays a thin pure dispatcher over scalar streak/window state. SPFG++ does the vector
math in `context.py._observe` and `frontier.py.FrontierTracker.ingest`, collapsing the vector into
the SAME `state` keys `verdict` already reads:
`valid_measurements_since_improvement`, `seconds_since_frontier_improved`, `indeterminate_streak`,
`sterile_streak`, `nonresult_streak`, `attempts_since_improvement`. **No new governor branch** —
the no-progress branch (governor.py:126-128) is correct as-is; the widening is upstream in how
`*_since_improvement` is computed.

### `apex_omega/kernel/verify.py` — NEW fields on `VerificationResult` + `to_dict`/`from_dict`
- ADD: `diff_bytes: int`, `changed_files: int`, `import_depth: int`, `failing_len: int`,
  `failing_set_sha: str`. (All lossless scalars — extend `to_dict`/`from_dict` symmetrically.)
- `candidate_from_verification` already takes `changed_files_len`; wire it from the new field.

### `apex_omega/eval/scoring.py` — populate the new fields
- `verification_from_commit0_evaluation`: set `failing_len = len(failing_nodeids)`,
  `failing_set_sha = sha1(sorted(set(failing_nodeids)))`, and `import_depth =
  _parse_import_depth(excerpts)` (NEW small pure helper, §4). `diff_bytes`/`changed_files` are set
  by the caller (context, which holds `res.fs_diff`), not here.

### `apex_omega/autogen/context.py` — `_scored`, `_observe`, attempt meta
- `_scored` / attempt-meta builder (context.py:957-969): add `diff_bytes = len(res.fs_diff or "")`,
  `changed_files = _count_changed_files(res.fs_diff)` (NEW pure helper) to BOTH the meta and the
  score WAL value (so the ladder reconstructs them).
- NEW per-cell `_observe` state (mirrors `_best_min_errors`): `self._best_vec` = an 8-slot
  best-so-far list (None baselines), `self._prev_failing_sha` (for churn), `self._best_failing_len`.
- `_observe` (context.py:548-668): replace the `improved = (round_gold>..) or (round_pass>..) or
  secondary_improved` line with the vector `rose` computation over `self._best_vec`; keep
  `secondary_improved` (min-errors) folded in as component 2. Reset arms on `rose` (unchanged reset
  block). Sterile reset becomes `any_new_useful or rose or churn`.
- `_wave_state` (context.py:670-692): UNCHANGED keys (the vector collapses into them).

### `apex_omega/engine/frontier.py` — widen the reconstruction predicate (3 sites)
- `FrontierTracker.ingest` (frontier.py:178-210): extend `improved` to the vector rule over a
  best-vector; keep the `errors` path (it's component 2). Add `collected/neg_failures/diff/
  import_depth/failing` from the new ingest args (defaulted, back-compat).
- `frontier_from_wal` (frontier.py:325-391): read the new `val.get(...)` scalars; apply the same
  widened `improved`. Set `valid_at_best_idx` on ANY component rise (already the mechanism for the
  errors secondary).
- `frontier_from_rollouts` (frontier.py:424-495): same widened predicate over the per-candidate
  scorecard fields it already parses (add `failed/errors/total` reads from the scorecard).
- `plateau_verdict`, `FrontierState.as_state`, `w_meas_effective`, `relaunch_decision`: **NO
  CHANGE** — they consume the collapsed window scalars, which the widened predicate feeds.

### NEW pure helpers (small, deterministic, unit-testable)
- `frontier._vector_improved(best: list, v: list) -> (bool, list)` — the single component-wise
  rise+baseline rule, imported by BOTH `context._observe` and the three reconstruction sites
  (single source of truth, like `plateau_verdict`).
- `scoring._parse_import_depth(excerpt: str) -> int` — bounded regex, §4.
- `context._count_changed_files(diff: str) -> int` — count `^diff --git`/`^+++ ` headers.

### Env knobs (frozen defaults, all overridable; NO repo identity in any decision path)
- Reuse existing `APEX_FRONTIER_PLATEAU_WALL_S` / `_MEAS` / `_INDET_CEIL`.
- NEW optional `APEX_FRONTIER_VECTOR=0` to disable the extra 6 components and fall back to exact
  SPFG+ behavior (gold + pass_rate + min-errors) for an apples-to-apples ablation. Default ON.

---

## 12. Risks & mitigations

- **R1 — a non-gold component drifts upward forever (false-continue).** `collected`/`neg_errors`
  are monotone bounded by `gold_total`; once they hit max they're flat. `nongold_passed` is bounded
  by `collected`. `diff_useful` is the real risk (an agent can grow the diff with junk). Mitigation:
  `diff_useful` is a cumulative-bytes BEST, so it stops rising once the diff stops growing; a run
  padding the diff each round is caught because padding adds bytes once then plateaus, and the
  no-progress dual-AND arm (which only `rose` resets) governs it. If junk-padding proves a real
  exploit in eval, demote `diff_useful` to a sterile-reset-only signal (like churn) so it cannot
  reset the no-progress arm — a one-line policy flip.
- **R2 — import_depth is a heuristic parse (~18% mis-rate class, per execution-free-critic
  research).** Mitigated by placing it LOW, reset-only, strict-increase, capped, and fail-neutral
  on parse failure. It can only KEEP a run alive, never kill, never accept.
- **R3 — failing_nodeids 50-cap hides shrink/churn on huge suites.** Accepted: the cap is a stable
  SAMPLE; churn/shrink on the sampled top-50 is still real evidence. Other components (errors,
  collected) cover the un-sampled bulk.
- **R4 — vector widening could mask a genuinely dead run that emits one churny diff forever.**
  Churn resets only the sterile streak, NOT the no-progress dual-AND arm, so such a run still
  plateau-cuts. This is the explicit design choice in §3/§9.
- **R5 — external validity:** like SPFG+, the component set is reasoned against the 4–7 repo ladder
  with no held-out repo. Report as the registered rule, not a tuned optimum; the
  `APEX_FRONTIER_VECTOR` flag enables a clean SPFG+ vs SPFG++ ablation on the next n≥3 re-run.

---

## 13. Implementability

Scored 4/5: all plumbing exists (the meta already carries gold_passed/gold_total/errors/failed/
failing_nodeids/finalization_status; `_scored` already journals the score value; the dual-AND +
indeterminate routing is built; the ladder reconstruction mirror is built). The work is (a) 5 new
scalar fields through `VerificationResult.to_dict`, (b) one shared `_vector_improved` helper, (c)
two small parse helpers, (d) widening the `improved` predicate at 4 sites to call the shared helper.
No new pytest runs, no LLM, no clock, no schema migration (additive WAL fields). The 1-point
deduction is for the `import_depth` parser and the failing-set churn rule needing per-repo
excerpt-format validation before they can be trusted as more than reset-only signals.
