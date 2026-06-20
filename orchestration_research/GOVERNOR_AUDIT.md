# SPFG+ Governor Audit — Large-Repo Validity (pydantic / networkx)

Deciding analyst report. Question under audit: on LARGE repos, does the SPFG+ governor CUT
runs that are genuinely making progress before the first gold test can flip green (UNFAIR), or
were the orchestrator runs genuinely making no creditable progress (VALID)?

## TL;DR Verdict

**MIXED, leaning VALID for the current run.** The central audit suspicion — that
*harness-stall cuts pydantic at indeterminate-streak 8 before the first gold pass* — is
**FALSIFIED by the evidence.** A total gold-suite collection collapse does NOT register as
INDETERMINATE; it registers as a **VALID `gold_passed=0` measurement**. So large repos are not
killed by a harness wall. They are killed (when killed) by `cut:sterile-diff-streak` or the
no-progress arms.

The governor is **VALID where the suite collects** (networkx: frontier rises 780->1763, no cut,
correctly kept alive) and **mechanically VALID but blind where the suite cannot collect at all**
(pydantic: pinned to a VALID `gold_passed=0` frontier). The blind spot is REAL: implementing 30
modules, fixing imports, or shrinking the collection-error count earns ZERO frontier credit
because the frontier is gold-pass COUNT only. In THIS run the cut was still fair (the streak that
fired was genuinely empty — 8 sandbox-blocked, zero-new-diff repair waves — and the
collection-error count never shrank, 5091->5091), but that fairness is an INFRA confound, not a
clean test of the governor.

## Evidence chain (verified)

### 1. Collection collapse is VALID, not INDETERMINATE (the falsifier)

`apex_omega/eval/scoring.py:88-112`: a `rc=4` collection-abort is flipped to indeterminate ONLY
via the v1 taxonomy (`harness_failure`/`parser_error`/`environment_failure`), a diagnostics flag,
or a native crash (`rc<0` or `rc in {134,137,138,139}`). A pytest collection error emits one
error-outcome node-id per uncollected module, so `errors>0 => signal_count>0 => SUCCESS =>
UNSOLVED`, i.e. a VALID `gold_passed=0` measurement.

LIVE PROOF — `/tmp/omega_phase_ab_n3max/hybrid-nogate__pydantic__s0`, WAL score records
(`result_status, indeterminate, passed, errors, total, pass_rate`):

```
('ok', False, 0, 5091, 5091, 0.0)   x9   <- VALID gold_passed=0, NOT indeterminate
('infra_nonresult', True, 0, 0, 5091, 0.0)  <- only the post-cut --memray abort
```

`evaluation_progress.json` evals 1-9: `passed=0 failed=0 errors=5091 collected=None rc=4`
(eval10 errors=0 = post-cut --memray abort). Collection-error count **5091 -> 5091, never
shrank** — the suite never began collecting.

### 2. The actual cut was sterile-diff, not harness-stall

`cells/autogen_orchestrator__pydantic/autogen_cell_report.json`:

```json
{"reason":"cut:sterile-diff-streak","is_cut":true,"best_gold_passed":0,
 "best_pass_rate":0.0,"valid_measurements":9,"indeterminate_total":0,
 "seconds_since_frontier_improved":27000.0}
```

WAL wave verdicts: 8x `{"continue":true,"reason":"continue"}` then
`{"continue":false,"reason":"cut:sterile-diff-streak"}`. `indeterminate_total:0` =>
harness-stall could never have fired. Suspicion FALSIFIED.

### 3. Real implementation work happened but earned zero frontier credit (the blind spot)

Fanout produced 5 completed module agents, ~112KB of genuine pydantic source diffs
(`json_schema.py update_json_schema` goes `pass -> schema.update(updates); return schema`, plus a
real importlib circular-import refactor). The conftest ImportError advanced ~4 import layers
deeper across evals (errors.py -> config.py TypeError -> _config.py:88 -> _typing_extra.py:207 ->
_discriminated_union ImportError), proving code changed. NONE of this moved the frontier: it is
gold-pass COUNT only (`frontier.py:13-17,190-198`), and pydantic has a single load-bearing import
bug gating all 5091 tests, so a real-work run is indistinguishable from a sterile one.

### 4. The cut was nonetheless fair HERE (infra confound)

The 8 loop-until-dry repair waves (rr710000-710007) all finished `policy_violation`/`timeout`
with `tokens=0` and `diff_bytes` frozen at exactly 112395 (== the banked fanout merge union) — no
new diff for 8 straight attempts => a genuine sterile streak (`governor.py:112`). BUT all 8 hit
denied `sandbox_escape` and auth preflight logged `Permission denied (os error 13)` — the agents
were largely BLOCKED from running. The one true cheating artifact (a 26.9MB / 2437-file `.venv39/`
diff) was correctly flagged and DISCARDED and did NOT drive the cut.

### 5. networkx is the validating counter-example

`hybrid-nogate__networkx__s0`: suite COLLECTS fully (`errors` steady ~255 of 5436), `passed`
climbs 780->806->837->...->1763, `valid=19 indet=0`, no cut, PID alive. The frontier correctly
credited the rise and reset both patience arms. `converge__networkx__s0`: the merged eval scored
`(1434, 81, 5436)` — frontier rose to 1434, no cut — even though many per-module sub-tree evals
scored `(0, 5436, 5436)` (transient broken sub-builds). The frontier rise rescued it.

## Quantitative tally of verified classifications

Per-cell verified classifications (auditor + adversarial verifier):

| Cell | Repo | State | Verified class | Cut |
|---|---|---|---|---|
| hybrid-nogate__pydantic__s0 | pydantic | done | **cheating/sterile (infra-confounded)** | cut:sterile-diff-streak (VALID) |
| hybrid-nogate__networkx__s0 | networkx | in-flight | **honest_ceiling / fairly-governed** | none (frontier rising) |
| converge__networkx__s0 | networkx | in-flight | **real_progress (frontier credited)** | none (frontier rising) |

Plus archive corroboration (all-arms, all-seeds): **every** pydantic cell in
`runs/ladder_n5`, `runs/ladder_final`, `runs/ladder_n5_framed` shows `maxpass=0` with VALID
measurements dominating (valid >> indet), i.e. pydantic universally collection-collapses to a
VALID `gold_passed=0` across **all** arms — symmetric.

Tally:
- **Genuine no-progress (VALID cut): 1/3** verified cells (pydantic — collection never shrank,
  streak genuinely empty). Mechanically valid; infra-confounded.
- **Fairly governed, still progressing (no cut): 2/3** (both networkx cells — frontier rising,
  governor kept them alive).
- **Unfair cut of a genuinely-progressing run: 0/3.** No cell was cut while a creditable signal
  was rising.

So for THIS run: **0% of cells were unfairly cut.** The governor's decisions are all defensible
on the evidence.

## When the governor IS unfair (the latent failure mode)

The pydantic cut was fair only because the repair loop was genuinely empty AND the
collection-error count was genuinely flat. Change either and the governor becomes UNFAIR:

**A run that is producing real, growing implementation diffs AND shrinking the collection-error
count on a single-load-bearing-bug repo (one import fix gates the whole suite) would be cut by
`sterile-diff-streak` / `no-progress` identically to a sterile run**, because every measurement
is a VALID `gold_passed=0` and the frontier sees no rise until the ONE fix lands. The governor has
no signal for "distance to first pass is shrinking." This is the precise unfair regime, and it is
exactly the regime large monolithic-import repos fall into. The current run dodged it only via the
infra block; a healthy-infra re-run could trip it.

## The precise missing progress signal

The frontier credits ONLY a rise in the gold-pass COUNT. On a repo whose gold suite cannot
collect, that count is frozen at 0 until a large fraction is implemented, so the governor is blind
to ALL of:

1. **collection-error count SHRINKING** (5091 -> 4000 -> ... = real progress toward first
   collect) — currently a flat sequence of VALID `gold_passed=0` measurements.
2. **tests COLLECTED rising** (suite going from not-collecting to collecting).
3. **distance-to-first-pass** (errors+failed shrinking toward the first pass).
4. **non-gold / visible-suite passes** rising (real behavior change before any gold id flips).
5. **new, non-empty, non-cheating implementation diffs** that change the import chain (the
   conftest error moving deeper IS progress but is invisible).

A SECONDARY frontier over (2)/(1) that RESETS the patience clocks would make these runs fairly
governed without weakening the sterile/no-progress cuts on genuinely dead runs.

## Concrete, minimal fixes (keyed to file/line)

### Fix 1 (PRIMARY): a secondary "implementation-progress" frontier that resets the patience clocks
`apex_omega/engine/frontier.py` `FrontierTracker.ingest` (around 177-198) and
`apex_omega/autogen/context.py` `_observe` (around 550-604).

Track a secondary monotone signal alongside `best` (gold count):
- `best_collected = max(collected)` (tests the suite managed to collect), and/or
- `best_neg_err = max(-(errors))` i.e. min collection errors so far (errors SHRINKING = progress).

Treat a strict improvement in EITHER as a frontier rise for the purpose of resetting BOTH patience
arms (`valid_at_best` / `wall_at_best` in frontier.py; `_valid_measurements_at_best` /
`_valid_wall_at_best` in context.py:603-604) — but DO NOT bank it as a gold solve (keep
`best_gold_passed` as the only acceptance number). Exact rule to add to `improved`:

```
improved = (pass_count > self.best) or (pass_rate > self.best_rate + 1e-9) \
           or (collected > self.best_collected) or (errors < self.best_min_errors)
```

`val.get("collected")` / `val.get("errors")` are already in the WAL score value. This directly
fixes the pydantic-style unfair regime: a run shrinking 5091->4000 errors keeps resetting the
clocks and is NOT cut.

### Fix 2: gate `sterile-diff-streak` on a real measurement, and only count NEW USEFUL diffs as the reset, not as the killer
`apex_omega/engine/governor.py:112-113` and `context.py:618-623`.

The sterile streak currently fires purely on identical/empty diffs. Add a guard so a sterile cut
requires that NO secondary-frontier progress occurred in the window either (`context.py`: reset
`_sterile_streak` not just on `any_new_useful or improved` but also on a secondary-frontier rise).
Minimal: extend `improved` in context.py:620 to the Fix-1 definition so the sterile reset inherits
it for free.

### Fix 3: unify the harness-stall threshold across tiers (the discrepancy)
The in-cell governor is built in `context.py:309-312` WITHOUT overriding `harness_stall_cut`
(default 8 in `governor.py:31`) or `sterile_streak_cut` (default 8), while the ladder tier uses
`INDET_CEIL=24` (`frontier.py:92`). Two different harness walls by tier. Pass the shared frontier
default through:

```
from ..engine.frontier import frontier_defaults
_w_time, _w_meas, _indet_ceil, _ = frontier_defaults()
self.governor = RunGovernor(engine=engine, agent_ceiling=engine.max_total_agents,
    token_budget=engine.budget.total, agent_budget=self.max_agents, plateau_k_dry=2,
    harness_stall_cut=_indet_ceil)
```

(Note: this does not affect pydantic, which had indeterminate_total=0; it removes a real
tier-inconsistency that WOULD bite a genuinely indeterminate large repo.)

### Fix 4 (large-repo aware patience floor — optional)
For repos with a large gold universe, the sterile/no-progress streaks of 8 attempts are short
relative to the implementation surface. Make the sterile/nonresult/harness streaks scale with the
gold universe (e.g. `max(8, ceil(log2(gold_total)))`) so a large repo gets more attempts before a
sterile/harness cut. Keep it repo-AGNOSTIC (function of gold_total, not repo name) to preserve the
pre-registered-rule property. Apply in `context.py` where `RunGovernor` is constructed using
`gold_total` from the repo_map.

## Is the current hard-repo eval biased by premature cuts?

**Not by UNFAIR premature cuts — 0/3 cells were unfairly cut.** But the hard-repo results ARE
biased in a deeper, structural way: **pydantic is currently UN-EVALUABLE as a gradient signal.**
Because its gold suite cannot collect, every arm is pinned to `gold_passed=0` regardless of how
much real implementation it produces. The "0 solved" pydantic result measures the suite's
single-load-bearing-import-bug gate, NOT arm quality — and it is confounded here by a sandbox/auth
infra failure that blocked the repair agents from running at all.

**Threat to the hybrid-vs-converge comparison:** the bias is **SYMMETRIC across arms** (both arms
hit the same collection collapse, the same VALID `gold_passed=0` frontier, the same sandbox block;
archive confirms all arms get maxpass=0 on pydantic). A symmetric bias does not invert the
ordering, but it DESTROYS pydantic's discriminating power — both arms score 0, so pydantic
contributes nothing to separating hybrid from converge and dilutes the average. networkx, which
collects, DOES discriminate (hybrid frontier 1763; converge merged 1434/peak 1763) and is the only
trustworthy hard-repo signal in the current run.

## Re-run recommendation

**YES — re-run the hard repos, but fix the INFRA confound FIRST, then the governor.** Priority:

1. **(Blocking) Resolve the sandbox/auth block** (`Permission denied os error 13`, denied
   `sandbox_escape` on every repair agent). Until the repair agents can actually run, pydantic is
   testing the sandbox, not the orchestrator or the governor. This is the dominant confound.
2. **Land Fix 1 + Fix 2** (secondary collection-error/collected frontier that resets the patience
   clocks; sterile reset inherits it) so a run shrinking the collection-error count is not cut as
   sterile/no-progress.
3. **Land Fix 3** (unify harness_stall_cut to INDET_CEIL across tiers).
4. Re-run hybrid vs converge on networkx + pydantic at n>=3 seeds. networkx is already
   trustworthy; pydantic becomes meaningfully evaluable only after (1)+(2) (it may still pin at 0
   if no run lands the load-bearing import fix, but at least progress will be credited and a
   genuinely-progressing run won't be cut).

Because the current cuts were not unfair, the existing networkx hybrid-vs-converge signal stands;
the pydantic numbers should be treated as infra-confounded non-results, excluded from the arm
comparison, not as a 0-vs-0 tie.
