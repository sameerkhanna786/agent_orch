# STOP_POLICY_DESIGN — the deciding-architect verdict

**Date:** 2026-06-20
**Author:** deciding architect (stop-policy design)
**Scope:** `apex_omega/engine/{governor,frontier}.py`, `apex_omega/autogen/context.py`,
`apex_omega/kernel/verify.py`, `apex_omega/eval/scoring.py`, `scripts/run_ladder.py`.

---

## 1. Decision

**CHOSEN: `SPFG++` (the 8-component execution-grounded progress VECTOR) as the deterministic
default core, with the `hybrid-veto` LLM seam GRAFTED on as an opt-in, budget-capped,
strictly-one-directional VETO layer that can ONLY downgrade a deterministic `cut` to
`continue` — never originate a kill and never bank a solve.**

Why this graft and not either design alone:

- **SPFG++ is the strongest no-false-kill design (judge: 88, top of every lens that matters
  here).** It is verified-correct against the code: the worktree is re-scored on EVERY attempt
  regardless of `finalization_status` (`context.py:945-947, 957`), so the `tokens=0` timeout
  artifact and guard-aborts are already neutralized at the source — a hard-killed-but-263KB-diff
  rollout still earns a real pytest measurement. SPFG++ then widens the *progress* signal from a
  single COUNT to 8 execution-grounded components so the dominant residual false-kill surface (a
  genuinely-progressing large repo whose gold-pass count is pinned at 0) is covered by 7 more
  reset signals, all of which are pure functions of journaled scalars → byte-identical replay,
  zero LLM cost, no reward-hacking surface.

- **The `hybrid-veto` graft buys the ONE thing SPFG++ structurally cannot: rescue of a
  metric-INVISIBLE real climber.** Every deterministic vector is, by construction, blind to
  progress that moves no integer it tracks. The judge docked hybrid-veto only because its LLM
  veto is budget-capped (so a slow climber past the budget is still cut) and adds a small
  decline-misrate — but a budget-capped, veto-ONLY, default-OFF seam adds **zero** new false-kill
  surface to the default config (the LLM is never on by default and can never originate a cut),
  while making a future opt-in rescue available. We take SPFG++'s safety floor as the default and
  hold the LLM veto as a flagged escape hatch, getting the 88-design's floor with the 84-design's
  ceiling and none of the 84-design's default-path cost.

We **reject** `novel-best` (CLPV) for the default core: its `NOVELTY_CAP=3`-then-freeze coordinate
and its strict lexicographic `P > P*` rule are both *false-continue defenses that trade directly
against no-false-kill* — a slow climber whose only movement for a stretch is structural (conftest
ImportError advancing one layer at a time, the exact pydantic regime) can exhaust the cap and be
cut, and an errors-shrinking measurement that jitters pass_rate down one bucket becomes a DRY
sample under lexicographic ordering. SPFG++'s component-wise "ANY component rose → reset" is
strictly more permissive and is the correct posture for the binding principle. We **reject**
`llm-governor-agent` as a standalone because its anti-over-claim firewall constrains the LLM to
re-cite only numbers the deterministic probe already checks, so it adds near-zero marginal rescue
while carrying full LLM cost — the `hybrid-veto` seam keeps the firewall idea but as a *bounded
add-on over the full SPFG++ vector*, not a replacement.

---

## 2. Kill policy (one crisp paragraph)

We KILL an orchestrator arm **only when execution evidence shows that nothing real is moving**,
and **never because it spent tokens or wall-time**. Progress is an 8-component execution-grounded
VECTOR — `(0) gold_passed`, `(1) collected = gold_total − errors`, `(2) neg_errors`,
`(3) nongold_passed = round(pass_rate·collected)`, `(4) neg_failures = −(failed+errors)`,
`(5) import_chain_depth`, `(6) failing-nodeid-set shrink/churn`, `(7) diff_useful = cumulative NEW
on-disk fs_diff bytes/files` — all derived from `VerificationResult` fields pytest already produces
plus 5 cheap new journaled scalars; a STRICT rise in **any one** component over its established
baseline resets every patience clock, so a climbing arm is never cut. The soft kill
`cut:no-progress` fires **iff EVERY component is flat across BOTH a full VALID-measurement window
(`valid_measurements_since_improvement ≥ W_meas_eff`, default 12, mode-scaled) AND a full journaled
VALID-measurement wall (`seconds_since_frontier_improved ≥ W_TIME`, default 7200 — a deterministic
nominal increment per valid measurement, NOT a live clock)** — the dual-AND reads no clock and no
token count. Three subordinate hard cuts handle objectively-dead states that no amount of identical
rollouts escapes: `cut:sterile-diff-streak` (8 attempts with no new useful diff AND no vector rise
AND no failing-set churn), `cut:nonresult-streak` (8 attempts of zero usable work), and the DISTINCT
`cut:harness-stall` (24 consecutive indeterminate measurements — a harness/scorer wall that was
never actually measured, kept separate so it is never booked as no-progress). Indeterminate
measurements are NEUTRAL to all 8 components and to both clocks. Acceptance is **never** touched by
any of this: only component 0 (`gold_passed`) is a solve number, the other 7 are reset-only and can
never bank a candidate (Cardinal Contract); acceptance-checkpointing banks any verified solve the
instant it passes, so a kill can never lose a real solve. An **opt-in** read-only LLM veto
(`APEX_OMEGA_GOVERNOR_LLM=1`, default OFF) may, at a deterministic-kill candidate, spend up to
`VETO_BUDGET` (default 2) journaled `ctx.ask`-style calls fed an execution-evidence packet (NO
transcript, NO tokens) to DOWNGRADE the kill to `continue` when it spots a real-but-metric-invisible
climber — it can **only** delay a kill, never originate one and never accept.

---

## 3. New progress signals (8-component vector)

Ordered most-meaningful-first. Only component 0 is an acceptance/solve number; 1–7 are
**reset-only** (they reset patience clocks and the sterile streak; they can NEVER bank a solve).

| # | Component | Source field (verified) | Today |
|---|-----------|--------------------------|-------|
| 0 | `gold_passed` | `vr.passed` (`scoring.py:151`), meta `context.py:964` | PRIMARY frontier (kept) |
| 1 | `collected = gold_total − errors` | `vr.total` (`scoring.py:72`) + `vr.errors` | `gold_total` NEVER read as progress |
| 2 | `neg_errors` (collection-errors shrinking) | `vr.errors` (`scoring.py:151`) | SPFG+ Fix 1 (kept; now component 2) |
| 3 | `nongold_passed = round(pass_rate·collected)` | `vr.pass_rate` (`scoring.py:75`) | pass_rate is tie-break only |
| 4 | `neg_failures = −(failed+errors)` (distance-to-first-pass) | `vr.failed` (`scoring.py:151`) | **NEVER read** anywhere |
| 5 | `import_chain_depth` (conftest ImportError advancing deeper) | parsed once from `vr.failure_excerpts` (`scoring.py:144-149`) | NEW |
| 6 | failing-nodeid set shrink/churn | `vr.failing_nodeids` (`scoring.py:134-143`) → journaled `failing_len` + `failing_set_sha` | used ONLY for repair seeding today |
| 7 | `diff_useful` = cumulative NEW on-disk `fs_diff` bytes + changed-files | `res.fs_diff` STRING (`context.py:968`), NOT usage/tokens | only EMPTINESS used today |

5 NEW journaled scalars (added symmetrically to `VerificationResult.to_dict`/`from_dict` so they
replay losslessly): `diff_bytes`, `changed_files`, `import_depth`, `failing_len`, `failing_set_sha`.

**Heuristic-safety placement.** Components 5 (`import_depth`, ~18% excerpt-parse misrate class) and
6 (`failing-set churn`) are placed LOW, are STRICT-INCREASE + RESET-ONLY, and are **fail-neutral on
parse failure** — a parse miss can only *keep an arm alive*, never kill or accept it. `import_depth`
is clamped at 64 but resets on EVERY strict rise (NOT per-window capped — this is precisely the
CLPV `NOVELTY_CAP` flaw we reject). Component 7 reads the diff STRING length, structurally immune to
the `tokens=0` artifact.

---

## 4. Kill criterion (exact)

`cut:no-progress` fires **iff**:

```
valid_measurements_since_improvement >= W_meas_eff   (default 12, mode-scaled via w_meas_effective)
AND seconds_since_frontier_improved  >= W_TIME        (default 7200 JOURNALED valid-meas seconds)
```

where **"improvement"** = a strict rise in ANY ONE of the 8 vector components over its
per-component established baseline. EVERY component must be flat across BOTH windows; ANY single
component rising in EITHER window resets BOTH arms. No clock, no token count is read.

Subordinate hard cuts (unchanged routing, sterile reset now widened):
- `cut:sterile-diff-streak` (8): consecutive attempts with `empty_diff OR repeated sha` **AND** no
  vector rise **AND** no failing-set churn.
- `cut:nonresult-streak` (8): consecutive attempts of zero usable work.
- `cut:harness-stall` (`indeterminate_streak >= 24`): a wall of harness/scorer-failed measurements;
  DISTINCT from no-progress (the arm was never actually measured).
- `stop:agent-ceiling`: honest no-headroom (not a failure-to-progress).
- Legacy `attempts_since_improvement` (64): retained as a backstop, now keyed on the vector rise.

---

## 5. Implementation plan (ordered, exact)

Single source of truth: one `_vector_improved(best_vec, new_vec)` helper shared by all three tiers
(mirrors how `plateau_verdict` is single-sourced today). `governor.verdict` is **UNCHANGED** — the
widened vector collapses into the SAME window scalars (`valid_measurements_since_improvement`,
`seconds_since_frontier_improved`) it already reads, so no governor branch changes.

### Step 1 — `apex_omega/kernel/verify.py` (carry the 5 new scalars losslessly)
- Add fields to `VerificationResult`: `diff_bytes: int = 0`, `changed_files: int = 0`,
  `import_depth: int = 0`, `failing_len: int = 0`, `failing_set_sha: str = ""`.
- Add all 5 to `to_dict()` and `from_dict()` (so the score WAL value carries them and a journal
  replay reconstructs them byte-identically). `failing_set_sha` is the sha1 of the SORTED-DEDUPED
  `failing_nodeids[:50]` — reduces the lossy/unstable-order list to a stable scalar BEFORE any
  decision, so replay never re-derives the unstable list.

### Step 2 — `apex_omega/eval/scoring.py` (populate the new scalars at score time)
- NEW pure helper `_parse_import_depth(excerpt: str) -> int`: count distinct dotted-module frames in
  the ImportError/ModuleNotFoundError chain in `failure_excerpts` (bounded regex over the ≤3000-char
  excerpt; returns 0 on any miss → fail-neutral).
- In `verification_from_commit0_evaluation`, before the `return VerificationResult(...)`: compute
  `failing_len = len(failing_nodeids)`, `failing_set_sha = sha1(sorted(set(failing_nodeids)))`,
  `import_depth = _parse_import_depth(excerpts)`; pass them into `VerificationResult(...)`.
  (`diff_bytes`/`changed_files` are populated in context.py where `res.fs_diff` is in hand.)

### Step 3 — `apex_omega/autogen/context.py` (the vector accounting in `_observe`)
- `__init__` (near `:329, :338, :361`): add `self._best_vec = None` (the per-component best vector),
  `self._prev_failing_sha = None`, `self._best_failing_len = None`.
- Candidate meta site (`:957-969`): add `diff_bytes = len(res.fs_diff or "")`,
  `changed_files = _count_changed_files(res.fs_diff)` to BOTH the meta dict AND the score WAL value
  (via `_scored`). NEW pure helper `_count_changed_files(diff_str)` = count of `+++ b/` headers.
- `_observe` (`:548-668`): build the 8-component `round_vec` from the batch's candidate meta;
  replace the scalar `improved` line (`:623-625`) with `improved = self._vector_rose(round_vec)`
  (delegating to the shared `frontier._vector_improved(self._best_vec, round_vec)`); update
  `self._best_vec` component-wise on any rise. The sterile reset (`:665-668`) becomes
  `if any_new_useful or improved or failing_set_churn: self._sterile_streak = 0`. `_wave_state`
  keys (`:680-692`) are UNCHANGED — the widened vector feeds the SAME window scalars.

### Step 4 — `apex_omega/engine/frontier.py` (widen at 3 reconstruction sites via ONE helper)
- NEW pure module function `_vector_improved(best: dict|None, v: dict) -> bool`: strict rise in any
  of the 8 components (None best → establish baseline, not itself an improvement; mirrors the
  existing `best_min_errors` baseline convention at `:198-201`).
- `FrontierTracker.ingest` (`:178-210`): accept the new scalars, fold them into the `improved`
  decision via `_vector_improved` (the existing `errors` path at `:196-202` becomes component 2 of
  the shared helper; `best` / `best_rate` stay as components 0/3).
- `frontier_from_wal` (`:349-390`): read `val.get('diff_bytes'/'changed_files'/'import_depth'/
  'failing_len'/'failing_set_sha')` from the WAL value and feed `_vector_improved`; set
  `valid_at_best_idx` on any vector rise (the `:386-388` mechanism, now vector-driven).
- `frontier_from_rollouts` (`:471-493`): same widening where the rollout record carries the fields
  (fail-neutral when absent — Mode-A keeps its current count-only behavior plus whatever scalars
  exist). `plateau_verdict` / `FrontierState.as_state` / `w_meas_effective` / `relaunch_decision`
  need NO change (vector collapses into the same window scalars).

### Step 5 — `apex_omega/engine/governor.py` (UNCHANGED) + opt-in LLM veto seam
- `governor.verdict` body is unchanged. Add ONE opt-in hook in `__init__`:
  `self._llm_veto = llm_hook` (default `None`), gated by `APEX_OMEGA_GOVERNOR_LLM=1`.
- At the `cut:no-progress` / `cut:sterile-diff-streak` return sites: if `self._llm_veto` is set and
  the per-arm veto budget (default `VETO_BUDGET=2`) is not exhausted, call the hook with the
  execution-evidence packet; if it returns `continue`, decrement the budget and return
  `(True, "veto:llm-rescue")` instead of cutting. The hook can ONLY downgrade; it is never consulted
  on a climbing frontier (that returns `continue` long before reaching these sites).

### Step 6 — LLM veto seam wiring (`context.py`, modeled mechanically on `ctx.ask`)
- NEW `_governor_llm_hook(packet) -> {"verdict": "continue"|"kill"}`: a forced read-only
  `self._engine.agent(ScopedTask(schema=GOVERNOR_SCHEMA, sandbox=read-only))` with stable
  `node_id = f"{ns}gov{plateau_index}"` and `scoped_inputs` keyed on the packet hash. The engine
  already journals + replays agent calls by node id, so on resume the recorded JSON verdict is a
  cache HIT — the LLM is NOT re-invoked and the branch is byte-identical. DOUBLE-journaled via
  `_wave_verdict`'s `resume_or_run_json` envelope. Packet = frontier history + per-attempt pytest
  counts + pre-computed deltas + failing-id sample + failure-excerpt tail; the agent
  transcript/self-report is DELIBERATELY EXCLUDED. Anti-over-claim cross-check: a `continue` veto is
  honored only if any cited number EXACTLY matches a value already in the packet.

### Step 7 — env flags
- Reuse `APEX_FRONTIER_PLATEAU_WALL_S` / `_MEAS` / `_INDET_CEIL`.
- NEW `APEX_FRONTIER_VECTOR=0` to ablate back to EXACT SPFG+ (default ON) — enables a clean
  SPFG+ vs SPFG++ A/B on the next n≥3 re-run.
- NEW `APEX_OMEGA_GOVERNOR_LLM=1` to enable the veto seam (default OFF; deterministic SPFG++ is the
  unchanged default), `APEX_OMEGA_GOVERNOR_VETO_BUDGET` (default 2).

---

## 6. Determinism / replay

Pure function of journaled scalars, no LLM and no live clock on the default path. (1) 6 of 8
components are fields/derivations of `VerificationResult.to_dict()` which IS the score WAL value;
the 5 new scalars are added symmetrically to `to_dict`/`from_dict`. (2) The decision is journaled by
position via `_wave_verdict`/`resume_or_run_json` and replays the cached `(continue, reason)`. (3)
The wall arm uses the journaled `_valid_wall_accum` incremented by a fixed per-valid-measurement
increment, never `time.time()`. (4) The one lossy field (`failing_nodeids`, cap 50, unstable order)
is reduced to a sorted-deduped sha + length BEFORE any decision; `import_depth` is parsed once at
score time and journaled as an int. (5) All three tiers share `_vector_improved`. The LLM veto, when
enabled, is journaled exactly like `ctx.ask` (cache HIT on resume), so even the opt-in path is
replay-deterministic.

---

## 7. False-kill robustness (the 5 evidenced artifacts)

1. **`tokens=0` timeout artifact** → component 7 reads `len(res.fs_diff)` + changed-files from the
   ON-DISK diff (not usage/tokens), so a 263KB/45-file diff produces a large `diff_useful` rise.
2. **Frozen-committed-diff-but-working** → the in-cell governor re-scores the worktree regardless of
   `finalization_status`, so errors-shrinking / collected-rising / failing-set-churn raise
   components 1/2/4/6; sterile now requires no-new-sha AND no-vector-rise AND no-churn.
3. **Collection-collapse (gold pinned 0)** → components 1,2,4,5,6,7 can all rise while gold=0; a run
   shrinking 5091→4000 errors or advancing the import chain keeps resetting both patience arms.
4. **Guard/policy-abort** → re-scored worktree + diff-string read mean a guard-abort with real edits
   still earns a measurement and a `diff_useful` rise.
5. **The pydantic sterile-streak-8 false cut** → those 8 included guard-aborts/time-kills doing real
   work; under SPFG++ the worktree re-score + churn + diff-bytes reset the sterile arm.

Indeterminate (harness/parser/native-crash/non-gold-downgrade) is neutral to all 8 components and to
both clocks, routed only to `cut:harness-stall`.

---

## 8. True-kill (a genuinely dead arm still dies)

A dead arm has ALL 8 components flat by definition. Routing: dead + empty/repeated diffs →
`cut:sterile-diff-streak` at 8; dead + harness-cannot-run → `cut:harness-stall` at 24; dead +
useless non-empty diffs moving NO component → `cut:no-progress` at the dual-AND window (churn resets
ONLY the sterile arm, never the no-progress arm, so a flailing-but-dead run still plateau-cuts);
explored ceiling → `stop:agent-ceiling`. No infinite run, no clock/token read. SPFG++ also tightens
the SPFG+ false-CONTINUE: useless fresh diffs no longer run to the loose 64 backstop because only a
real component rise (not merely a new sha) resets the no-progress arm.

---

## 9. Cost

Negligible/bounded: 6 of 8 components are arithmetic on counts the scorer already computes (zero
extra cost); `diff_bytes`/`changed_files` = `len()` + one regex over the in-memory diff (once per
attempt); `import_depth` = one bounded regex over the ≤3000-char excerpt; `failing_set_sha` = sha1
over ≤50 sorted ids. WAL grows ~5 sub-kilobyte scalar fields per score record. NO extra pytest runs,
NO LLM tokens, NO network on the default path. The opt-in LLM veto costs at most `VETO_BUDGET=2`
journaled read-only calls per arm and ONLY at a deterministic-kill candidate (never on a climbing
frontier) — and avoids the documented PRM reward-hacking (AgentPRM validation reward rose while real
success fell) because it can never bank acceptance and can only downgrade a kill.

---

## 10. Risks & mitigations

- **`diff_useful` padded with junk to false-continue** → it is a cumulative-bytes BEST (padding adds
  bytes once then plateaus); demotable to sterile-reset-only (one-line flip) if eval shows an
  exploit.
- **`import_depth` heuristic parse (~18% misrate class)** → LOW placement, reset-only, strict-
  increase, capped at 64, fail-neutral on parse failure (can only keep alive, never kill/accept).
- **`failing_nodeids` 50-cap hides shrink/churn on huge suites** → accepted as a stable SAMPLE;
  errors/collected cover the un-sampled bulk.
- **A dead run emitting one churny diff forever** → churn resets ONLY the sterile streak, NOT the
  no-progress dual-AND arm, so it still plateau-cuts.
- **LLM veto decline-misrate (~18%) is a NEW false-kill surface** → mitigated by keeping the seam
  OFF by default; when on it can only DELAY a kill (budget-capped), and a wrong decline just lets
  the deterministic SPFG++ kill proceed (the safe default), so the seam's worst case is ≤ the
  deterministic-only worst case.
- **External validity** → vector reasoned against the 4–7 repo ladder with no held-out repo; report
  as the registered rule, not a tuned optimum. `APEX_FRONTIER_VECTOR` enables a clean SPFG+ vs
  SPFG++ ablation on the next n≥3 re-run.
