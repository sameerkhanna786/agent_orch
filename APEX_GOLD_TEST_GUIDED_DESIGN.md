# APEX-Ω — Gold-Test-Suite-Guided Design (both arms): Design + Implementation Plan

> 33-agent improvement workflow (map+ground+research → 4 design proposals → judged → synthesized → 7 adversarial concern-validations → impl-plan + critic + verified final). Every load-bearing claim re-verified against the live repo/runtime.

## Executive summary

FINAL package for the gold-test-suite-guided DESIGN CONTRACT (both arms). I verified every load-bearing claim against the live repo + runtime before writing.

WHAT I CONFIRMED EMPIRICALLY (not from the package text):
1. The original design's CENTRAL content directive is INVERTED. I ran pytest's idval branch logic on py3.14 and read the candidate's real files: candidate is `class Locale(Enum)` (plain) with `DEFAULT = EN` alias and `values()=[item.value for item in cls]` (alias-EXCLUDING). pytest's `_idval` checks `isinstance(str)` BEFORE `Enum`, so converting to `(str,Enum)` renders MEMBER-passed tests `[ru]` (value form) and would NEWLY BREAK the 1491 repr-form ids (`[Locale.RU]`) that need a PLAIN enum. The real, deterministic fix lives in `values()`: return `[m.value for m in cls.__members__.values()]` (alias-INCLUSIVE, two 'en' -> pytest disambiguates en0/en1) with DEFAULT declared after EN. I reproduced both: plain Enum -> `Locale.RU` repr; alias-inclusive values() -> duplicate `en`. This correction is the spine of the final design; the StrEnum/PEP-663 guidance is REMOVED.
2. `@pytest.fixture(params=Locale.values())` in conftest (locale enters via FIXTURE name, not decorator argname) -> AST decorator-only parsing undercounts; conftest fixture-axis resolution is mandatory.
3. `VerificationResult` has only `missing_expected` (count), no `missing_test_ids` list; scoring.py forwards only the count -> §4 list plumbing is a real multi-line cross-boundary change.
4. `OrchestrationContext` is constructed at architect.py:369, NOT commit0_autogen.py:283 -> `_cell_deadline` must be set in `__init__`.
5. Effective `CELL_TIMEOUT=3600` (run_ladder.py:36/178 override the 7200 fn default) -> eval_cap=1200, not 1800.
6. `ScopedTask(timeout_seconds=self.timeout_seconds==cell_timeout)` at context.py:175/248 -> a single agent can eat the whole wall (the true run-4 root cause); the package's loop-deadline alone does not fix this. I add per-agent timeout decoupling.
7. All reused helpers exist exactly (_split_parametrized:42, _dynamic_param_shape:62, _generated_ordinal_param_shape:82, parameterized_node_id_base:526, summarize_expected_pytest_coverage.missing_test_ids:514/522).

The final design keeps strict exact-id acceptance UNTOUCHED, makes field[0] safety STRUCTURAL (not positional luck), closes the live repair-excerpt leak by DROPPING excerpts to base-only sanitized ids (not fragile regex masking), gates the contract on the actual enum-repr-vs-value FIDELITY SIGNATURE (so jinja/pydantic are skipped), is time-budget-aware via per-agent decoupling + loop + base-wave deadline guards calibrated to 3600, and is vendor-agnostic (plain text/dict upstream of executor.spawn).

HONEST SCOPE: verified beneficiary set is {mimesis} (N=1). No empirical end-to-end missing->0 run has been done; that is the required validation. I am calibrated accordingly below.

## Confidence
**Overall: 74%**

CALIBRATED VERDICT: The plumbing, gating, fairness firewall, budget-awareness, and acceptance-immutability are sound and verified against the live code this session. The ONE thing the entire payoff rests on — the mimesis content directive — was INVERTED in the original design and I corrected it with empirical proof (reproduced pytest's str-before-Enum idval branch and the alias-inclusive duplicate; read the candidate's actual enums.py/locales.py/conftest.py). The corrected mechanism (keep plain Enum, make values() alias-inclusive via __members__, declare DEFAULT after EN) is the deterministic fix for the dominant 55.7% repr gap + the en0/en1 duplicate, and it does NOT break the 1491 repr-form ids that the original StrEnum directive would have broken.

WHAT IS NOT YET PROVEN (the honest gap): no full end-to-end scoring run has been done. The claim "missing->0 for mimesis" rests on (a) the verified dominant-mechanism reproduction and (b) the gold inventory — NOT a full v1 scorer pass with a corrected impl. The ~44% finer-rendering tail (beyond the enum-repr + en0/en1 mechanisms) is a "strong target," not a guaranteed fix. The contract_is_leak_safe FP/FN balance and the AST fixture-arity correctness on real mimesis source are designed-correct but unmeasured. jinja's OFF-vs-ON neutrality is now signature-gated (skip) but must be A/B'd before shipping.

The design is implementable as written (all anchors verified, 3 wiring defects fixed), vendor-agnostic, keeps strict exact-id acceptance untouched, honors the fairness boundary structurally, and is time-budget-aware with the true run-4 root cause (per-agent timeout decoupling) now addressed. It is a TARGETED fix for one fidelity regime (data-generation libraries with enum-repr-vs-value parametrization), not a general mechanism — stated explicitly. Ship the firewall + unit tests first; gate the cell run on the validation A/B below.

| concern | confidence | note |
|---|---|---|
| Mimesis ID-fidelity fix is correct and can reach missing->0 | 70% | Corrected directive (plain Enum + alias-inclusive values() + DEFAULT-after-EN) is VERIFIED by reproducing pytest idval branch + alias-inclusive duplicate this session, and matches the candidate's actual bug (values() excludes alias). |
| Fairness / no answer-leak | 84% | Structural field[0] gate + source-asserted-value audit + fail-closed unknown-K + repair drop-excerpts close every leak path the validators found, and make safety structural not positional. |
| No regression of cheap wins (voluptuous, jinja) | 80% | voluptuous structurally gated OFF (verified n=8,149ids). |
| Time-budget-awareness (no run-4 recurrence) | 82% | Derivation is content-not-compute (verified). |
| Both arms genuinely improved | 74% | Template injection verified real (scout=null -> contract is the ONLY enrichment). |
| Implementable + vendor-agnostic | 90% | All anchors verified exact this session; the 3 wiring defects (_cell_deadline location, missing_test_ids multi-line plumbing, scout injection channel) are fixed. |
| Generalization beyond mimesis | 55% | Honestly narrow: verified beneficiary set is {mimesis} (N=1). |

## Residual risks
- missing->0 for mimesis is UNVERIFIED end-to-end. The corrected enum/values() directive is proven for the dominant ~55.7% repr mechanism + en0/en1, but the remaining ~44% finer id-rendering tail may leave residual missing. Only a full v1 scorer pass with a corrected impl settles this.
- Whether an agent RELIABLY acts on the prompt directive (keeps plain Enum, edits values() to be alias-inclusive) is unproven. Run-3 showed agents write correct code, but a structural prompt nudge changing a specific enum/values() decision is a behavioral hypothesis, not a guarantee.
- contract_is_leak_safe false-positive/false-negative balance on the unknown-K and structural-field[0] branches is unmeasured against the real 6159-id inventory; a false-negative would leak, a false-positive would spuriously suppress the contract.
- AST fixture-axis resolution for indirect/conftest params (locale via fixture NAME) is designed but unrun on real mimesis source; if it undercounts K, the fail-closed default over-strips (safe for fairness) but may degrade the contract's fidelity signal.
- jinja signature-gating SKIP is reasoned (no enum-repr regime) but not yet A/B-confirmed; if jinja somehow trips the value-form regex it would receive a contract against the time wall.
- Per-agent timeout cap value (timeout//3) is heuristic; too tight could prematurely kill a legitimately long agent on a heavy repo, too loose under-protects the wall. Needs calibration against per-difficulty run-4 dur_s.
- The validation A/B is contract-OFF-vs-ON within an arm (both firewalled), NOT vs B0, because B0 rides the staged-oracle leak — so the A/B measures the contract's marginal effect, not parity with the leak-riding baseline. The B0 fairness asymmetry remains documented-not-closed (closing it would corrupt the scoring oracle).
- Worst-case mimesis budget (heavy base solve + one coverage-repair iter under 3600) is reasoned, not measured; coverage-repair must be gated off when remaining < agent_worst_case + eval_cap, relying on _checkpoint_accepted to bank the base solve.

## Fairness analysis (answer-leak boundary)

THE ANSWER-LEAK BOUNDARY — what is DERIVED/SURFACED vs WITHHELD, and why it is legitimate under commit0.

LEGITIMATE (surfaced) — all re-derivable from files the agent ALREADY has (the visible tests are the spec):
- Required modules/symbols (from `repo_map["modules"]`), per-file/per-base required test COUNTS, parametrization dimension KEYS (locale codes `de-at`, enum members `Locale.RU`), enum RENDERING REGIME (member-passed→repr vs values()-fixture→value), and the alias/dup-index FACT (`en0/en1`). These are spec-level structure: the agent can read the same `@parametrize`/`@fixture(params=...)` in the test files and pytest --collect-only would render the same ids.
- The enum directive surfaces a STRUCTURAL property ("keep plain Enum; make values() alias-inclusive"), not an answer. It tells the agent how to make ids RENDER correctly, which the agent must do anyway to pass the visible suite.

WITHHELD (the firewall) — anything that embeds an expected OUTPUT:
- fields[1:] of every K≥2 id are ALWAYS stripped (e.g. `test_romanize_cyrillic_string[Locale.RU-привет-privet]` → only `Locale.RU` survives; `привет` and `privet` are stripped and asserted absent).
- field[0] is kept ONLY IF it matches the structural allowlist (enum member / BCP-47 locale / small-int / source symbol). A unique non-allowlist field[0] (verified: `test_luhn_checksum[5563455651-2]`, `test_calculate_checksum[030670890-2]`) is shape-redacted to `<digits:N>`. This makes field[0] safety STRUCTURAL, closing the keep-field[0]-vs-uniqueness contradiction the validators flagged.
- The leak validator's payload set now = `{fields[1:]} ∪ {RHS literals of asserts in visible test source} ∪ {non-structural field[0]}`, and it asserts NONE appear verbatim in the rendered contract. Auditing source-asserted values (not just id fields[1:]) is the only STRUCTURAL guarantee (closes the positional-luck dependency: mimesis happens to put every output column in fields[1:], but the audit no longer relies on that).
- K-UNKNOWN ⇒ fail-closed to K≥2 (over-strip a key tail rather than risk leaking a value). AST arity includes conftest fixture axes (verified mandatory — locale enters via fixture name), so hyphenated value-form keys (`en-au`) are bounded correctly, not mangled.

THE LIVE REPAIR LEAK (the genuine open channel, now closed): context.py:239/240 injected raw failing nodeids + raw pytest tracebacks (which contain the expected output, e.g. `assert 'Foo' == 'Likid Geimfari'`). Default fix: DROP excerpts entirely; pass only base-only sanitized failing ids. The legitimate Reflexion signal is WHICH base failed; the assertion RHS is the answer. Regex RHS-masking is fragile against multi-line/unicode-escaped output, so it is opt-in only, never default.

BOUNDARY RUN TWICE: `contract_is_leak_safe` runs before injection AND after the autogen scout merges `contract_corrections`, so an LLM scout cannot re-introduce a payload. FAIL ⇒ suppress + `leakage_audit: FAIL`.

KNOWN ASYMMETRY (documented, not corrupted): B0/baseline also stages `_stage_expected_ids_filter`'s value-bearing id file into the solve repo — a leak on the baseline path. We do NOT touch that file because it is the SCORING ORACLE (commit0_benchmark.py:14256). This means a contract-ON arm (firewalled) vs B0 (leak-riding) A/B is CONFOUNDED — so the validation A/B is contract-OFF-vs-ON within the SAME arm (both firewalled), NOT against B0. This is a validity threat I state explicitly rather than hide.

WHY ACCEPTANCE STAYS FAIR: the contract is prompt-side text only. Strict exact-id acceptance (`total>0 ∧ failed==0 ∧ errors==0 ∧ missing==0 ∧ pass_rate>=1.0`) is mechanically untouched. A mis-derived or over-stripped contract can only ever make the prompt LESS informative; it can NEVER reject a genuinely-passing candidate or accept a failing one.

## Validation plan

REQUIRED EMPIRICAL VALIDATION (the design is NOT validated until these pass).

PHASE 0 — firewall green before any cell run:
- `python -m pytest /Users/sameertkhanna/Documents/agent_orch/tests/test_design_contract.py /Users/sameertkhanna/Documents/agent_orch/tests/ -q` (existing 92 + new must be green). Critical assertions: romanize privet/привет ABSENT; luhn field[0] number shape-redacted; source-asserted value flagged; dual-regime directives present with NO StrEnum/str,Enum/__str__ text; en0/en1 dup_index==[0,1]; hyphenated en-au survives; enum-empty suppresses domains.

PHASE 1 — settle the #2 content claim END-TO-END (the only test that proves missing->0):
- In a mimesis worktree, apply the corrected directive (keep plain Enum; change locales/enums values() to `[m.value for m in cls.__members__.values()]`; DEFAULT after EN). Run the FULL v1 scorer (real pytest + exact-id contract). MEASURE: missing==0 (target) AND confirm the 1491 repr-form ids `[Locale.RU]` STILL match (the original directive would have broken them). If missing>0, report the residual class (which files/ids) — this quantifies the ~44% tail honestly.

PHASE 2 — contract A/B on mimesis (repair OFF so missing->0 is attributable to the contract alone):
- TEMPLATE arm OFF (APEX_OMEGA_DESIGN_CONTRACT=0) vs ON (=1), LADDER_CELL_TIMEOUT=3600, APEX_OMEGA_REPAIR_ITERS=0, `--only omega_template_k8 --repos mimesis`. This isolates the template arm (the one with NO prior fidelity diagnosis; scout=null -> contract is the only enrichment).
- AUTOGEN arm ON (scout validates/refines). MEASURE per cell: missing (target ==0 or strictly < OFF), leakage_audit.payload_hits==0.

PHASE 3 — LEAK CONFIRMATION (no answer reaches the solver):
- Grep the persisted solver prompt + cell narration for: `привет`, `privet`, any romanize output, any `5563455651`/`030670890`-style luhn input, any `Likid Geimfari`-style asserted output. ALL must be ABSENT. Confirm contract_is_leak_safe FP/FN on the full 6159 inventory (run it standalone over the real ids + asserted-value set; report counts).

PHASE 4 — CHEAP-WIN tripwire (must not regress):
- `--only omega_template_k8,omega_autogen_k8 --repos voluptuous,jinja`, contract ON. ASSERT: voluptuous agents_used==1 ∧ scout==null (contract gated OFF); jinja SKIPPED by signature gate (contract block empty) AND solved==1 ∧ wall_s < 3600 (within ~1.5x of the 607s baseline). If jinja receives a non-empty contract, the signature gate failed — block ship.

PHASE 5 — BUDGET safety:
- Confirm contract-ON adds no measurable wall-clock vs OFF (derivation is ms). Confirm per-agent ScopedTask timeout is capped below the cell wall in logs. Confirm no cell exceeds 3600 + the harness 600s grace.

GATE TO SHIP: Phase 0 + Phase 1 (missing==0 with 1491 repr-ids intact) + Phase 3 (no leak) + Phase 4 (no cheap-win regression). Phase 2/5 quantify the marginal win. n>=3 seeds for Phase 2 before claiming the coin-flip is deterministic.

---

# FINAL DESIGN — Gold-Test-Suite-Guided DESIGN CONTRACT (both arms)

## 0. The one correction the whole payoff rests on (VERIFIED this session)

The original design's lead directive (`class Locale(str, Enum)` so `str(Locale.DE_AT)=="de-at"`) is **provably backwards**. Verified two ways:

- **Runtime**: pytest's `_idval` checks `isinstance(val,(str,bytes))` BEFORE `isinstance(val, enum.Enum)`. I reproduced on py3.14: a plain-Enum member renders `Locale.RU` (repr); a `(str,Enum)` member renders `ru` (value). The mimesis venv is py3.10/pytest-7.4.4 — same branch ordering.
- **Gold inventory**: 1491 ids (24.2%; test_base/test_transport/test_schema/test_pytest use `@parametrize('locale', list(Locale))`) require the REPR form `[Locale.RU]`. Converting to `(str,Enum)` fixes value-form fixtures but BREAKS those 1491 ids — it shifts the missing set, never reaches missing→0.
- **Candidate files (read)**: `enums.py:30` `class Locale(Enum)` (plain), `:70` `DEFAULT = EN` (alias), `:73-75` `values()=[item.value for item in cls]`. `conftest.py:14` `@pytest.fixture(params=Locale.values())`.

**The real, deterministic fix** (reproduced this session):
- Keep `Locale` a PLAIN `enum.Enum` (preserves the 1491 repr-form ids).
- `Locale.values()` must be alias-INCLUSIVE: `return [m.value for m in cls.__members__.values()]` (candidate's `[item.value for item in cls]` skips aliases). I verified: alias-inclusive yields `['en','ru','de-at','en']` (two `en`) → pytest disambiguates to `en0`/`en1`; alias-excluding yields a single `en`.
- Declare `DEFAULT = EN` AFTER `EN` so the duplicate orders as `en0`(EN) then `en1`(DEFAULT).

This is a **DUAL-RENDERING regime per file** (member-passed → repr `[Locale.RU]`; `values()`-fixture-consumed → value `[ru]`/`[en0]`/`[en1]`), NOT a single global enum-base property. The contract conveys both, derived from whether the id's first field is `EnumName.MEMBER` vs a bare code. `en0/en1` becomes DETERMINISTIC (the original §1.3 hedge is removed).

## 1. Spine + the mechanism

Spine = **architect-as-test-analyst**: AST-arity-grounded value-stripping as step 0, double leak-check (before injection AND after scout merge). Grafts: loop-level time deadline + disjoint trigger (gold-coverage-oracle); single `prompt_builder` injection + per-cell `leakage_audit` (static-test-contract); difficulty/size+SIGNATURE gating, priority-ordered emit, explicit scout/author lines (lean-contract-preamble).

**Hard constraints baked in:** (1) reuse only the importable module-level helpers (`parameterized_node_id_base`, `_split_parametrized`, `_dynamic_param_shape`, `_generated_ordinal_param_shape`) — NOT the nested helpers in `summarize_expected_pytest_coverage`. (2) Do NOT strip `_stage_expected_ids_filter`'s file — it is the scoring oracle (commit0_benchmark.py:14256 via `APEX_EXPECTED_IDS_FILE`). (3) Scout/author injection needs explicit, separately-capped fields (`scout_extra` is `[:2000]`-full; `json.dumps(repo_map)[:6000]` is full).

## 2. The new primitive — `apex_omega/eval/design_contract.py` (NEW, pure, zero-agent)

Computed once per cell over already-loaded `expected_ids` + on-disk test ASTs. No LLM, no pytest, no second `--collect-only`. Public surface: `derive_design_contract`, `render_contract_prompt`, `contract_is_leak_safe`, `sanitize_node_ids_for_prompt`, `redact_excerpts` (used in degraded/base-only mode — see §4), `_collect_parametrize_arity`, `_asserted_equal_values`.

### 2.1 Stage A — ground arity from AST INCLUDING conftest fixtures (the firewall's foundation)
Walk every test file AND conftest.py up the `tests/` tree. For each base `path::Class::method`: K = Σ argnames across stacked `@pytest.mark.parametrize` + 1 per consumed `@pytest.fixture(params=...)` axis (resolved by parameter NAME, since the mimesis locale enters via fixture name `person`/`generic`, NOT an argname `locale` — verified in conftest.py:14-49). K UNKNOWN ⇒ base absent ⇒ caller FAIL-CLOSES to K≥2.

### 2.2 Stage B — value-stripping (the fairness firewall), with STRUCTURAL field[0]
This closes the critique's #1a/#1b and the unreconciled keep-field[0]-vs-uniqueness contradiction:
- Split base/params via `parameterized_node_id_base`.
- **field[0] is kept ONLY IF it matches the structural allowlist** (enum member `EnumName.MEMBER`; BCP-47 locale `[a-z]{2}(-[a-z]{2})?` incl. `en-au`/`de-at`/`pt-br`; small-int length; or a name resolving to a source symbol in `modules`). Else field[0] is shape-redacted to `<digits:N>`/`<ordinal>`/`<value>`. (Verified necessity: `test_luhn_checksum[5563455651-2]`, `test_calculate_checksum[030670890-2]` put a unique non-allowlist NUMBER in field[0]; the old keep-always rule leaked it.)
- `K==1`: whole bracket is one key → SAFE, keep (but still allowlist-checked for the field[0] number case).
- `K≥2`: field[0] per the rule above; fields[1:] ALWAYS stripped.
- `K UNKNOWN`: treat as K≥2 (over-strip is safe).
- Numeric/ordinal → `<digits:N>`/`<ordinal>` via the importable shape helpers. Never emit a literal number.
- **Hyphenated value-form field[0]** (`en-au`) is grounded by fixture arity, NOT a blind first-dash split, so it is NOT mangled to `en`. The allowlist re-admits it whole.

### 2.3 Enum DUAL-REGIME facts (the corrected mimesis fix; replaces old §1.2/§1.3)
For each enum referenced in ids, emit, per-regime, derived from whether the id field is `EnumName.MEMBER` (repr) vs a bare code (value):
- `repr_directive` (when repr-form ids present): "Keep `class Locale(enum.Enum)` (PLAIN). Tests passing MEMBERS render `[Locale.MEMBER]`; converting to str/StrEnum BREAKS those." (NO StrEnum/`__str__` text.)
- `value_directive` (when value-form codes present): "Tests consuming `Locale.values()` fixtures render `[<code>]`; define members `NAME='code'` (lowercase, hyphenated)."
- `alias_directive` (when `<key>N` duplicates present): "A value appears twice (en0/en1): declare the ALIAS (e.g. DEFAULT=EN) AFTER its base, and make `values()` ALIAS-INCLUSIVE: `[m.value for m in cls.__members__.values()]` (NOT `[m.value for m in cls]`)."
- `dup_index{key:[idx...]}` (e.g. `en:[0,1]`), `param_domains{arg:[ordered keys]}` (inventory order, structural-only).

### 2.4 Leak validator (first-class, structural, run twice)
`contract_is_leak_safe(contract, expected_ids, asserted_values)` builds the payload set = `{fields[1:] across the inventory}` ∪ `{asserted-equal RHS literals parsed from visible test source}` ∪ `{any field[0] that is NOT structural}`, then asserts NONE appear verbatim in the FULL (uncapped) rendered contract. (Closes #1a AND #1b: field[0] and the source-asserted values are now audited, not just fields[1:].) FAIL ⇒ suppress contract, log `leakage_audit: FAIL`. Re-run AFTER the scout `contract_corrections` merge so an LLM scout cannot re-introduce a payload.

### 2.5 `render_contract_prompt` — priority-ordered, truncate-from-tail, cap 3500
Order: (1) enum dual-regime directives (lead, never truncated); (2) required modules/symbols; (3) param domains + dup-index; (4) per-file counts (tail; first to drop). **If `enum_semantics` is empty, SUPPRESS the param-domain/count dump** (closes #3: the ~44% non-enum portion is the part most likely to over-specify a no-enum repo like jinja).

## 3. Gating — protect the cheap wins via the FIDELITY SIGNATURE, not size (closes the jinja gap)

```python
def _contract_gate(repo_map, expected_ids):
    if os.environ.get("APEX_OMEGA_DESIGN_CONTRACT", "1") == "0": return False  # A/B toggle
    static = str(repo_map.get("difficulty") or "").lower()
    if _DIFFICULTY_ORDER.get(static, 1) == 0: return False          # easy -> OFF (voluptuous)
    if (repo_map.get("n_source_files") or 0) < 15 or len(expected_ids) < 200: return False
    # FIDELITY-SIGNATURE trigger: only fire when ids exhibit the enum-repr-vs-value regime.
    has_member = any(re.search(r"\[[A-Z][A-Za-z0-9_]*\.", t) for t in expected_ids[:4000])
    has_value  = any(re.search(r"\[[a-z]{2}(-[a-z]{2})?[\]-]", t) for t in expected_ids[:4000])
    return bool(has_member or has_value)
```
Consequences (verified repo facts): voluptuous (easy, n=8, 149 ids) → False. **jinja (n=33, 851 ids, NO enum-repr regime)** → signature False → SKIPPED (this is the fix; the old size-only gate fired on jinja). pydantic (import-cascade, passed=0) → no value/member id signature in practice AND covered by the futility tripwire below. mimesis → True.

**Futility tripwire** (closes #7a): if a repo's prior attempts in this cell have only ever produced `errors!=0` or `passed==0`, skip the contract (no point steering a candidate that cannot import). For the first attempt this is unknowable, so the signature gate + size floor are the a-priori guard; the futility check applies to any contract-seeded repair.

## 4. Close the LIVE repair leak — DROP excerpts, sanitize ids (closes #1c)

The genuine answer-leak is on the repair path (context.py:239 raw `failing[:30]`, :240 raw `excerpts[:1500]`). The original regex-masking of `assert ==` RHS is fragile against multi-line pytest rewrite output (`E assert X==Y`, `+ where`, unicode-escaped cyrillic). **Decision: default to dropping excerpts entirely** and passing only base-only `sanitize_node_ids_for_prompt(failing, arity_by_base)`. The legitimate Reflexion signal is WHICH base failed; the assertion RHS IS the answer. `redact_excerpts` is retained only as an opt-in (`APEX_OMEGA_REPAIR_EXCERPTS=1`) with a real-traceback unit test, never the default. Sanitize :239/:240/:253 and the meta `failing_nodeids`/`failure_excerpts` at :265-266 so the leak cannot propagate down the lineage. `arity_by_base` is threaded onto ctx. Runtime-inert at the default `repair_iters=0` (stated honestly); lands the firewall for when repair is enabled.

## 5. Time-budget-awareness — calibrated to 3600, per-agent decoupling added (closes #4)

- **Derivation is content, not compute**: one in-process AST+string pass, hoisted out of the per-attempt closure, once per cell (~ms, zero agents/pytest/waves). Cannot reproduce the run-4 4000s blowout.
- **Calibration corrected**: read `ctx.timeout_seconds` at runtime (3600 at the ladder wall, NOT 7200). `eval_cap = max(300, min(1800, timeout//3)) = 1200` at 3600. The original "default is 7200" self-correction was the error.
- **Per-agent timeout DECOUPLING (the verified true run-4 root cause)**: cap each `ScopedTask.timeout_seconds` (context.py:175/248) at `min(self.timeout_seconds, self.timeout_seconds//3 or a per-difficulty value)` so one hung agent cannot consume the whole wall. This is the highest-leverage fix and was absent from the package; I add it.
- **Loop-level deadline**: `ctx._cell_deadline = monotonic()+timeout_seconds` set in `OrchestrationContext.__init__` (NOT commit0_autogen.py:283 where no ctx exists — wiring defect fixed); `ctx.time_remaining()`. Guard in `solve_and_repair` (after :309) AND before each base wave in `plan_waves` (:340 — base fan-out was what actually blew run-4): require `time_remaining() >= agent_worst_case + eval_cap` before launching. `agent_worst_case` is per-difficulty (~1700s heavy / ~400s light from run-4 dur_s).
- **Disjoint trigger** for any contract-seeded repair: only `pass_rate>=1.0 ∧ missing>0` (mimesis signature; disjoint from the failure-repair gate). Repair stays default-OFF.
- `_checkpoint_accepted` (context.py:116) untouched — any verified solve banks instantly even if the cell clips the wall.

## 6. Strict exact-id acceptance + fairness — UNCHANGED, mechanically enforced

No edits to `scoring.py` acceptance, `kernel/verify.py` gate, `kernel/select.py`, `summarize_expected_pytest_coverage`, or the staged scoring-oracle file. Acceptance stays `total>0 ∧ failed==0 ∧ errors==0 ∧ missing==0 ∧ pass_rate>=1.0`. The contract is prompt-side guidance only; a mis-derived contract can NEVER reject a passing candidate. Fairness enforced by: AST-arity (incl. fixtures) + structural field[0] + fail-closed unknown-K + `contract_is_leak_safe` (twice, with source-asserted-value audit) + the repair firewall (drop-excerpts default) + per-cell `leakage_audit` + the shipped unit tests.

## 7. Honest scope (closes #6, #7)
- Both-arms RUNTIME benefit at default config is concentrated in §2.1 (prompt contract). §4 firewall + §5 loop-deadline are code-present but runtime-inert at `repair_iters=0`; the per-agent decoupling and base-wave guard ARE active. Stated plainly.
- Template arm: contract is the ONLY structural enrichment (scout=null → empty plan), so high-leverage there; but per-arm effect is UNPROVEN — §9 A/Bs the template arm in isolation.
- Beneficiary set in the verified corpus is exactly {mimesis} (N=1). jinja's win was a scoring-bug fix; voluptuous gated off; pydantic categorically import-broken. This is a TARGETED fix for the enum-repr/heavy-parametrization (data-generation) regime, not a general mechanism. Predictive repo property: heavy enum/locale parametrization with repr-vs-value divergence.

## 8. What to keep / cut
KEEP: the new pure module, the dual-regime enum facts, structural field[0], the source-asserted-value audit, the signature gate, per-agent decoupling, loop+base-wave deadline, drop-excerpts repair default, the unit-test firewall, the env A/B toggle + `--only` filter.
CUT from the original: the `(str,Enum)`/StrEnum/PEP-663 directive (empirically wrong); the keep-field[0]-unconditionally rule; the regex-RHS-masking-as-default excerpt redactor; the "7200" calibration; the en0/en1 "not guaranteed-deterministic" hedge; the size-only jinja gate.
