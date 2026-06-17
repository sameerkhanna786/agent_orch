# APEX-Ω — Gold-Test-Suite-Guided Design — File-Level Implementation Plan

# FILE-LEVEL IMPLEMENTATION PLAN (all paths absolute)

## Ordered steps
1. NEW `/Users/sameertkhanna/Documents/agent_orch/apex_omega/eval/design_contract.py`
2. NEW `/Users/sameertkhanna/Documents/agent_orch/tests/test_design_contract.py` (firewall + content; MUST be green before any cell run)
3. EDIT `/Users/sameertkhanna/Documents/agent_orch/apex_omega/eval/commit0_autogen.py`
4. EDIT `/Users/sameertkhanna/Documents/agent_orch/apex_omega/autogen/architect.py`
5. EDIT `/Users/sameertkhanna/Documents/agent_orch/apex_omega/autogen/context.py`
6. EDIT `/Users/sameertkhanna/Documents/agent_orch/apex_omega/eval/scoring.py` + `/Users/sameertkhanna/Documents/agent_orch/apex_omega/kernel/verify.py` (opt-in list plumbing)
7. EDIT `/Users/sameertkhanna/Documents/agent_orch/scripts/run_ladder.py` (--only filter, env toggle, scout in parse_result, G6 tripwire)
8. RUN validation A/B (§ validation_plan)
9. NO CHANGE: kernel/select.py, templates.py, budget.py, commit0_benchmark.py acceptance + `_stage_expected_ids_filter`.

## design_contract.py — critical drafted pieces

```python
import ast, os, re
from pathlib import Path
from apex.core._apex_expected_ids_filter import (_split_parametrized, _dynamic_param_shape,
                                                 _generated_ordinal_param_shape)
_LOCALE_RE  = re.compile(r"^[a-z]{2}(-[a-z]{2})?$")
_MEMBER_RE  = re.compile(r"^[A-Z][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")
_SMALLINT_RE= re.compile(r"^\d{1,3}$")
_DUP_RE     = re.compile(r"^(?P<key>.+?)(?P<idx>\d+)$")

def _is_structural(tok, symbols):
    if _MEMBER_RE.match(tok) or _LOCALE_RE.match(tok) or _SMALLINT_RE.match(tok): return True
    if tok in symbols: return True
    m = _DUP_RE.match(tok)
    return bool(m and (_LOCALE_RE.match(m.group("key")) or m.group("key") in symbols))

def sanitize_node_ids_for_prompt(node_ids, arity_by_base, symbols=frozenset()):
    out = []
    for nid in node_ids:
        base, params = _split_parametrized(nid)
        if params is None: out.append(nid); continue
        k = arity_by_base.get(base)
        f0 = params.split("-", 1)[0]
        if k == 1 and _is_structural(f0, symbols):
            out.append(f"{base}[{params}]")                 # whole bracket is one structural key
        elif _is_structural(f0, symbols):
            out.append(f"{base}[{f0}-...]")                 # keep structural key, strip rest
        else:
            shape = _dynamic_param_shape(f0) or _generated_ordinal_param_shape(f0) or "<value>"
            out.append(f"{base}[{shape}-...]")             # shape-redact non-structural field[0]
    return out
```

`_collect_parametrize_arity(test_root)` — two-pass: pass1 discovers `@pytest.fixture(params=...)` names across test files AND `conftest.py`; pass2 per test fn sums `@parametrize` argnames + 1 per consumed fixture (matched by `node.args.args[i].arg in fixture_names`). Returns `(arity_by_base, argnames_by_base)`; base absent on K-unknown. (Verified mandatory: conftest.py:14 `@pytest.fixture(params=Locale.values())`.)

`_asserted_equal_values(test_root)` — walk ASTs, collect RHS `ast.Constant` str of every `assert _ == <lit>`. Feeds the leak audit (#1b).

`_enum_facts(expected_ids)` — DUAL-REGIME (drafted in design §2.3): classify each id's field[0] as `EnumName.MEMBER` (repr) vs locale code (value); collect `<key>N` dup indices; emit per-enum `repr_directive`/`value_directive`/`alias_directive`/`dup_index`. NO StrEnum/`__str__` text anywhere (asserted by a negative test).

`derive_design_contract(repo_dir, expected_ids, modules)` → dict `{required_modules, required_api, required_counts, param_domains(ordered, structural-only), enum_semantics, _arity_by_base}`. `param_domains` admits field[0] only when `k==1 or _is_structural(f0)`.

`contract_is_leak_safe(contract, expected_ids, asserted_values)` — payload = `{fields[1:]} ∪ asserted_values ∪ {non-structural field[0]}`; render uncapped; assert none verbatim in blob. (Closes #1a + #1b.)

`render_contract_prompt(contract, max_chars=3500)` — priority order: enum dual-regime → modules → param_domains+dup → per-file counts (tail-truncate). If `enum_semantics` empty, suppress domains/counts (closes #3 over-spec).

## commit0_autogen.py
- Add module-scope `_contract_gate` (design §3, with env toggle + signature trigger; import `_DIFFICULTY_ORDER` from architect).
- After `repo_map = build_repo_map(...)` (:345): derive once, leak-audit (pass `_asserted_equal_values(str(repo_dir))`), store on `repo_map["design_contract"]` (rendered block) + `repo_map["arity_by_base"]` (dict). On FAIL: `engine.log(...)`, leave block "".
- `prompt_builder` (:275): read idiomatically — `contract_block = (ctx.repo_map.get("design_contract") or "")[:3500]; return issue + contract_block + plan + hint`. (Matches the existing `ctx.repo_map.get("approach")` pattern at :263; avoids the closure-capture fragility flagged in #5.)
- `scout_extra["design_contract"] = repo_map.get("design_contract")` (:339 region) for the autogen scout path.
- Return dict (:382 region): `"leakage_audit": {"gated_on": bool(block), "payload_hits": 0 if block else None}`.

## architect.py
- `build_scout_prompt` (:235): the body whitelist (:239-241) does NOT include `design_contract` and `scout_extra` is `[:2000]`-full (verified) → append an explicit, separately-`[:2500]`-capped line; retarget lens i==1 to "validate/refine the DESIGN CONTRACT; return fidelity_risk + contract_corrections (never expected output values)".
- `SCOUT_SCHEMA` (:213): add `"fidelity_risk":{"type":"string"}`, `"contract_corrections":{"type":"string"}`.
- `agent_scout` (:253): aggregate `fidelity_risk`; if corrections, merge then RE-RUN `contract_is_leak_safe`; on FAIL drop + log (no new agent — reuses the existing fan-out).
- `build_author_prompt` (:127): one explicit `[:2000]`-capped contract line OUTSIDE the full `json.dumps(repo_map)[:6000]` slice; author guidance = route fidelity-risk (stronger vendor / decompose by module), do NOT relax acceptance.

## context.py
- `OrchestrationContext.__init__` (:49): `import time; self._cell_deadline = (time.monotonic()+float(timeout_seconds)) if timeout_seconds else None; self._arity_by_base = dict((repo_map or {}).get("arity_by_base") or {})`. Add `def time_remaining(self): return float('inf') if self._cell_deadline is None else self._cell_deadline - time.monotonic()`.
- Per-agent decoupling (:175, :248): `_agent_to = min(int(self.timeout_seconds or 7200), max(600, int(self.timeout_seconds or 7200)//3))` → pass `timeout_seconds=_agent_to` to ScopedTask. (Closes #4b — the true run-4 root cause.)
- `solve_and_repair` guard (after :309): `eval_cap = max(300, min(1800, int(self.timeout_seconds or 7200)//3)); agent_wc = self._agent_worst_case(); \n if self.time_remaining() < agent_wc + eval_cap: break`.
- `plan_waves` (:340): same guard before launching each base wave (closes #4c — base fan-out blew run-4).
- Repair firewall (:239/:240/:253/:265-266): `from ..eval.design_contract import sanitize_node_ids_for_prompt, redact_excerpts`; replace raw `failing[:30]` with `sanitize_node_ids_for_prompt(failing[:30], self._arity_by_base)`; DROP `excerpts[:1500]` by default (base-only ids carry the Reflexion signal), include `redact_excerpts(...)` ONLY if `os.environ.get("APEX_OMEGA_REPAIR_EXCERPTS")=="1"`; sanitize meta written for the next iteration.

## scoring.py + verify.py (opt-in list plumbing — confirmed multi-line, #5b)
- verify.py: add `missing_test_ids: list = field(default_factory=list)` to `VerificationResult` + to `to_dict` ([:200]).
- scoring.py `verification_from_commit0_evaluation` (:52): populate from `_g("expected_test_coverage",{}).get("missing_test_ids") or []`. (The §5 disjoint trigger uses the existing `missing_expected` count; only repair needs the list.) Acceptance gate untouched.

## run_ladder.py
- Add `--only <comma-list>` filter over ARMS (one-liner in the arm loop).
- `parse_result` (:130/:139): add `"scout": d.get("scout")`, `"agent_budget": d.get("agent_budget")`.
- G6 tripwire in the report path: voluptuous `agents==1 ∧ scout in (None,"null")`; jinja `solved==1 ∧ wall_s < CELL_TIMEOUT`. Emit a `regression` status line if violated.

## Test plan — tests/test_design_contract.py (pure, no subprocess)
1. Leak romanize: `[Locale.RU-привет-privet]` K=3 → contract has `Locale.RU`, `привет`+`privet` ABSENT; `contract_is_leak_safe == (True,[])`.
2. field[0] structural gate (#1a): `[5563455651-2]` K=2 non-allowlist field[0] → emitted `[<digits:10>-...]`, both tokens absent.
3. Source-asserted audit (#1b): a token that LOOKS structural but is an `assert == <tok>` value → flagged, suppressed.
4. Enum DUAL regime: ids `[Locale.HU]`+`[de-at]`+`[ru]`+`[en0]`+`[en1]` → `regime=="dual"`, `repr_directive` keeps PLAIN Enum, `alias_directive` mentions alias-inclusive `values()`+declare-after, `dup_index["en"]==[0,1]`.
5. NEGATIVE no-str-mixin (guards #2 regression): rendered prompt NEVER contains `StrEnum`/`str, Enum`/`__str__`.
6. Hyphenated locale field[0]: K=2 `[en-au-<val>]` from fixture arity → `en-au` survives, `<val>` stripped.
7. Unknown-K fail-closed: base absent → K≥2 strip.
8. Priority cap: small max_chars → lead enum line present, counts dropped.
9. enum-empty suppression: no-enum id set → domains/counts suppressed; `_contract_gate` False for that repo_map.
10. drop-excerpts default: repair sanitizer returns base-only ids, no RHS.
11. Light integration: temp repo (conftest `@fixture(params=Locale.values())` + `test_x[Locale.RU]`) → `derive_design_contract` end-to-end dual+leak-safe.

## Empirical validation-run command
```bash
python -m pytest /Users/sameertkhanna/Documents/agent_orch/tests/test_design_contract.py \
  /Users/sameertkhanna/Documents/agent_orch/tests/ -q
# mimesis TEMPLATE arm, contract OFF then ON, repair OFF (attribute missing->0 to contract alone)
APEX_OMEGA_REPAIR_ITERS=0 APEX_OMEGA_DESIGN_CONTRACT=0 LADDER_CELL_TIMEOUT=3600 \
  python scripts/run_ladder.py --only omega_template_k8 --repos mimesis
APEX_OMEGA_REPAIR_ITERS=0 APEX_OMEGA_DESIGN_CONTRACT=1 LADDER_CELL_TIMEOUT=3600 \
  python scripts/run_ladder.py --only omega_template_k8 --repos mimesis
# autogen arm ON
APEX_OMEGA_REPAIR_ITERS=0 APEX_OMEGA_DESIGN_CONTRACT=1 LADDER_CELL_TIMEOUT=3600 \
  python scripts/run_ladder.py --only omega_autogen_k8 --repos mimesis
# cheap-win tripwire + jinja A/B (signature gate must SKIP jinja)
APEX_OMEGA_DESIGN_CONTRACT=1 LADDER_CELL_TIMEOUT=3600 \
  python scripts/run_ladder.py --only omega_template_k8,omega_autogen_k8 --repos voluptuous,jinja
```
