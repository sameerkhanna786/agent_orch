""".. QUARANTINED — NOT wired into the evaluated path (fairness directive 2026-06-16) ..

This module derives implementation directives FROM the gold test suite and was previously
injected into the worker + author prompts. That violates the eval fairness contract — the
orchestrator/agents must receive ONLY the original commit0 prompt + the gold test suite, and it
is the model's/orchestrator's job to figure out the API/enum/parametrization shape itself. The
injection was removed from ``commit0_autogen.py`` and ``architect.build_author_prompt``; this
module is retained (unreferenced) for history/reference only and MUST NOT be re-wired into any
evaluated or published comparison.

Gold-test-suite-guided DESIGN CONTRACT (pure, zero-agent, zero-LLM).

Derives a VALUE-STRIPPED "design contract" from the gold expected-test-id inventory
plus the visible test ASTs, so a solver/architect can target the EXACT required API
surface + parametrization (enum dual-render semantics, locale/provider domains,
required ids) WITHOUT being handed answer payloads.

FAIRNESS FIREWALL: a parametrized id's field[0] is kept ONLY if structurally safe
(enum member ``Enum.MEMBER``, BCP-47 locale ``xx``/``xx-yy``, small int, or a name
resolving to a source symbol). Value-bearing tokens — fields[1:], asserted-equal RHS
literals, and non-structural field[0] numbers (luhn/checksum) — are stripped or
shape-redacted. ``contract_is_leak_safe`` audits the rendered blob and the caller
SUPPRESSES the contract on any hit.

ENUM facts are DUAL-REGIME (verified): pytest ``_idval`` checks ``isinstance(str)``
BEFORE ``isinstance(Enum)``, so member-passed tests render ``[Locale.RU]`` (repr ->
needs a PLAIN ``enum.Enum``) while ``values()``-fixture-consumed tests render
``[ru]``/``[en0]``/``[en1]`` (value). The corrected fix is NOT str/StrEnum; it is a
plain Enum + an alias-inclusive ``values()``. No forbidden mixin text is ever emitted
(guarded by a negative test).
"""

from __future__ import annotations

import ast
import os
import re
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Optional

from apex.core._apex_expected_ids_filter import (
    _dynamic_param_shape,
    _generated_ordinal_param_shape,
    _split_parametrized,
)

_LOCALE_RE = re.compile(r"^[a-z]{2}(-[a-z]{2})?$")
_MEMBER_RE = re.compile(r"^[A-Z][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")
_SMALLINT_RE = re.compile(r"^\d{1,3}$")
_DUP_RE = re.compile(r"^(?P<key>.+?)(?P<idx>\d+)$")
_MEMBER_IN_ID_RE = re.compile(r"\[[A-Z][A-Za-z0-9_]*\.")
_VALUE_IN_ID_RE = re.compile(r"\[[a-z]{2}(-[a-z]{2})?[\]\-]")


# --------------------------------------------------------------------------- #
# field[0] structural allowlist + safe extraction
# --------------------------------------------------------------------------- #
def _is_structural(tok: str, symbols=frozenset()) -> bool:
    """True iff ``tok`` is a SAFE parametrization key (never a value-bearing answer)."""
    if not tok:
        return False
    if _MEMBER_RE.match(tok) or _LOCALE_RE.match(tok) or _SMALLINT_RE.match(tok):
        return True
    if tok in symbols:
        return True
    m = _DUP_RE.match(tok)
    return bool(m and (_LOCALE_RE.match(m.group("key")) or m.group("key") in symbols))


def _structural_prefix(params: str, symbols=frozenset()) -> Optional[str]:
    """Longest dash-bounded prefix of ``params`` that is structurally safe (so a
    hyphenated locale like ``en-au`` or a member ``Locale.DE_AT`` survives whole and
    is NOT mangled by a blind first-dash split). None if no safe prefix exists."""
    segs = params.split("-")
    for j in range(len(segs), 0, -1):
        cand = "-".join(segs[:j])
        if _is_structural(cand, symbols):
            return cand
    return None


def _shape_redact(token: str) -> str:
    return _dynamic_param_shape(token) or _generated_ordinal_param_shape(token) or "<value>"


def sanitize_node_ids_for_prompt(node_ids, arity_by_base, symbols=frozenset()):
    """Render node ids safe for a prompt: keep the structural field[0] (the
    parametrization KEY) and strip every value-bearing field. Non-structural field[0]
    is shape-redacted. Unknown arity -> treat as K>=2 (over-strip is safe)."""
    out = []
    for nid in node_ids:
        base, params = _split_parametrized(nid)
        if params is None:
            out.append(nid)
            continue
        k = arity_by_base.get(base) if isinstance(arity_by_base, dict) else None
        if k == 1:
            if _is_structural(params, symbols):
                out.append(f"{base}[{params}]")            # whole bracket is one safe key
            else:
                out.append(f"{base}[{_shape_redact(params)}]")
            continue
        sp = _structural_prefix(params, symbols)            # K>=2 or unknown
        if sp is not None:
            out.append(f"{base}[{sp}-...]")                 # keep key, strip rest
        else:
            out.append(f"{base}[{_shape_redact(params.split('-', 1)[0])}-...]")
    return out


# --------------------------------------------------------------------------- #
# Stage A — arity from AST (parametrize argnames + consumed params-fixtures)
# --------------------------------------------------------------------------- #
def _find_test_root(repo_dir: Path) -> Optional[Path]:
    for name in ("tests", "test", "testing"):
        d = repo_dir / name
        if d.is_dir():
            return d
    return repo_dir if repo_dir.is_dir() else None


def _is_fixture_with_params(dec: ast.AST) -> bool:
    # @pytest.fixture(params=...) / @fixture(params=...)
    if not isinstance(dec, ast.Call):
        return False
    fn = dec.func
    name = getattr(fn, "attr", None) or getattr(fn, "id", None)
    if name != "fixture":
        return False
    return any(kw.arg == "params" for kw in dec.keywords)


def _parametrize_argnames(dec: ast.AST):
    # @pytest.mark.parametrize("a,b", ...) -> ["a","b"]
    if not isinstance(dec, ast.Call):
        return None
    fn = dec.func
    if getattr(fn, "attr", None) != "parametrize":
        return None
    if not dec.args:
        return None
    first = dec.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return [s.strip() for s in first.value.replace(" ", "").split(",") if s.strip()]
    if isinstance(first, (ast.List, ast.Tuple)):
        names = []
        for e in first.elts:
            if isinstance(e, ast.Constant) and isinstance(e.value, str):
                names.append(e.value.strip())
        return names or None
    return None


def _collect_parametrize_arity(repo_dir, test_root):
    """Return (arity_by_base, argnames_by_base). base == 'relpath::[Class::]func'
    relative to repo_dir. Best-effort; a base absent on K-UNKNOWN is the caller's
    fail-closed signal (treat as K>=2)."""
    repo_dir = Path(repo_dir)
    test_root = Path(test_root) if test_root else repo_dir
    files = [p for p in test_root.rglob("*.py")] if test_root.is_dir() else []
    trees = {}
    for f in files:
        try:
            trees[f] = ast.parse(f.read_text(errors="ignore"))
        except Exception:
            continue
    fixture_names = set()
    for t in trees.values():
        for node in ast.walk(t):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    if _is_fixture_with_params(dec):
                        fixture_names.add(node.name)

    arity_by_base, argnames_by_base = {}, {}

    def rel(f: Path) -> str:
        try:
            return f.resolve().relative_to(repo_dir.resolve()).as_posix()
        except Exception:
            return f.name

    def visit(node, relpath, classname):
        for child in getattr(node, "body", []):
            if isinstance(child, ast.ClassDef):
                visit(child, relpath, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test"):
                k = 0
                argnames = []
                for dec in child.decorator_list:
                    names = _parametrize_argnames(dec)
                    if names:
                        k += len(names)
                        argnames.extend(names)
                for a in child.args.args:
                    if a.arg in fixture_names:
                        k += 1
                        argnames.append(a.arg)
                base = f"{relpath}::{classname}::{child.name}" if classname else f"{relpath}::{child.name}"
                if k > 0:
                    arity_by_base[base] = k
                    argnames_by_base[base] = argnames

    for f, t in trees.items():
        visit(t, rel(f), None)
    return arity_by_base, argnames_by_base


def _asserted_equal_values(repo_dir, test_root):
    """RHS string/number literals of ``assert x == <lit>`` across the visible tests —
    candidate answer values that must NOT leak into the contract."""
    repo_dir = Path(repo_dir)
    test_root = Path(test_root) if test_root else repo_dir
    vals = set()
    files = [p for p in test_root.rglob("*.py")] if test_root.is_dir() else []
    for f in files:
        try:
            tree = ast.parse(f.read_text(errors="ignore"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
                ops = node.test.ops
                if ops and isinstance(ops[0], ast.Eq):
                    for cmp in node.test.comparators:
                        if isinstance(cmp, ast.Constant) and isinstance(cmp.value, (str, int, float)):
                            vals.add(str(cmp.value))
    return vals


# --------------------------------------------------------------------------- #
# Stage B — enum DUAL-REGIME facts + param domains
# --------------------------------------------------------------------------- #
def _enum_facts(expected_ids):
    has_member = has_value = False
    enum_names = []
    value_codes = set()
    dup = defaultdict(set)
    for nid in expected_ids:
        _base, params = _split_parametrized(nid)
        if params is None:
            continue
        first = _structural_prefix(params, frozenset()) or params.split("-", 1)[0]
        if _MEMBER_RE.match(first):
            has_member = True
            ename = first.split(".", 1)[0]
            if ename not in enum_names:
                enum_names.append(ename)
        else:
            dm = _DUP_RE.match(first)
            key = dm.group("key") if dm else first
            if _LOCALE_RE.match(key):
                has_value = True
                value_codes.add(key)
                if dm:
                    dup[key].add(int(dm.group("idx")))
    regime = "dual" if (has_member and has_value) else ("repr" if has_member else ("value" if has_value else "none"))
    facts = {"regime": regime, "enums": {}, "dup_index": {k: sorted(v) for k, v in dup.items() if len(v) > 1},
             "value_codes": sorted(value_codes)}
    if regime == "none":
        return facts
    # repr_directive for EVERY enum that appears in member-form ids (e.g. mimesis renders
    # BOTH `[Gender.MALE]` and `[CardType.MASTER_CARD]` — a single-enum contract guided only
    # one of them, so `total=0` on exact-id match. Emit one per enum, cap a handful for size).
    for ename in enum_names[:6]:
        facts["enums"][ename] = {"repr_directive": (
            f"Define `class {ename}(enum.Enum)` as a PLAIN Enum. Tests that pass {ename} MEMBERS "
            f"render their parametrization id as `[{ename}.MEMBER]` (the member repr); changing its "
            f"base class or text rendering breaks those ids.")}
    # Value-regime: some tests are parametrized by an enum's VALUES (consumed via a fixture as
    # `.values()`), rendering `[<value>]` (e.g. `[fr]`). We do NOT invent a class name — the
    # value-form ids carry none — so the directive describes the SHAPE generically and stays
    # fully repo-agnostic (no hardcoded enum/repo knowledge). Keyed under a sentinel so render
    # (which emits only directive TEXT, never the key) surfaces it.
    if has_value:
        d = facts["enums"].setdefault("_value_regime", {})
        d["value_directive"] = (
            "Some tests are parametrized by an enum's VALUES rather than its members (the enum is "
            "consumed via a fixture as `.values()`); their ids render `[<value>]` — the value "
            "string, e.g. a lowercase/hyphenated code like `xx-yy`. Declare that enum's members "
            "with their EXACT value strings (`NAME = '<value>'`), not the member name."
        )
        if facts["dup_index"]:
            d["alias_directive"] = (
                "A value appears more than once (e.g. en0/en1): declare the alias AFTER its base "
                "member and make `values()` ALIAS-INCLUSIVE "
                "(`[m.value for m in cls.__members__.values()]`, NOT `[m.value for m in cls]`)."
            )
    return facts


def _param_domains(expected_ids, arity_by_base, symbols):
    """Ordered, structural-only parametrization keys per test base (inventory order)."""
    domains = OrderedDict()
    for nid in expected_ids:
        base, params = _split_parametrized(nid)
        if params is None:
            continue
        k = arity_by_base.get(base)
        if k == 1:
            # review-fix #11: a bare 1-3 digit token is an ANSWER value for a single-arg test
            # (e.g. test_score[42]), not a safe key — keep members/locales/symbols, redact ints.
            key = params if (_is_structural(params, symbols) and not _SMALLINT_RE.match(params)) else None
        else:
            key = _structural_prefix(params, symbols)
        if key:
            bucket = domains.setdefault(base, [])
            if key not in bucket:
                bucket.append(key)
    # keep only the most-populated handful so the prompt stays compact
    items = sorted(domains.items(), key=lambda kv: len(kv[1]), reverse=True)[:8]
    return OrderedDict(items)


def derive_design_contract(repo_dir, expected_ids, modules=()):
    repo_dir = Path(repo_dir)
    expected_ids = list(expected_ids or [])
    symbols = frozenset(modules or [])
    test_root = _find_test_root(repo_dir)
    arity_by_base, _argnames = _collect_parametrize_arity(repo_dir, test_root)
    enum = _enum_facts(expected_ids)
    domains = _param_domains(expected_ids, arity_by_base, symbols)
    counts = Counter(nid.split("::", 1)[0] for nid in expected_ids)
    return {
        "required_modules": sorted(symbols),
        "required_counts": dict(counts),
        "param_domains": domains,
        "enum_semantics": enum,
        "_arity_by_base": arity_by_base,
    }


# --------------------------------------------------------------------------- #
# Leak validator + renderer
# --------------------------------------------------------------------------- #
def contract_is_leak_safe(contract, expected_ids, asserted_values=frozenset()):
    """Return (ok, hits). Payload = fields[1:] (joined + per-segment) ∪ non-structural
    field[0] ∪ asserted-equal RHS literals. ok iff NONE appear verbatim in the FULL
    (uncapped) rendered contract."""
    blob = render_contract_prompt(contract, max_chars=10 ** 9)
    symbols = frozenset(contract.get("required_modules") or [])
    arity = contract.get("_arity_by_base") or {}
    payload = set()
    for nid in (expected_ids or []):
        _base, params = _split_parametrized(nid)
        if params is None:
            continue
        k = arity.get(_base)
        if k == 1:
            # review-fix #11: smallint single-arg key is an answer value -> payload, not safe.
            if _is_structural(params, symbols) and not _SMALLINT_RE.match(params):
                continue                                    # whole bracket is one safe key
            payload.add(params)                             # smallint / non-structural single key IS payload
            continue
        sp = _structural_prefix(params, symbols)
        if sp is None:
            payload.add(params.split("-", 1)[0])            # non-structural field[0] is payload
            rest = params.split("-", 1)[1] if "-" in params else ""
        else:
            rest = params[len(sp):].lstrip("-")
        if rest:
            payload.add(rest)
            for seg in rest.split("-"):
                if seg:
                    payload.add(seg)
    for v in (asserted_values or []):
        if v:
            payload.add(str(v))
    hits = sorted({p for p in payload if p and len(str(p)) >= 2 and str(p) in blob})
    return (len(hits) == 0, hits)


def render_contract_prompt(contract, max_chars=3500):
    """Priority-ordered render. If there is no enum signature, SUPPRESS entirely
    (the non-enum portion over-specifies a repo like jinja). Truncate from the tail
    (counts first), always keeping the lead enum directives."""
    enum = contract.get("enum_semantics") or {}
    if not enum or enum.get("regime") in (None, "none"):
        return ""
    lead = ["DESIGN CONTRACT (derived from the visible test suite; implement to reproduce the "
            "EXACT expected pytest ids — do NOT modify the tests):"]
    for _ename, d in (enum.get("enums") or {}).items():
        for key in ("repr_directive", "value_directive", "alias_directive"):
            if d.get(key):
                lead.append("- " + d[key])
    if enum.get("dup_index"):
        lead.append("- Duplicate-value pytest indices: " + ", ".join(
            f"{k}->{v}" for k, v in sorted(enum["dup_index"].items())))
    mods = contract.get("required_modules") or []
    mid = []
    if mods:
        mid.append("- Required source modules/symbols: " + ", ".join(map(str, mods[:40])))
    pd = contract.get("param_domains") or {}
    for arg, keys in list(pd.items())[:6]:
        mid.append(f"- Parametrization keys for {arg}: " + ", ".join(map(str, keys[:60])))
    counts = contract.get("required_counts") or {}
    tail = []
    if counts:
        tail.append("- Required test counts per file: " + ", ".join(
            f"{k}={v}" for k, v in sorted(counts.items())[:40]))
    lead_text = "\n".join(lead)
    full = "\n".join(lead + mid + tail)
    if len(full) <= max_chars:
        return full
    body = "\n".join(lead + mid)
    if len(body) <= max_chars:
        return body                                          # dropped counts (tail)
    if len(lead_text) <= max_chars:
        return lead_text                                     # dropped domains too
    return lead_text[:max_chars]                             # keep the lead enum directives


def safe_contract_text(repo_dir, expected_ids, modules=(), max_chars=3500):
    """Single prompt-wiring entry point (pure, zero-agent, zero-LLM).

    Derive + render the design contract, returning ``""`` UNLESS it is provably
    leak-safe — i.e. no value-bearing payload (fields[1:], non-structural field[0],
    or an asserted-equal RHS literal) appears verbatim in the rendered blob. Also
    returns ``""`` when there is no enum signature (render suppresses it) or no
    expected ids. Never raises (a derivation failure -> no contract, not a crash)."""
    expected_ids = list(expected_ids or [])
    if not expected_ids:
        return ""
    try:
        contract = derive_design_contract(repo_dir, expected_ids, modules=modules)
        test_root = _find_test_root(Path(repo_dir))
        asserted = _asserted_equal_values(repo_dir, test_root)
        ok, _hits = contract_is_leak_safe(contract, expected_ids, asserted)
        if not ok:
            return ""
        return render_contract_prompt(contract, max_chars=max_chars)
    except Exception:
        return ""


def redact_excerpts(excerpts, arity_by_base=None, symbols=frozenset()):
    """Reduce a raw pytest failure excerpt to sanitized node ids only (drop RHS value
    payloads). Used only when APEX_OMEGA_REPAIR_EXCERPTS=1; default repair path drops
    excerpts entirely (the base-only sanitized ids carry the Reflexion signal)."""
    ids = re.findall(r"[\w./]+\.py::[\w:.<>\[\]\- ]+", excerpts or "")
    ids = [i.strip() for i in ids]
    return "\n".join(sanitize_node_ids_for_prompt(ids, arity_by_base or {}, symbols))
