"""Minimal, dependency-free, DETERMINISTIC JSON-Schema validator.

This implements the validation half of the dynamic-workflows ``agent(prompt, {schema})``
contract (guide §2.1 / §2.2): a structured reply is validated against the requested
JSON Schema, and on a miss the caller re-asks with a nudge (up to N times) before
returning ``None`` (fail-open) or throwing (``strict``).

Determinism is load-bearing: the error string a miss produces feeds the next nudge
prompt, which is journaled — so the nudge loop must replay byte-identically on resume.
That means: stable paths (no dict/set iteration order surprises — we walk ``required``
and ``properties`` in sorted order), no addresses, no clocks.

Scope: the SUBSET the paradigm's schemas actually use — ``type`` (incl. a list of
types), ``required``, ``properties`` (recursive), ``items``, ``enum``,
``minItems`` / ``maxItems``, ``additionalProperties`` (bool), and ``anyOf`` / ``oneOf``
(lenient). It is NOT a full Draft-2020 implementation; for any construct it does not
model it ERRS TOWARD ACCEPTING (never a false reject), so an unusual-but-valid reply
is never nudged into a loop.
"""

from __future__ import annotations

from typing import Any


def _type_ok(value: Any, t: str) -> bool:
    if t == "object":
        return isinstance(value, dict)
    if t == "array":
        return isinstance(value, (list, tuple))
    if t == "string":
        return isinstance(value, str)
    if t == "integer":
        # bool is an int subclass in Python; a JSON integer is not a boolean.
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "null":
        return value is None
    return True  # unknown type keyword -> accept (never a false reject)


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if value is None:
        return "null"
    return type(value).__name__


def validate_schema(value: Any, schema: Any, path: str = "$") -> tuple[bool, str]:
    """Validate ``value`` against ``schema``. Returns ``(ok, error)`` where ``error`` is a
    stable, human/agent-readable message (empty when ok). A non-dict ``schema`` accepts
    anything (no constraints)."""
    if not isinstance(schema, dict):
        return True, ""

    # anyOf / oneOf — value must satisfy at least one branch (lenient: oneOf treated as anyOf).
    has_combinator = False
    for kw in ("anyOf", "oneOf"):
        branches = schema.get(kw)
        if isinstance(branches, list) and branches:
            has_combinator = True
            errs = []
            for i, sub in enumerate(branches):
                ok, err = validate_schema(value, sub, path)
                if ok:
                    break
                errs.append(err)
            else:
                return False, f"{path}: matched none of {kw} ({'; '.join(e for e in errs if e)[:200]})"

    # type (string or list of strings)
    t = schema.get("type")
    if isinstance(t, str):
        if not _type_ok(value, t):
            return False, f"{path}: expected {t}, got {_kind(value)}"
    elif isinstance(t, list) and t:
        if not any(_type_ok(value, str(tt)) for tt in t):
            return False, f"{path}: expected one of {t}, got {_kind(value)}"

    # enum
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        if value not in enum:
            return False, f"{path}: {value!r} is not one of {enum}"

    # object constraints
    if isinstance(value, dict):
        required = schema.get("required")
        if isinstance(required, list):
            for key in sorted(str(k) for k in required):
                if key not in value:
                    return False, f"{path}: missing required property '{key}'"
        props = schema.get("properties")
        props = props if isinstance(props, dict) else {}
        # additionalProperties:False is enforced ONLY when the schema has no anyOf/oneOf — a
        # matched branch may legitimately contribute properties we don't union here, and the
        # module contract is to ERR TOWARD ACCEPTING (never a false reject).
        if schema.get("additionalProperties") is False and not has_combinator:
            extra = sorted(k for k in value.keys() if k not in props)
            if extra:
                return False, f"{path}: unexpected propert{'y' if len(extra) == 1 else 'ies'} {extra}"
        for key in sorted(props.keys()):
            if key in value:
                ok, err = validate_schema(value[key], props[key], f"{path}.{key}")
                if not ok:
                    return False, err

    # array constraints
    if isinstance(value, (list, tuple)):
        mn = schema.get("minItems")
        if isinstance(mn, int) and len(value) < mn:
            return False, f"{path}: expected at least {mn} items, got {len(value)}"
        mx = schema.get("maxItems")
        if isinstance(mx, int) and len(value) > mx:
            return False, f"{path}: expected at most {mx} items, got {len(value)}"
        items = schema.get("items")
        if isinstance(items, dict):
            for i, el in enumerate(value):
                ok, err = validate_schema(el, items, f"{path}[{i}]")
                if not ok:
                    return False, err

    return True, ""
