"""Type-aware assertion constraints for testgen prompts.

Many AssertionError-bucket failures in our v6 TestGenEvalLite run came
from agents writing assertions whose shape did not match the focal
function's actual return type — most often, comparing a ``dict`` return
to a list literal, asserting truthiness on a ``None``-returning function,
or ``len()``-checking a generator. The downstream pytest run then surfaces
either ``AssertionError`` (wrong value), ``TypeError`` (wrong shape), or
``AttributeError`` (wrong protocol).

This module reads the focal source statically and emits a small block of
human-readable constraints to inject into the agent's prompt so that the
shape of the assertion matches the documented return type. It is general:

* No language- or benchmark-specific paths.
* Pure AST analysis — never imports the focal module (some focal modules
  are not importable on the host, or have side effects on import).
* Conservative — when the annotation is missing, untyped, or too dynamic
  to interpret (``Any``, ``object``, no annotation at all), we emit no
  constraint for that function.

The output is consumed by ``apex.evaluation.prompt_morphs.render_prompt``
via the new ``typed_constraints`` argument and by direct callers in
``testgeneval_benchmark`` that wire up the prompt before the V5 morph.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Iterable, Optional

# Annotations we treat as "no useful information" — they would produce a
# constraint that is either trivially true or actively misleading.
_UNINFORMATIVE_ANNOTATIONS = frozenset(
    {
        "Any",
        "typing.Any",
        "object",
        "T",  # naked typevars
    }
)


@dataclass(frozen=True)
class FunctionReturnConstraint:
    """One focal function's return-shape advice, ready for the prompt."""

    function_name: str
    qualified_name: str  # e.g. "MyClass.do_thing" for methods
    return_annotation: str
    advice: str

    def render(self) -> str:
        return f"- `{self.qualified_name}` → `{self.return_annotation}`: {self.advice}"


@dataclass
class TypedAssertionConstraints:
    """Aggregate constraints for an entire focal source."""

    constraints: list[FunctionReturnConstraint] = field(default_factory=list)
    parse_error: Optional[str] = None

    def is_empty(self) -> bool:
        return not self.constraints

    def render_prompt_block(self, *, max_constraints: int = 10) -> str:
        """Render as a markdown-style bullet list for prompt injection.

        Returns an empty string when there is nothing useful to say —
        callers should test the return value before concatenation so the
        prompt doesn't gain a stray header.
        """

        if self.is_empty():
            return ""
        items = self.constraints[:max_constraints]
        lines = [
            "Return-type constraints (derived from focal annotations):",
        ]
        lines.extend(c.render() for c in items)
        if len(self.constraints) > max_constraints:
            remaining = len(self.constraints) - max_constraints
            lines.append(f"- ... and {remaining} more")
        lines.append(
            "Use these constraints to shape your assertions: prefer "
            "`isinstance` / structural checks that match the return type, "
            "and avoid comparing to literals of an incompatible shape."
        )
        return "\n".join(lines)


def build_typed_constraints(focal_source: str) -> TypedAssertionConstraints:
    """Build constraint advice for every annotated function in *focal_source*.

    Returns an empty ``TypedAssertionConstraints`` when the focal source
    parses successfully but no usable annotations are found. Returns a
    populated ``parse_error`` (and an empty constraint list) when the
    source itself fails to parse — callers should still emit the
    prompt without the constraint block, but the diagnostic is kept so
    upstream telemetry can record it.
    """

    if not focal_source:
        return TypedAssertionConstraints()

    try:
        tree = ast.parse(focal_source)
    except SyntaxError as exc:
        return TypedAssertionConstraints(parse_error=f"focal source SyntaxError: {exc}")

    constraints: list[FunctionReturnConstraint] = []
    for qualified_name, func_node in _iter_annotated_functions(tree):
        annotation = func_node.returns
        if annotation is None:
            continue
        rendered_annotation = _render_annotation(annotation)
        if not rendered_annotation:
            continue
        if rendered_annotation in _UNINFORMATIVE_ANNOTATIONS:
            continue
        advice = _advice_for_annotation(annotation, rendered_annotation)
        if not advice:
            continue
        constraints.append(
            FunctionReturnConstraint(
                function_name=func_node.name,
                qualified_name=qualified_name,
                return_annotation=rendered_annotation,
                advice=advice,
            )
        )

    return TypedAssertionConstraints(constraints=constraints)


def _iter_annotated_functions(
    tree: ast.AST,
) -> Iterable[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Yield ``(qualified_name, node)`` for every function with a return annotation.

    Methods get their qualified name (``ClassName.method``); free
    functions get their bare name. Functions without ``returns`` are
    still yielded so callers can decide what to do — most callers will
    skip those, but oracle-grounding may want to know they exist.
    """

    def walk(node: ast.AST, qualifier: str) -> Iterable[tuple[str, ast.AST]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{qualifier}.{child.name}" if qualifier else child.name
                yield qualified, child
                yield from walk(child, qualified)
            elif isinstance(child, ast.ClassDef):
                next_qualifier = f"{qualifier}.{child.name}" if qualifier else child.name
                yield from walk(child, next_qualifier)
            else:
                yield from walk(child, qualifier)

    for qualified, node in walk(tree, ""):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield qualified, node


def _render_annotation(annotation: ast.AST) -> str:
    """Render an annotation node back to source text. Falls back to ``""`` on error."""

    try:
        return ast.unparse(annotation).strip()
    except Exception:  # pragma: no cover - defensive against odd AST shapes
        return ""


def _advice_for_annotation(annotation: ast.AST, rendered: str) -> str:
    """Build the human-readable assertion-shape advice for one return type.

    Categorization is conservative — we recognize a small set of shapes
    and emit a tight, fact-dense suggestion. Annotations we don't
    recognize get a generic isinstance hint so the agent at least knows
    the declared type.
    """

    kind = _classify_annotation(annotation)

    if kind == "none":
        return (
            "this function returns None — assert the call returns None "
            "(`assert ... is None`); test side effects via observable state, "
            "not the return value."
        )

    if kind == "bool":
        return (
            "boolean return — use `assert ... is True` / `is False` (or `assert <call>`); "
            "do NOT compare to integers or strings."
        )

    if kind == "int" or kind == "float":
        return (
            f"numeric return ({rendered}) — assert the exact numeric value; "
            "for floats use `pytest.approx`. Do NOT compare to a string or list."
        )

    if kind == "str":
        return (
            "string return — assert exact string equality or substring with "
            "`in`; do NOT use `len()`-only checks."
        )

    if kind == "bytes":
        return 'bytes return — assert against a `b"..."` literal; do NOT compare to a `str`.'

    if kind == "list" or kind == "sequence":
        element = _render_subscript_element(annotation)
        elem_hint = (
            f" of `{element}`" if element and element not in _UNINFORMATIVE_ANNOTATIONS else ""
        )
        return (
            f"list/sequence return{elem_hint} — assert against a list literal "
            "or check `len(...)` and per-element values; do NOT compare to a "
            "scalar or dict literal."
        )

    if kind == "tuple":
        return (
            "tuple return — assert against a tuple literal of the right arity; "
            "do NOT compare to a list literal (`(1,2) != [1,2]` in pytest)."
        )

    if kind == "set" or kind == "frozenset":
        return (
            "set return — assert against a set literal `{a, b}` or compare via "
            "`set(...) == {...}`; ordering must NOT be assumed."
        )

    if kind == "dict" or kind == "mapping":
        key_val = _render_dict_key_value(annotation)
        kv_hint = f" with keys/values typed `{key_val[0]}` → `{key_val[1]}`" if key_val else ""
        return (
            f"dict/mapping return{kv_hint} — assert with `isinstance(result, dict)` "
            "and check `result[key] == ...` for known keys. Do NOT compare to a "
            "list literal — `[...] != {...}`."
        )

    if kind == "iterator" or kind == "generator":
        return (
            "iterator/generator return — materialize with `list(...)` before "
            "asserting; do NOT call `len(...)` on the raw return."
        )

    if kind == "optional":
        inner = _render_optional_inner(annotation)
        inner_hint = f" (inner `{inner}`)" if inner else ""
        return (
            f"Optional return{inner_hint} — write at least one test where the "
            "result is `None` AND one where it is the inner type; do NOT assume "
            "non-None without an explicit check."
        )

    if kind == "union":
        return (
            f"union return ({rendered}) — branch your assertions: at least one "
            "test per arm of the union, each guarded by `isinstance(result, ...)`."
        )

    if kind == "callable":
        return (
            "callable return — assert with `callable(result)`; you may invoke "
            "the returned callable in a follow-up assertion if you know its "
            "signature."
        )

    if kind == "literal":
        return (
            f"Literal return ({rendered}) — assert against the exact literal "
            "value(s); the type explicitly forbids any other value."
        )

    # Generic fallback for "we have a type but don't know the family"
    return (
        f"declared return type `{rendered}` — guard the assertion with "
        f"`isinstance(result, {_isinstance_form(rendered)})` and check the "
        "structural properties this type exposes."
    )


def _classify_annotation(annotation: ast.AST) -> str:
    """Categorize an annotation node into a small set of shape kinds."""

    name = _annotation_root_name(annotation)
    lower = name.lower()

    if lower in {"none", "nonetype"}:
        return "none"
    if lower == "bool":
        return "bool"
    if lower == "int":
        return "int"
    if lower == "float":
        return "float"
    if lower == "str":
        return "str"
    if lower == "bytes" or lower == "bytearray":
        return "bytes"
    if lower in {"list", "sequence", "iterable", "collection"}:
        return "sequence" if lower != "list" else "list"
    if lower == "tuple":
        return "tuple"
    if lower in {"set", "frozenset", "mutableset", "abstractset"}:
        return "set"
    if lower in {"dict", "mapping", "mutablemapping", "ordereddict", "defaultdict"}:
        return "dict" if lower == "dict" else "mapping"
    if lower in {"iterator", "asynciterator"}:
        return "iterator"
    if lower in {"generator", "asyncgenerator"}:
        return "generator"
    if lower in {"callable", "awaitable", "coroutine"}:
        return "callable"
    if lower == "optional":
        return "optional"
    if lower == "union":
        return "union"
    if lower == "literal":
        return "literal"

    # ast.Constant(None) — appears when annotation is the literal `None`
    if isinstance(annotation, ast.Constant) and annotation.value is None:
        return "none"

    # ``X | Y`` PEP 604 unions
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        if _binop_includes_none(annotation):
            return "optional"
        return "union"

    return "unknown"


def _annotation_root_name(annotation: ast.AST) -> str:
    """Return the root name of a (possibly subscripted, dotted) annotation."""

    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Attribute):
        return annotation.attr
    if isinstance(annotation, ast.Subscript):
        return _annotation_root_name(annotation.value)
    if isinstance(annotation, ast.Constant):
        return type(annotation.value).__name__
    return ""


def _render_subscript_element(annotation: ast.AST) -> str:
    """For ``list[Foo]`` / ``Sequence[Bar]`` return ``"Foo"`` / ``"Bar"``."""

    if not isinstance(annotation, ast.Subscript):
        return ""
    slice_node = annotation.slice
    # Handle ast.Index for older python / unparse safely
    if isinstance(slice_node, ast.Tuple):
        # e.g. dict[str, int] — first element is key type
        if slice_node.elts:
            return _render_annotation(slice_node.elts[0])
        return ""
    return _render_annotation(slice_node)


def _render_dict_key_value(annotation: ast.AST) -> Optional[tuple[str, str]]:
    """For ``dict[K, V]`` return ``(K_str, V_str)``; otherwise ``None``."""

    if not isinstance(annotation, ast.Subscript):
        return None
    slice_node = annotation.slice
    if isinstance(slice_node, ast.Tuple) and len(slice_node.elts) >= 2:
        key = _render_annotation(slice_node.elts[0])
        value = _render_annotation(slice_node.elts[1])
        if key and value:
            return key, value
    return None


def _render_optional_inner(annotation: ast.AST) -> str:
    """For ``Optional[X]`` return ``X``; for ``X | None`` return ``X``."""

    if isinstance(annotation, ast.Subscript):
        return _render_annotation(annotation.slice)
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        # Find the non-None arm
        for side in (annotation.left, annotation.right):
            if isinstance(side, ast.Constant) and side.value is None:
                continue
            return _render_annotation(side)
    return ""


def _binop_includes_none(binop: ast.BinOp) -> bool:
    for side in (binop.left, binop.right):
        if isinstance(side, ast.Constant) and side.value is None:
            return True
    return False


def _isinstance_form(rendered: str) -> str:
    """Strip subscripts so the rendered annotation is usable inside ``isinstance``.

    ``list[int]`` → ``list``; ``dict[str, int]`` → ``dict``; ``Foo.Bar`` → ``Foo.Bar``.
    Falls back to the raw rendering if we can't safely strip it.
    """

    if "[" in rendered:
        return rendered.split("[", 1)[0].strip()
    return rendered
