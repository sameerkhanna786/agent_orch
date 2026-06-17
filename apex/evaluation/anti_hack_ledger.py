"""V5 anti-hack oracle-evidence ledger (APEX Novelty 1).

Test-generation models reliably "hack" weak oracles: write assertions
whose expected values come straight out of the LLM's head, with no
grounding in actual program behavior. Such tests pass the F→P harness
when the LLM happens to guess right and fail it when the guess is
wrong — pure noise as a quality signal.

This ledger:
  1. Inspects each assertion in a generated test and tags it with one
     of these provenance labels:
       * ``ground_truth_executed`` — value captured from a real program
         run (W4 oracle_repair, W7 hierarchical_gap_fill probing, etc.)
       * ``existing_test_copied`` — value lifted from the project's own
         test suite (pre-existing oracle, presumed trustworthy)
       * ``signature_derived`` — value derived from a type / dataclass
         signature (e.g. ``isinstance(x, MyClass)``, len-checks)
       * ``llm_fabricated`` — none of the above, value invented by the
         model. Treated as untrustworthy by default.
       * ``loose`` — assertion that doesn't actually constrain output
         (``assert x is not None``, ``assert len(x) >= 0``, ``assertTrue(x)``)
  2. Computes a per-test ``hack_score`` ∈ [0, 1] = fraction of
     assertions tagged as ``llm_fabricated`` or ``loose``.
  3. Provides a static gate: tests above ``hack_score_max`` are
     rejected before they even reach the docker run; tests at the
     borderline get a downweight in cross-candidate voting.

The ledger is conservative: when in doubt, it tags as
``llm_fabricated`` and lets the dual-version verifier + critic decide
whether the assertion holds up under execution. We only HARD-REJECT
tests where the *entire* assertion set is loose or unsupported.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


_LOOSE_CALL_NAMES = {
    "assertTrue",
    "assertFalse",
    "assertIsNotNone",
    "assertIsNone",
}


@dataclass(frozen=True)
class AssertionProvenance:
    test_function: str
    line: int
    snippet: str
    label: str  # one of the provenance labels above
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LedgerReport:
    test_id: str
    assertions: list[AssertionProvenance] = field(default_factory=list)
    hack_score: float = 0.0
    rejected: bool = False
    reject_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "assertions": [a.to_dict() for a in self.assertions],
            "hack_score": self.hack_score,
            "rejected": self.rejected,
            "reject_reason": self.reject_reason,
        }


def build_ledger(
    *,
    test_id: str,
    test_source: str,
    captured_oracle_values: Optional[dict[str, Any]] = None,
    existing_test_source: str = "",
    focal_signature_summary: str = "",
    hack_score_max: float = 0.6,
    require_at_least_one_grounded: bool = True,
) -> LedgerReport:
    """Inspect ``test_source`` and build a per-assertion provenance ledger.

    Args:
        captured_oracle_values: dict mapping assertion-key strings (an
            opaque hash from the oracle_repair / gap_fill steps) to the
            captured value. Any assertion whose comparison RHS hashes
            to a known key is tagged ``ground_truth_executed``.
        existing_test_source: full text of the project's pre-existing
            test suite (or relevant slice). Assertion lines that appear
            verbatim are tagged ``existing_test_copied``.
        focal_signature_summary: text summary of the focal class /
            dataclass signature, used to spot ``isinstance(x, T)`` and
            shape-based assertions which we tag ``signature_derived``.
        hack_score_max: tests above this fraction of fabricated/loose
            assertions are rejected outright.
        require_at_least_one_grounded: when True, any test whose
            assertions are 100% fabricated/loose is rejected even if
            the hack_score is below the cutoff (defends against
            single-assertion-fabrication tests passing the gate).
    """

    captured = captured_oracle_values or {}
    try:
        tree = ast.parse(test_source)
    except SyntaxError as exc:
        return LedgerReport(
            test_id=test_id,
            assertions=[],
            hack_score=1.0,
            rejected=True,
            reject_reason=f"unparseable: {exc.msg}",
        )

    assertions: list[AssertionProvenance] = []
    for func_name, node in _iter_test_functions(tree):
        for assertion_node in _iter_assertions(node):
            snippet = _snippet_for(assertion_node, test_source)
            label, reason = _label_assertion(
                node=assertion_node,
                snippet=snippet,
                captured_oracle_values=captured,
                existing_test_source=existing_test_source,
                focal_signature_summary=focal_signature_summary,
            )
            assertions.append(
                AssertionProvenance(
                    test_function=func_name,
                    line=getattr(assertion_node, "lineno", 0),
                    snippet=snippet,
                    label=label,
                    reason=reason,
                )
            )

    hack_score = _compute_hack_score(assertions)
    rejected, reject_reason = _decide_rejection(
        assertions=assertions,
        hack_score=hack_score,
        hack_score_max=hack_score_max,
        require_at_least_one_grounded=require_at_least_one_grounded,
    )
    return LedgerReport(
        test_id=test_id,
        assertions=assertions,
        hack_score=hack_score,
        rejected=rejected,
        reject_reason=reject_reason,
    )


def downweight_oracle_score(
    *,
    raw_oracle_score: float,
    hack_score: float,
    full_kill_threshold: float = 0.85,
) -> float:
    """Apply a soft penalty to a test's oracle_score based on hack_score.

    Score curve: ``hack_score`` 0 → no penalty (multiplier 1.0).
    ``hack_score`` ≥ ``full_kill_threshold`` → multiplier 0.0.
    Linear in between. Returned as a float so the cross-candidate
    voter can keep tie-breaking precision.
    """

    if raw_oracle_score <= 0:
        return 0.0
    if hack_score <= 0:
        return float(raw_oracle_score)
    if hack_score >= full_kill_threshold:
        return 0.0
    multiplier = max(0.0, 1.0 - (hack_score / full_kill_threshold))
    return float(raw_oracle_score) * multiplier


def _iter_test_functions(tree: ast.AST):
    """Yield ``(name, function_node)`` for module-level, class-level, and
    conditionally-imported test functions.

    Audit H3: the previous implementation only yielded module-scope
    ``test_*`` functions. ``unittest.TestCase`` methods get
    ``ClassName.method_name``.

    M-1 fix: also walk module-level ``If``/``Try`` bodies so tests
    defined inside ``try/except ImportError:``, ``if PY_VERSION >= ...``
    guards, or ``if HAS_DEP:`` conditional imports are still counted.
    Without this, conditionally-defined tests collapse to zero
    assertions and the ledger rejects the candidate as
    "no assertions detected".
    """

    def _emit(name: str, node):
        yield name, node

    def _walk_body(body, prefix: str = ""):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_") or node.name == "test":
                    name = f"{prefix}.{node.name}" if prefix else node.name
                    yield name, node
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                        child.name.startswith("test_") or child.name == "test"
                    ):
                        cls_prefix = f"{prefix}.{node.name}" if prefix else node.name
                        yield f"{cls_prefix}.{child.name}", child
            elif isinstance(node, ast.If):
                yield from _walk_body(node.body, prefix=prefix)
                yield from _walk_body(node.orelse, prefix=prefix)
            elif isinstance(node, ast.Try):
                yield from _walk_body(node.body, prefix=prefix)
                for handler in node.handlers:
                    yield from _walk_body(handler.body, prefix=prefix)
                yield from _walk_body(node.orelse, prefix=prefix)
                yield from _walk_body(node.finalbody, prefix=prefix)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                yield from _walk_body(node.body, prefix=prefix)

    yield from _walk_body(getattr(tree, "body", []))


def _iter_assertions(func_node: ast.AST):
    """Yield AST nodes that look like assertions inside a test body."""

    for node in ast.walk(func_node):
        if isinstance(node, ast.Assert):
            yield node
        elif isinstance(node, ast.Call):
            attr = node.func
            name = getattr(attr, "attr", None) or getattr(attr, "id", None)
            if name and name.startswith("assert"):
                yield node


def _snippet_for(node: ast.AST, source: str) -> str:
    try:
        text = ast.unparse(node)
    except Exception:
        text = ""
    return (text or "").strip()[:200]


def _label_assertion(
    *,
    node: ast.AST,
    snippet: str,
    captured_oracle_values: dict[str, Any],
    existing_test_source: str,
    focal_signature_summary: str,
) -> tuple[str, str]:
    """Decide a label + short reason for one assertion.

    Audit H8: ``captured_oracle_values`` is checked BEFORE the loose
    pattern so a runtime-captured ``assert x is not None`` (whose value
    we DID observe) gets credited as ``ground_truth_executed`` instead
    of being demoted to ``loose``. The oracle-first order is also the
    correct semantics — execution evidence trumps surface heuristics.

    Audit H7: ``existing_test_copied`` now uses an AST-shape match
    (variable names + literals normalized) instead of fragile substring
    matching. The previous substring check on truncated ``ast.unparse``
    output flipped labels across Python minor versions and miscredited
    coincidental matches like ``assert result == 1``.
    """

    if captured_oracle_values:
        if _matches_structured_oracle_fingerprint(node, captured_oracle_values):
            return "ground_truth_executed", "structured runtime oracle fingerprint"
        for key in captured_oracle_values:
            if _legacy_grounding_key_matches(node, snippet, key):
                return "ground_truth_executed", "value captured at runtime"
    if (
        existing_test_source
        and snippet
        and _matches_existing_by_ast_shape(node, existing_test_source)
    ):
        return "existing_test_copied", "shape match against project tests"
    if _is_loose_assertion(node, snippet):
        return "loose", "non-discriminating assertion"
    if focal_signature_summary and _looks_signature_derived(
        node=node,
        snippet=snippet,
        focal_signature_summary=focal_signature_summary,
    ):
        return "signature_derived", "shape/type derived from signature"
    return "llm_fabricated", "no execution-grounded evidence"


def _matches_structured_oracle_fingerprint(
    node: ast.AST,
    captured_oracle_values: dict[str, Any],
) -> bool:
    fingerprints = (
        captured_oracle_values.get("oracle_fingerprints")
        or captured_oracle_values.get("__apex_oracle_fingerprints__")
        or captured_oracle_values.get("structured_fingerprints")
        or []
    )
    if not isinstance(fingerprints, list):
        return False
    assertion = _assertion_fingerprint(node)
    if not assertion:
        return False
    for raw in fingerprints:
        fp = dict(raw or {}) if isinstance(raw, dict) else {}
        if not fp:
            continue
        if fp.get("assertion_op") and fp.get("assertion_op") != assertion.get("assertion_op"):
            continue
        if fp.get("rhs_literal_shape") != assertion.get("rhs_literal_shape"):
            continue
        if fp.get("rhs_literal_repr") != assertion.get("rhs_literal_repr"):
            continue
        return True
    return False


def _assertion_fingerprint(node: ast.AST) -> dict[str, str] | None:
    compare: ast.Compare | None = None
    if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
        compare = node.test
    elif isinstance(node, ast.Call):
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name in {"assertEqual", "assertEquals"} and len(node.args) >= 2:
            rhs = node.args[1]
            literal = _literal_fingerprint(rhs)
            if literal is None:
                return None
            return {"assertion_op": "==", **literal}
        if name in {"assertAlmostEqual"} and len(node.args) >= 2:
            rhs = node.args[1]
            literal = _literal_fingerprint(rhs)
            if literal is None:
                return None
            return {"assertion_op": "approx", **literal}
    if compare is None or not compare.ops or not compare.comparators:
        return None
    op = compare.ops[0]
    op_text = _compare_op_text(op)
    if op_text not in {"==", "is", "in", "raises", "approx"}:
        return None
    literal = _literal_fingerprint(compare.comparators[0])
    if literal is None:
        return None
    return {"assertion_op": op_text, **literal}


def _literal_fingerprint(node: ast.AST) -> dict[str, str] | None:
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None
    return {
        "rhs_literal_shape": _value_shape(value),
        "rhs_literal_repr": repr(value),
    }


def _value_shape(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return f"list:{len(value)}"
    if isinstance(value, tuple):
        return f"tuple:{len(value)}"
    if isinstance(value, set):
        return f"set:{len(value)}"
    if isinstance(value, dict):
        return f"dict:{len(value)}"
    return type(value).__name__


def _compare_op_text(op: ast.cmpop) -> str:
    if isinstance(op, ast.Eq):
        return "=="
    if isinstance(op, ast.Is):
        return "is"
    if isinstance(op, ast.In):
        return "in"
    return type(op).__name__


def _legacy_grounding_key_matches(node: ast.AST, snippet: str, key: Any) -> bool:
    """Compatibility path for old ``{repr_key: value}`` capture maps.

    Scalar substrings such as ``"1"`` or ``"True"`` are deliberately not
    trusted: they match too much unrelated assertion text. Non-scalar literal
    snippets remain accepted for older persisted diagnostics.
    """

    if not isinstance(key, str) or not key or key not in snippet:
        return False
    try:
        value = ast.literal_eval(key)
    except (ValueError, SyntaxError):
        return len(key.strip()) >= 4
    if isinstance(value, (int, float, bool, str)) or value is None:
        return False
    assertion = _assertion_fingerprint(node)
    if not assertion:
        return False
    return assertion.get("rhs_literal_repr") == repr(value)


def _matches_existing_by_ast_shape(node: ast.AST, existing_test_source: str) -> bool:
    """Return True iff ``node`` matches any assertion in ``existing_test_source``
    by normalized AST shape.

    Reuses the shape normalizer from ``apex.evaluation.test_dedup`` —
    variable names and literal values are normalized away, so two
    assertions that differ only in identifier choice or magic numbers
    hash to the same shape. Returns False on parse failure so the
    fallback labels still apply (false-negative bias).
    """

    try:
        from apex.evaluation.test_dedup import _normalize_test_node
    except Exception:  # pragma: no cover - defensive
        return False
    try:
        target_shape = _normalize_test_node(node)
    except Exception:
        return False
    if not target_shape:
        return False
    try:
        existing_tree = ast.parse(existing_test_source)
    except SyntaxError:
        return False
    for existing_node in ast.walk(existing_tree):
        if not isinstance(existing_node, (ast.Assert, ast.Call)):
            continue
        try:
            shape = _normalize_test_node(existing_node)
        except Exception:
            continue
        if shape and shape == target_shape:
            return True
    return False


def _is_loose_assertion(node: ast.AST, snippet: str) -> bool:
    """Detect assertions that don't actually constrain the output."""

    if isinstance(node, ast.Assert):
        test = node.test
        if isinstance(test, ast.Compare):
            for op in test.ops:
                if isinstance(op, (ast.IsNot,)):
                    # ``x is not None`` / ``x is not anything``
                    return True
            # ``len(x) >= 0`` is tautological
            if (
                isinstance(test.left, ast.Call)
                and getattr(test.left.func, "id", None) == "len"
                and any(
                    isinstance(op, (ast.GtE, ast.Gt))
                    and isinstance(c, ast.Constant)
                    and isinstance(c.value, int)
                    and c.value <= 0
                    for op, c in zip(test.ops, test.comparators)
                )
            ):
                return True
        # ``assert x``  with no comparator → loose
        if isinstance(test, (ast.Name, ast.Attribute, ast.Constant)):
            return True
    if isinstance(node, ast.Call):
        attr = node.func
        name = getattr(attr, "attr", None) or getattr(attr, "id", None)
        if name in _LOOSE_CALL_NAMES:
            return True
    return False


def _looks_signature_derived(
    *,
    node: ast.AST,
    snippet: str,
    focal_signature_summary: str,
) -> bool:
    """Tag an assertion as ``signature_derived`` only when the symbol it
    references actually appears in the focal signature summary.

    Without this check, any ``assert isinstance(x, FabricatedClass)``
    would slip through the anti-hack gate even though ``FabricatedClass``
    was invented by the model. We extract the type/attribute name from
    the AST and require it to be a substring of the signature summary
    (which the caller derives from the focal AST). The generic
    ``assert len(...) >= N`` form is accepted because it asserts a shape
    constraint that doesn't rely on a fabricated symbol.
    """

    if snippet.startswith("assert len("):
        return True
    names = list(_collect_symbol_names_from_call(node))
    if not names:
        return False
    return any(name and name in focal_signature_summary for name in names)


def _collect_symbol_names_from_call(node: ast.AST) -> Iterable[str]:
    """Yield type/attr identifiers referenced by isinstance/hasattr calls
    inside an assertion node. Used to verify the symbol exists in the
    focal signature, not just that the assertion shape matches."""

    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func_name = getattr(sub.func, "attr", None) or getattr(sub.func, "id", None)
        if func_name not in {"isinstance", "hasattr"}:
            continue
        for arg in sub.args[1:]:  # arg[0] is the value being checked
            if isinstance(arg, ast.Name):
                yield arg.id
            elif isinstance(arg, ast.Attribute):
                # e.g. ``mymod.MyClass`` → yield trailing attr
                yield arg.attr
            elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                # hasattr(x, "method_name")
                yield arg.value
            elif isinstance(arg, ast.Tuple):
                for elt in arg.elts:
                    if isinstance(elt, ast.Name):
                        yield elt.id
                    elif isinstance(elt, ast.Attribute):
                        yield elt.attr


def _compute_hack_score(assertions: list[AssertionProvenance]) -> float:
    if not assertions:
        return 1.0
    bad = sum(1 for a in assertions if a.label in {"llm_fabricated", "loose"})
    return round(bad / len(assertions), 3)


def _decide_rejection(
    *,
    assertions: list[AssertionProvenance],
    hack_score: float,
    hack_score_max: float,
    require_at_least_one_grounded: bool,
) -> tuple[bool, str]:
    if not assertions:
        return True, "no assertions detected"
    if hack_score > hack_score_max:
        return True, f"hack_score {hack_score} > {hack_score_max}"
    if require_at_least_one_grounded:
        grounded = {"ground_truth_executed", "existing_test_copied", "signature_derived"}
        if not any(a.label in grounded for a in assertions):
            return True, "no execution-grounded assertion present"
    return False, ""
