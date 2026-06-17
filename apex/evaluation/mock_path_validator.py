"""Static validator for ``mock.patch`` / ``monkeypatch.setattr`` target paths.

Many AttributeError-bucket failures in our v6 TestGenEvalLite run came
from agents writing patch targets the focal module doesn't expose
(``mock.patch('foo.bar.baz')`` where ``baz`` doesn't exist on
``foo.bar``). Pytest then reports an `AttributeError` raised by the
``patch.start()`` call rather than a real test failure, and the per-test
acceptance gate spends a pytest invocation discovering it.

This module catches the bad targets statically — without running pytest
— so the in-process gate can surface them in the same drop/repair cycle
as real test failures, but without paying the pytest cost for tests
guaranteed to error.

Design constraints:
* No focal-module imports unless explicitly enabled by the caller. Some
  focal modules have side effects on import; callers that know it's safe
  pass ``allow_import=True``.
* Bias toward false-negatives: when the target string can't be evaluated
  statically (computed at runtime, threaded through a fixture, etc.),
  we say "unknown" rather than "invalid". Better to ship a flaky test
  than to drop a working one.
* Pure stdlib — no extra deps.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# We treat these as the canonical patch-API surfaces. Anything matching
# these attribute chains (modulo aliasing) is checked.
_PATCH_FUNCTION_NAMES = frozenset(
    {
        "patch",
        "patch.object",
        "patch.dict",
        "patch.multiple",
    }
)
_MONKEYPATCH_METHOD_NAMES = frozenset({"setattr", "delattr"})


@dataclass(frozen=True)
class MockPathFinding:
    """One unresolvable patch target plus the test it lives in."""

    test_name: str  # enclosing test function name, or "<module>"
    target: str  # the literal path string from the call
    call_kind: str  # "patch" | "patch.object" | "monkeypatch.setattr" | etc.
    line: int
    reason: str  # human-readable diagnostic

    def render(self) -> str:
        return f"{self.test_name}:{self.line}  {self.call_kind}('{self.target}') — {self.reason}"


@dataclass
class MockPathValidationResult:
    """Aggregate validator output for a candidate test file."""

    findings: list[MockPathFinding] = field(default_factory=list)
    inspected_calls: int = 0
    skipped_dynamic: int = 0
    parse_error: Optional[str] = None

    @property
    def has_findings(self) -> bool:
        return bool(self.findings)

    def offending_test_names(self) -> set[str]:
        return {f.test_name for f in self.findings if f.test_name != "<module>"}

    def to_dict(self) -> dict:
        return {
            "findings": [
                {
                    "test_name": f.test_name,
                    "target": f.target,
                    "call_kind": f.call_kind,
                    "line": f.line,
                    "reason": f.reason,
                }
                for f in self.findings
            ],
            "inspected_calls": self.inspected_calls,
            "skipped_dynamic": self.skipped_dynamic,
            "parse_error": self.parse_error,
        }


def is_mock_path_validation_enabled() -> bool:
    """Honor ``APEX_MOCK_PATH_VALIDATOR_ENABLED`` env (default ON)."""

    raw = os.environ.get("APEX_MOCK_PATH_VALIDATOR_ENABLED")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def validate_mock_paths(
    source: str,
    *,
    allow_import: bool = False,
    extra_modules: tuple[str, ...] = (),
) -> MockPathValidationResult:
    """Statically scan *source* for mock.patch / monkeypatch.setattr targets.

    Args:
        source: the candidate test file source.
        allow_import: when True, the validator may actually import the
            focal module to walk attribute chains. Defaults to False so
            the validator never triggers focal side effects. Callers
            that have already vetted the focal module (or run under a
            sandbox) can opt in.
        extra_modules: additional module names that should be considered
            "import-safe" for the import-walk path (only relevant when
            ``allow_import=True``).

    Returns:
        ``MockPathValidationResult`` listing any patch targets we are
        confident don't resolve. Dynamic (non-string-literal) targets,
        or targets whose root module can't be located, are NOT flagged
        — the validator biases hard toward false-negatives.
    """

    if not source:
        return MockPathValidationResult()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return MockPathValidationResult(parse_error=f"SyntaxError: {exc}")

    aliases = _collect_patch_aliases(tree)
    findings: list[MockPathFinding] = []
    inspected = 0
    skipped_dynamic = 0

    for test_name, call_node in _iter_calls_in_tests(tree):
        kind = _classify_call(call_node, aliases)
        if not kind:
            continue
        target_arg = _first_target_arg(call_node, kind)
        if target_arg is None:
            # No first arg, or the API doesn't take a string target
            # (e.g. ``patch.dict({...})``). Skip silently.
            continue
        if not isinstance(target_arg, ast.Constant) or not isinstance(target_arg.value, str):
            skipped_dynamic += 1
            continue
        target = target_arg.value
        inspected += 1
        reason = _resolve_target(
            target,
            allow_import=allow_import,
            extra_modules=extra_modules,
        )
        if reason is None:
            continue
        findings.append(
            MockPathFinding(
                test_name=test_name,
                target=target,
                call_kind=kind,
                line=getattr(call_node, "lineno", 0),
                reason=reason,
            )
        )

    return MockPathValidationResult(
        findings=findings,
        inspected_calls=inspected,
        skipped_dynamic=skipped_dynamic,
    )


def _collect_patch_aliases(tree: ast.AST) -> dict[str, str]:
    """Map locally-bound names to their canonical patch-API path.

    Handles the common import shapes:

    * ``from unittest.mock import patch`` → ``{"patch": "patch"}``
    * ``from unittest import mock``      → ``{"mock": "mock"}``
    * ``import unittest.mock as um``     → ``{"um": "mock"}``
    * ``from mock import patch as p``    → ``{"p": "patch"}``

    The values are the canonical names ``patch``, ``mock``,
    ``monkeypatch`` so the call classifier can match without re-deriving
    the alias chain at every call site.
    """

    aliases: dict[str, str] = {
        "patch": "patch",  # default — many tests use ``with patch(...)``
        "monkeypatch": "monkeypatch",
        "mock": "mock",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in {"unittest.mock", "mock", "pytest_mock"}:
                for alias in node.names:
                    bound = alias.asname or alias.name
                    if alias.name == "patch":
                        aliases[bound] = "patch"
                    elif alias.name == "MagicMock":
                        aliases[bound] = "MagicMock"
            if module == "unittest":
                for alias in node.names:
                    if alias.name == "mock":
                        bound = alias.asname or "mock"
                        aliases[bound] = "mock"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"unittest.mock", "mock"}:
                    bound = alias.asname or alias.name.split(".")[0]
                    aliases[bound] = "mock"
    return aliases


def _iter_calls_in_tests(tree: ast.AST):
    """Yield ``(test_function_name, ast.Call)`` for every call in a test scope.

    Module-level calls (e.g. patches set up at import time) are reported
    with ``test_name="<module>"``. The walker is best-effort: we yield
    every call in source order and let the classifier filter.
    """

    def walk(node: ast.AST, current_test: str):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                next_test = child.name if child.name.startswith("test_") else current_test
                yield from walk(child, next_test)
            elif isinstance(child, ast.ClassDef):
                yield from walk(child, current_test)
            elif isinstance(child, ast.Call):
                yield current_test, child
                yield from walk(child, current_test)
            else:
                yield from walk(child, current_test)

    yield from walk(tree, "<module>")


def _classify_call(call: ast.Call, aliases: dict[str, str]) -> str:
    """Return the canonical call kind, or ``""`` if the call isn't a patch."""

    func = call.func
    # Bare name: ``patch(...)``
    if isinstance(func, ast.Name):
        canonical = aliases.get(func.id)
        if canonical == "patch":
            return "patch"
        return ""

    if isinstance(func, ast.Attribute):
        # ``mock.patch(...)`` / ``mock.patch.object(...)``
        chain = _attribute_chain(func)
        if not chain:
            return ""
        head = chain[0]
        head_canonical = aliases.get(head, head)

        if head_canonical == "monkeypatch" or head == "monkeypatch":
            if chain[-1] in _MONKEYPATCH_METHOD_NAMES:
                return f"monkeypatch.{chain[-1]}"
            return ""

        if head_canonical == "mock":
            tail = ".".join(chain[1:])
            if tail in _PATCH_FUNCTION_NAMES:
                return tail
            return ""

        # ``patch.object(...)`` etc — head is the patch alias itself
        if head_canonical == "patch":
            tail = ".".join(chain)
            # tail is e.g. "patch.object" — strip the leading alias word
            if "." in tail:
                rest = tail.split(".", 1)[1]
                full = f"patch.{rest}"
                if full in _PATCH_FUNCTION_NAMES:
                    return full
            return ""

    return ""


def _attribute_chain(node: ast.AST) -> list[str]:
    """For ``a.b.c`` return ``["a", "b", "c"]``; ``[]`` if not pure attr access."""

    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        parts.reverse()
        return parts
    return []


def _first_target_arg(call: ast.Call, kind: str) -> Optional[ast.AST]:
    """Return the AST node of the first arg that should be a target path string.

    For ``patch("a.b.c")`` and ``patch.dict("a.b.c", ...)`` and
    ``patch.multiple("a.b.c", ...)`` it's positional arg 0.
    For ``patch.object(target_obj, "attr")`` the first arg is an object
    (not a string), so the validator can't help — return None.
    For ``monkeypatch.setattr("a.b.c", value)`` it's positional arg 0.
    For ``monkeypatch.setattr(obj, "attr", value)`` it's not a path —
    return None.
    """

    if not call.args:
        return None
    first = call.args[0]
    if kind == "patch.object":
        return None  # first arg is an object, not a string path
    if kind == "patch.multiple":
        return first if isinstance(first, ast.Constant) else None
    if kind in {"monkeypatch.setattr", "monkeypatch.delattr"}:
        # Distinguish setattr("a.b", v) from setattr(obj, "attr", v).
        # Heuristic: if there are 3+ positional args, the first is the
        # object, not a path string.
        if len(call.args) >= 3:
            return None
        return first
    return first


def _resolve_target(
    target: str,
    *,
    allow_import: bool,
    extra_modules: tuple[str, ...],
) -> Optional[str]:
    """Try to resolve *target* (``"a.b.c"``) to a real attribute.

    Returns ``None`` when the target resolves OR when we cannot make a
    confident judgment (no module found, can't import safely). Returns a
    diagnostic string only when the root module exists AND the attribute
    chain is provably broken.
    """

    if not target or "." not in target:
        # Single-token targets (``patch("X")``) often refer to local
        # objects re-bound at runtime — we cannot resolve them statically.
        return None

    parts = target.split(".")
    # Find the longest prefix that is a real importable module.
    module_name = ""
    attr_chain: list[str] = []
    for split in range(len(parts) - 1, 0, -1):
        candidate_module = ".".join(parts[:split])
        if _module_exists(candidate_module):
            module_name = candidate_module
            attr_chain = parts[split:]
            break

    if not module_name:
        # Couldn't find the module portion. Bias toward false-negative
        # — many local symbols, fixtures, and patched-in test helpers
        # won't resolve here even when they're correct.
        return None

    if not attr_chain:
        # The whole target IS a module — that's valid (``patch.dict('os.environ',...)``
        # for example, where the target IS a real attribute on os).
        return None

    if not allow_import and module_name not in extra_modules:
        # Static-only mode: we know the module exists (find_spec said so)
        # but we won't actually import it to walk the attribute chain.
        # Return None — we can't make a strong claim either way.
        return None

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - depends on focal env
        logger.debug("mock_path_validator: import %s failed: %s", module_name, exc)
        return None

    obj: object = module
    for idx, attr in enumerate(attr_chain):
        if not hasattr(obj, attr):
            chain_so_far = ".".join([module_name, *attr_chain[:idx]])
            return (
                f"attribute '{attr}' not found on {chain_so_far!r} "
                f"(module {module_name} resolved, chain broke at position {idx + 1})"
            )
        obj = getattr(obj, attr)
    return None


def _module_exists(name: str) -> bool:
    """Return True iff ``importlib.util.find_spec(name)`` succeeds.

    ``find_spec`` does NOT execute the module — it only resolves the
    finder chain — so this is safe even on focal modules with import
    side effects.
    """

    if not name:
        return False
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, ModuleNotFoundError):
        return False
    except Exception:  # pragma: no cover - defensive
        return False
    return spec is not None
