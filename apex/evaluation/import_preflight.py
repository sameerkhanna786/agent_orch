"""Fast import preflight for generated Python tests."""

from __future__ import annotations

import ast
import builtins
import importlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImportPreflightResult:
    status: str
    diagnostic: str = ""
    imports: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UndefinedNamesResult:
    """Result of the W5 undefined-name preflight pass."""

    status: str
    undefined_names: list[str] = field(default_factory=list)
    diagnostic: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Names that are always available in pytest collected modules without an
# explicit import. ``pytest`` itself is NOT included — if a test references
# ``pytest`` symbols (``pytest.fixture``, ``pytest.raises``) it must import
# the package; conftest fixtures/markers come through function arguments.
_IMPLICIT_TEST_NAMES: frozenset[str] = frozenset(
    {
        "__name__",
        "__file__",
        "__doc__",
        "__builtins__",
        "__package__",
        "__loader__",
        "__spec__",
        "__path__",
        "__class__",
        "__future__",
        "self",  # method bodies bind self via the call site, not a Load
        "cls",  # ditto for classmethods
    }
)


def detect_undefined_names(
    source: str,
    *,
    focal_module_exports: Iterable[str] | None = None,
) -> UndefinedNamesResult:
    """Find ``Name`` references in ``source`` that aren't defined anywhere.

    The W5 plan: parse the test file, collect every ``Name`` load and the
    base of every ``Attribute`` chain, and flag ones that don't resolve to:
      * a Python builtin (``builtins.__dir__()``)
      * a name imported in the same file (``Import`` / ``ImportFrom``)
      * a name defined in the same file (``FunctionDef``, ``ClassDef``,
        ``Assign`` / ``AnnAssign`` targets)
      * a name listed in ``focal_module_exports`` (``__all__`` of the module
        under test, when known to the caller)
      * a function-local binding (parameters, ``with ... as``, ``for ...``
        targets, walrus assignments, comprehension iteration variables)

    The pass is intentionally conservative: when in doubt (dynamic
    attribute access, ``getattr`` chains, exotic assignment shapes), we
    let the reference through. False positives over-reject good
    candidates, which is far worse than missing some bad ones.
    """

    text = source if isinstance(source, str) else str(source or "")
    if not text.strip():
        return UndefinedNamesResult(status="pass")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        # If the source doesn't parse the W3 syntax gate handles it; we
        # cannot meaningfully analyze references in unparseable text.
        return UndefinedNamesResult(
            status="skipped",
            diagnostic=f"ast.parse failed: {exc.msg or exc}",
        )

    builtin_names = set(dir(builtins))
    defined: set[str] = set(builtin_names)
    defined.update(_IMPLICIT_TEST_NAMES)
    if focal_module_exports:
        defined.update(str(name) for name in focal_module_exports)

    # Pass 1: collect all module-level + nested-scope definitions in one
    # sweep. We don't track scopes precisely; conservatism wins. A name
    # defined in any function counts as "defined somewhere in the file".
    for node in ast.walk(tree):
        _collect_definitions(node, defined)

    # Pass 2: walk again and flag bare ``Name`` loads that aren't defined.
    undefined: list[str] = []
    seen_undefined: set[str] = set()
    for ref in _iter_name_loads(tree):
        if ref in defined or ref in seen_undefined:
            continue
        # Conservative: if the source contains the name in a getattr / setattr
        # / hasattr string literal we treat it as defined (dynamic access).
        if _name_appears_dynamically(ref, text):
            continue
        undefined.append(ref)
        seen_undefined.add(ref)

    if undefined:
        return UndefinedNamesResult(
            status="fail",
            undefined_names=undefined,
            diagnostic=f"undefined name references: {', '.join(undefined[:8])}",
        )
    return UndefinedNamesResult(status="pass")


def _collect_definitions(node: ast.AST, defined: set[str]) -> None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            if name:
                defined.add(name)
    elif isinstance(node, ast.ImportFrom):
        for alias in node.names:
            if alias.name == "*":
                # ``from x import *`` makes everything visible; mark the
                # module as conservative-passthrough by sentinel-ing nothing.
                # We can't enumerate the wildcard, so we add no name; the
                # ``getattr``-style fallback below catches typical false
                # positives and the import_preflight subprocess catches
                # broken imports separately.
                continue
            name = alias.asname or alias.name
            if name:
                defined.add(name)
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        defined.add(node.name)
        # Function parameters are defined inside their body. Since we use a
        # single global name set, add them so references inside don't fire.
        args = node.args
        for arg_list in (
            args.posonlyargs or [],
            args.args or [],
            args.kwonlyargs or [],
        ):
            for arg in arg_list:
                if arg.arg:
                    defined.add(arg.arg)
        if args.vararg and args.vararg.arg:
            defined.add(args.vararg.arg)
        if args.kwarg and args.kwarg.arg:
            defined.add(args.kwarg.arg)
    elif isinstance(node, ast.ClassDef):
        defined.add(node.name)
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            _collect_assign_targets(target, defined)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        _collect_assign_targets(node.target, defined)
    elif isinstance(node, ast.NamedExpr):  # walrus
        _collect_assign_targets(node.target, defined)
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        _collect_assign_targets(node.target, defined)
    elif isinstance(node, ast.With):
        for item in node.items:
            if item.optional_vars is not None:
                _collect_assign_targets(item.optional_vars, defined)
    elif isinstance(node, ast.AsyncWith):
        for item in node.items:
            if item.optional_vars is not None:
                _collect_assign_targets(item.optional_vars, defined)
    elif isinstance(node, ast.ExceptHandler):
        if node.name:
            defined.add(node.name)
    elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        for generator in node.generators:
            _collect_assign_targets(generator.target, defined)
    elif isinstance(node, ast.Lambda):
        for arg_list in (
            node.args.posonlyargs or [],
            node.args.args or [],
            node.args.kwonlyargs or [],
        ):
            for arg in arg_list:
                if arg.arg:
                    defined.add(arg.arg)
        if node.args.vararg and node.args.vararg.arg:
            defined.add(node.args.vararg.arg)
        if node.args.kwarg and node.args.kwarg.arg:
            defined.add(node.args.kwarg.arg)
    elif isinstance(node, ast.Global):
        for name in node.names:
            defined.add(name)
    elif isinstance(node, ast.Nonlocal):
        for name in node.names:
            defined.add(name)


def _collect_assign_targets(target: ast.AST, defined: set[str]) -> None:
    if isinstance(target, ast.Name):
        defined.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            _collect_assign_targets(elt, defined)
    elif isinstance(target, ast.Starred):
        _collect_assign_targets(target.value, defined)
    # Subscript / Attribute assignments don't introduce new bare names.


def _iter_name_loads(tree: ast.AST) -> Iterable[str]:
    seen: set[str] = set()
    for node in ast.walk(tree):
        # Bare Name reference in Load context.
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in seen:
                seen.add(node.id)
                yield node.id
        # Base of an Attribute chain (e.g. ``foo.bar.baz`` -> emit ``foo``).
        elif isinstance(node, ast.Attribute):
            base = _attribute_base(node)
            if base is not None and base not in seen:
                seen.add(base)
                yield base


def _attribute_base(node: ast.Attribute) -> str | None:
    current: ast.AST = node.value
    while isinstance(current, ast.Attribute):
        current = current.value
    if isinstance(current, ast.Name):
        return current.id
    return None


def _name_appears_dynamically(name: str, source: str) -> bool:
    """Conservative escape hatch: treat names referenced via dynamic access
    as defined.

    The plan calls out ``getattr(obj, "name")`` style references — if the
    name appears as a string literal anywhere in the source we cannot
    safely flag the bare reference as undefined.
    """

    if not name:
        return False
    quoted = (f'"{name}"', f"'{name}'")
    return any(token in source for token in quoted)


def preflight_imports(
    source: str,
    *,
    workdir: Path,
    env: dict[str, str] | None = None,
    timeout: float = 5.0,
    python_executable: str | None = None,
) -> ImportPreflightResult:
    imports = extract_import_specs(source)
    if not imports:
        return ImportPreflightResult(status="pass", imports=[])
    executable = python_executable or sys.executable
    with tempfile.TemporaryDirectory(prefix="apex_import_preflight_") as tmp:
        spec_path = Path(tmp) / "imports.json"
        driver_path = Path(tmp) / "driver.py"
        spec_path.write_text(json.dumps(imports), encoding="utf-8")
        driver_path.write_text(_IMPORT_DRIVER, encoding="utf-8")
        run_env = dict(os.environ)
        if env:
            run_env.update({str(k): str(v) for k, v in env.items()})
        try:
            completed = subprocess.run(
                [executable, str(driver_path), str(spec_path)],
                cwd=str(workdir),
                env=run_env,
                capture_output=True,
                text=True,
                timeout=float(timeout),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ImportPreflightResult(
                status="fail",
                diagnostic="import preflight timed out",
                imports=imports,
            )
        except OSError as exc:
            return ImportPreflightResult(
                status="fail",
                diagnostic=f"{type(exc).__name__}: {exc}",
                imports=imports,
            )
    if completed.returncode == 0:
        return ImportPreflightResult(status="pass", imports=imports)
    return ImportPreflightResult(
        status="fail",
        diagnostic=(completed.stderr or completed.stdout or "import preflight failed")[-4000:],
        imports=imports,
    )


def preflight_import_policy(source: str, *, style: Any | None = None) -> ImportPreflightResult:
    """Check generated imports against the selected runner profile.

    This is deliberately separate from importability probing: a host venv may
    have pytest installed, but a unittest/Django/SymPy runner profile can still
    reject bare pytest helpers because the target environment owns the policy.
    """

    imports = extract_import_specs(source)
    if style is None:
        return ImportPreflightResult(status="pass", imports=imports)
    try:
        from .test_style import imports_forbidden_by_style

        forbidden = imports_forbidden_by_style(source, style)
    except Exception:
        forbidden = []
    if forbidden:
        return ImportPreflightResult(
            status="fail",
            diagnostic="runner-disallowed imports: " + ", ".join(forbidden),
            imports=imports,
        )
    return ImportPreflightResult(status="pass", imports=imports)


def extract_import_specs(source: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(source or "")
    except SyntaxError:
        return []
    imports: list[dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({"type": "import", "module": alias.name})
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(
                {
                    "type": "from",
                    "module": node.module,
                    "names": [alias.name for alias in node.names if alias.name != "*"],
                }
            )
    return imports


_IMPORT_DRIVER = r"""
import importlib
import json
import sys

imports = json.loads(open(sys.argv[1], encoding="utf-8").read())
for item in imports:
    if item["type"] == "import":
        importlib.import_module(item["module"])
    elif item["type"] == "from":
        module = importlib.import_module(item["module"])
        for name in item.get("names") or []:
            getattr(module, name)
"""


# ---------------------------------------------------------------------------
# P2 step 8: extend symbol preflight to attribute chains.
#
# ``detect_undefined_names`` catches undefined ROOT names (``foo`` in
# ``foo.bar()``). It does NOT catch the case where ``foo`` is correctly
# imported but ``foo.bar`` doesn't exist on the resolved module — that
# fails at runtime as ``AttributeError`` and contributed to ~16 v6 task
# failures we'd like to surface earlier.
#
# The new helpers below walk attribute chains anchored at imports we
# resolved at preflight time, follow the chain through the imported
# module's symbol table, and flag dangling tails. Bias is hard toward
# false-negatives — see the per-helper notes for what we deliberately
# refuse to flag.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UndefinedAttributeChain:
    """One unresolvable attribute chain plus where it lives."""

    chain: str  # e.g. "foo.bar.baz" — verbatim source rendering of the chain
    root: str  # e.g. "foo" — the bound name at the chain root
    missing_attr: str  # the attribute that broke the chain
    resolved_module: str  # e.g. "foo" or "foo.bar" — last good module name
    line: int
    enclosing_function: str  # test_* function or "<module>"
    reason: str

    def render(self) -> str:
        return (
            f"{self.enclosing_function}:{self.line}  {self.chain} "
            f"— attribute '{self.missing_attr}' missing on {self.resolved_module!r}"
        )


@dataclass
class UndefinedAttributesResult:
    """Aggregate output for the chain-walk detector."""

    findings: list[UndefinedAttributeChain] = field(default_factory=list)
    inspected_chains: int = 0
    skipped_dynamic: int = 0
    parse_error: Optional[str] = None

    @property
    def has_findings(self) -> bool:
        return bool(self.findings)

    def offending_test_names(self) -> set[str]:
        return {f.enclosing_function for f in self.findings if f.enclosing_function != "<module>"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [
                {
                    "chain": f.chain,
                    "root": f.root,
                    "missing_attr": f.missing_attr,
                    "resolved_module": f.resolved_module,
                    "line": f.line,
                    "enclosing_function": f.enclosing_function,
                    "reason": f.reason,
                }
                for f in self.findings
            ],
            "inspected_chains": self.inspected_chains,
            "skipped_dynamic": self.skipped_dynamic,
            "parse_error": self.parse_error,
        }


def is_attribute_chain_check_enabled() -> bool:
    """Honor ``APEX_ATTRIBUTE_CHAIN_CHECK_ENABLED`` env (default ON)."""

    raw = os.environ.get("APEX_ATTRIBUTE_CHAIN_CHECK_ENABLED")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def detect_undefined_attribute_chains(
    source: str,
    *,
    allow_import: bool = False,
    extra_modules: tuple[str, ...] = (),
) -> UndefinedAttributesResult:
    """Find attribute chains in *source* that don't resolve.

    Chains anchored at ``Import`` / ``ImportFrom`` bound names are
    walked through the import graph. Anything else (locals, fixtures,
    parameters, walrus targets) is left alone — there's no general
    static way to know what runtime type those receive.

    Args:
        source: candidate test file source.
        allow_import: when True, the detector imports the chain's root
            module to walk past ``find_spec``. When False (default), we
            confirm the module exists but never walk attributes — so
            unresolvable tails go silently. Callers that have already
            vetted the focal sandbox can opt in.
        extra_modules: when ``allow_import=False``, modules listed here
            are imported anyway. Useful for the focal module path that
            the in-process gate has already PYTHONPATH-mounted.
    """

    if not source:
        return UndefinedAttributesResult()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return UndefinedAttributesResult(parse_error=f"SyntaxError: {exc}")

    import_targets = _collect_import_targets(tree)
    if not import_targets:
        return UndefinedAttributesResult()

    findings: list[UndefinedAttributeChain] = []
    inspected = 0
    skipped_dynamic = 0
    seen_chains: set[str] = set()
    dynamic_names = _collect_dynamic_attribute_names(source)

    for enclosing, attr_node in _iter_attribute_nodes_with_enclosing(tree):
        chain_parts = _full_attribute_chain(attr_node)
        if not chain_parts or len(chain_parts) < 2:
            # Not a chain rooted at a Name (e.g. ``func().attr``)
            continue
        root, *attrs = chain_parts
        if root not in import_targets:
            continue
        # Some chain element appears as a string literal (``getattr(obj, 'name')``)
        # — bias to false-negative.
        if any(attr in dynamic_names for attr in attrs):
            skipped_dynamic += 1
            continue
        chain_text = ".".join(chain_parts)
        if chain_text in seen_chains:
            continue
        seen_chains.add(chain_text)
        inspected += 1
        target = import_targets[root]
        finding = _resolve_chain(
            chain_parts=chain_parts,
            target=target,
            allow_import=allow_import,
            extra_modules=extra_modules,
            line=getattr(attr_node, "lineno", 0),
            enclosing_function=enclosing,
        )
        if finding is not None:
            findings.append(finding)

    return UndefinedAttributesResult(
        findings=findings,
        inspected_chains=inspected,
        skipped_dynamic=skipped_dynamic,
    )


@dataclass(frozen=True)
class _ImportTarget:
    """How a bound name was introduced.

    * ``module`` form: ``import foo``      ⇒ resolved=``"foo"``,    rest=()
    * ``module`` form: ``import foo.bar``  ⇒ resolved=``"foo.bar"`` (alias's),
      but ``import foo.bar`` actually binds ``foo``; we reflect that with
      ``resolved`` = the alias's bound name target (``foo``) and rest=()
    * ``from`` form:   ``from pkg import x`` ⇒ resolved=``"pkg"``, rest=("x",)
    """

    bound_name: str
    resolved_module: str
    rest: tuple[str, ...]


def _collect_import_targets(tree: ast.AST) -> dict[str, _ImportTarget]:
    """Map each bound name to the module/attribute it points to.

    Only top-level imports (``Import`` / ``ImportFrom``) are honored.
    Conditional imports (``try/except ImportError``) are still recorded
    — they bind the name in the success case, and the in-process gate
    already verifies the imports separately.
    """

    targets: dict[str, _ImportTarget] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    targets[alias.asname] = _ImportTarget(
                        bound_name=alias.asname,
                        resolved_module=alias.name,
                        rest=(),
                    )
                else:
                    head = alias.name.split(".", 1)[0]
                    targets[head] = _ImportTarget(
                        bound_name=head,
                        resolved_module=head,
                        rest=(),
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if not module:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound = alias.asname or alias.name
                targets[bound] = _ImportTarget(
                    bound_name=bound,
                    resolved_module=module,
                    rest=(alias.name,),
                )
    return targets


def _iter_attribute_nodes_with_enclosing(tree: ast.AST):
    """Yield ``(enclosing_test_function_name, outermost_attribute_node)``
    for every top-level Attribute load.

    "Outermost" means we yield the longest chain only — given
    ``a.b.c``, we yield the ``Attribute(attr='c', value=Attribute(...))``
    once. The walk explicitly does NOT descend into the Attribute's
    own ``.value`` chain to avoid emitting redundant sub-chains.
    Inside a non-test enclosing scope we propagate the most recent
    test function name; module-level chains use ``"<module>"``.
    """

    def walk(node: ast.AST, current: str):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                next_enclosing = child.name if child.name.startswith("test_") else current
                yield from walk(child, next_enclosing)
            elif isinstance(child, ast.ClassDef):
                yield from walk(child, current)
            elif isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Load):
                yield current, child
                # Do NOT recurse into ``child`` itself — its inner
                # Attribute(s) are sub-chains of what we just yielded.
                # We DO recurse into Call/Subscript/etc. nested children
                # of the chain so calls inside ``a.b(c)`` still get
                # walked at ``c``, but we skip the inner Attribute spine.
                for sub_field, sub_value in ast.iter_fields(child):
                    if sub_field == "value":
                        # Skip the inner Attribute spine — walking into
                        # it would re-yield ``a.b`` after ``a.b.c``.
                        if isinstance(sub_value, ast.Attribute):
                            continue
                        if isinstance(sub_value, ast.AST):
                            yield from walk(sub_value, current)
                        continue
                    if isinstance(sub_value, ast.AST):
                        yield from walk(sub_value, current)
                    elif isinstance(sub_value, list):
                        for item in sub_value:
                            if isinstance(item, ast.AST):
                                yield from walk(item, current)
            else:
                yield from walk(child, current)

    yield from walk(tree, "<module>")


def _full_attribute_chain(node: ast.Attribute) -> list[str]:
    """Resolve ``a.b.c`` to ``["a", "b", "c"]``; ``[]`` for non-Name root."""

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


def _collect_dynamic_attribute_names(source: str) -> set[str]:
    """Collect attribute names that appear as string literals anywhere.

    Mirrors ``_name_appears_dynamically`` but for individual attribute
    tokens. ``getattr(foo, "bar")`` ⇒ ``"bar"`` becomes a dynamic name
    we won't flag in chain walks. False-negative bias: better to miss a
    bad chain than to drop a correct test.
    """

    if not source:
        return set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            # Common case: a single identifier-shaped string.
            if value.isidentifier():
                out.add(value)
    return out


def _resolve_chain(
    *,
    chain_parts: list[str],
    target: _ImportTarget,
    allow_import: bool,
    extra_modules: tuple[str, ...],
    line: int,
    enclosing_function: str,
) -> Optional[UndefinedAttributeChain]:
    """Walk *chain_parts* against *target* and return a finding if it breaks.

    *chain_parts*[0] is the bound name (the root); the remaining parts
    are the attribute tokens to resolve.
    """

    chain_text = ".".join(chain_parts)
    root = chain_parts[0]
    attrs = chain_parts[1:]
    full_attrs = list(target.rest) + list(attrs)
    if not full_attrs:
        return None

    if not _module_exists(target.resolved_module):
        # Conservative: root module not on sys.path ⇒ no finding.
        return None

    if not allow_import and target.resolved_module not in extra_modules:
        # Static-only: we know the root module is importable but won't
        # actually import it to walk the attribute chain.
        return None

    try:
        module = importlib.import_module(target.resolved_module)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(
            "import_preflight: import %s failed: %s",
            target.resolved_module,
            exc,
        )
        return None

    obj: object = module
    resolved_path = target.resolved_module
    for idx, attr in enumerate(full_attrs):
        if not hasattr(obj, attr):
            return UndefinedAttributeChain(
                chain=chain_text,
                root=root,
                missing_attr=attr,
                resolved_module=resolved_path,
                line=line,
                enclosing_function=enclosing_function,
                reason=(
                    f"attribute '{attr}' not found on {resolved_path!r} "
                    f"(chain broke at position {idx + 1} of {len(full_attrs)})"
                ),
            )
        obj = getattr(obj, attr)
        resolved_path = f"{resolved_path}.{attr}"
    return None


def _module_exists(name: str) -> bool:
    """Return True iff ``importlib.util.find_spec(name)`` succeeds.

    ``find_spec`` does NOT execute the module — same trick as the
    mock-path validator.
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
