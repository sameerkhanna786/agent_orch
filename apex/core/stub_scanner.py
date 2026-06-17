"""Cross-language stub-residue scanner.

Walks the changed files of a candidate patch and flags public functions
whose bodies are still placeholder stubs (``pass`` / ``return None`` /
``raise NotImplementedError`` / ``unimplemented!()`` / ...).

Python uses real AST for precision (filters dunder + leading-underscore
private names; ignores trivial properties). Other languages use the
``stub_patterns()`` regexes the test-runner adapter exposes — these are
deliberately conservative substring matches so the scanner stays simple
and dependency-free (no tree-sitter requirement).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class StubFinding:
    path: str
    symbol: str
    reason: str


_PYTHON_STUB_BODY_TYPES = {ast.Pass, ast.Constant}  # `pass` and `...`


def _name_from_expr(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _name_from_expr(node.value)
    if isinstance(node, ast.Call):
        return _name_from_expr(node.func)
    return None


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for dec in node.decorator_list:
        name = _name_from_expr(dec)
        if name:
            names.add(name)
    return names


def _base_names(node: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for base in node.bases:
        name = _name_from_expr(base)
        if name:
            names.add(name)
    return names


def _direct_public_methods(node: ast.ClassDef) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        child
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not child.name.startswith("_")
    ]


def _protocol_classes(class_nodes: list[ast.ClassDef]) -> set[str]:
    protocol_classes: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in class_nodes:
            bases = _base_names(node)
            if "Protocol" not in bases and not (bases & protocol_classes):
                continue
            if node.name not in protocol_classes:
                protocol_classes.add(node.name)
                changed = True
    return protocol_classes


def _subclass_overrides(class_nodes: list[ast.ClassDef]) -> set[tuple[str, str]]:
    overrides: set[tuple[str, str]] = set()
    for node in class_nodes:
        method_names = {method.name for method in _direct_public_methods(node)}
        if not method_names:
            continue
        for base in _base_names(node):
            for method_name in method_names:
                overrides.add((base, method_name))
    return overrides


def _is_test_file(rel: str, path: Path) -> bool:
    rel_lower = rel.lower()
    return (
        "/tests/" in rel_lower
        or rel_lower.startswith("tests/")
        or rel_lower.startswith("test/")
        or path.name.startswith("test_")
        or path.name.endswith("_test.py")
        or path.name.endswith(".test.js")
        or path.name.endswith(".test.ts")
        or path.name.endswith(".spec.js")
        or path.name.endswith(".spec.ts")
        or path.name.endswith("_test.go")
    )


def _collect_python_interface_context(
    workspace: Path,
    changed_files: list[str],
) -> tuple[set[str], set[tuple[str, str]]]:
    class_nodes: list[ast.ClassDef] = []
    context_paths: dict[str, Path] = {}
    for rel in changed_files:
        if not rel:
            continue
        path = workspace / rel
        if path.suffix.lower() != ".py" or _is_test_file(rel, path):
            continue
        context_paths[str(path)] = path
        try:
            siblings = list(path.parent.iterdir())
        except OSError:
            siblings = []
        for sibling in siblings:
            if sibling.suffix.lower() != ".py" or not sibling.is_file():
                continue
            try:
                sibling_rel = str(sibling.relative_to(workspace))
            except ValueError:
                sibling_rel = sibling.name
            if _is_test_file(sibling_rel, sibling):
                continue
            context_paths[str(sibling)] = sibling
    for path in context_paths.values():
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError):
            continue
        class_nodes.extend(
            node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        )
    return _protocol_classes(class_nodes), _subclass_overrides(class_nodes)


def _python_function_is_stub(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Optional[str]:
    body = list(node.body)
    # Strip a leading docstring if present.
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body:
        return "empty body"
    if len(body) > 1:
        return None
    only = body[0]
    if isinstance(only, ast.Pass):
        return "body is `pass`"
    if (
        isinstance(only, ast.Expr)
        and isinstance(only.value, ast.Constant)
        and only.value.value is Ellipsis
    ):
        return "body is `...`"
    if isinstance(only, ast.Return):
        if only.value is None:
            return "body is `return` with no value"
        if isinstance(only.value, ast.Constant) and only.value.value is None:
            return "body is `return None`"
    if isinstance(only, ast.Raise) and isinstance(only.exc, ast.Call):
        func = only.exc.func
        name = ""
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name == "NotImplementedError":
            return "body raises NotImplementedError"
    if (
        isinstance(only, ast.Raise)
        and isinstance(only.exc, ast.Name)
        and only.exc.id == "NotImplementedError"
    ):
        return "body raises NotImplementedError"
    return None


def _python_function_is_documented_noop(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    doc = ast.get_docstring(node) or ""
    if _python_function_is_stub(node) not in {
        "body is `pass`",
        "body raises NotImplementedError",
    }:
        return False
    return any(
        marker in doc.strip().lower()
        for marker in (
            "do nothing",
            "no-op",
            "noop",
            "intentionally empty",
            "intentionally does nothing",
        )
    )


def _class_declares_partial_implementation(node: ast.ClassDef) -> bool:
    doc = (ast.get_docstring(node) or "").strip().lower()
    return any(
        marker in doc
        for marker in (
            "not a complete implementation",
            "partial implementation",
            "only use",
            "only used",
            "only implemented",
            "not supported",
        )
    )


def _scan_python(
    path: Path,
    *,
    protocol_classes: Optional[set[str]] = None,
    subclass_overrides: Optional[set[tuple[str, str]]] = None,
) -> list[StubFinding]:
    findings: list[StubFinding] = []
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError):
        return findings

    class_nodes = [
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    ]
    protocol_classes = protocol_classes or _protocol_classes(class_nodes)
    subclass_overrides = subclass_overrides or _subclass_overrides(class_nodes)
    class_base_names = {node.name: _base_names(node) for node in class_nodes}
    partial_implementation_classes = {
        node.name for node in class_nodes if _class_declares_partial_implementation(node)
    }

    def should_skip_function(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        parent_class: Optional[str] = None,
    ) -> bool:
        # Skip private and dunder by convention.
        if node.name.startswith("_"):
            return True
        decorators = _decorator_names(node)
        # Skip deliberate abstract methods.
        if "abstractmethod" in decorators or "abstractproperty" in decorators:
            return True
        # Skip @overload signatures (the typing convention is to leave them
        # as `...` bodies; the implementation lives elsewhere).
        if "overload" in decorators:
            return True
        if parent_class is None:
            return False
        # Interface declarations can legitimately use pass/NotImplemented;
        # the scanner is meant to flag implementation residue.
        if parent_class in protocol_classes:
            return True
        if (parent_class, node.name) in subclass_overrides:
            return True
        if class_base_names.get(parent_class):
            if _python_function_is_documented_noop(node):
                return True
            if (
                parent_class in partial_implementation_classes
                and _python_function_is_stub(node) == "body raises NotImplementedError"
            ):
                return True
        return False

    def add_finding(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        parent_class: Optional[str] = None,
    ) -> None:
        if should_skip_function(node, parent_class=parent_class):
            return
        reason = _python_function_is_stub(node)
        if reason:
            symbol = node.name if parent_class is None else f"{parent_class}.{node.name}"
            findings.append(StubFinding(path=str(path), symbol=symbol, reason=reason))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_finding(node)
        elif isinstance(node, ast.ClassDef):
            for child in _direct_public_methods(node):
                add_finding(child, parent_class=node.name)
    return findings


def _scan_with_tree_sitter(path: Path, suffix: str) -> Optional[list[StubFinding]]:
    """Tree-sitter based scan; returns ``None`` when unavailable."""
    try:
        from . import tree_sitter_helper
    except ImportError:
        return None
    if not tree_sitter_helper.is_available():
        return None
    language = tree_sitter_helper.language_for_suffix(suffix)
    if language is None:
        return None
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    locations = tree_sitter_helper.stub_function_findings(source, language)
    return [
        StubFinding(path=str(path), symbol=f"{loc.name}@line:{loc.line}", reason=loc.reason)
        for loc in locations
    ]


def _scan_with_patterns(path: Path, patterns: list[str]) -> list[StubFinding]:
    findings: list[StubFinding] = []
    if not patterns:
        return findings
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return findings
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error:
            continue
    for line_no, line in enumerate(source.splitlines(), start=1):
        for cre in compiled:
            if cre.search(line):
                findings.append(
                    StubFinding(
                        path=str(path),
                        symbol=f"line:{line_no}",
                        reason=f"matches stub pattern /{cre.pattern}/",
                    )
                )
                break
    return findings


_LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "javascript",
    ".tsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "java",
    ".kts": "java",
    ".scala": "java",
    ".rb": "ruby",
}


def scan_files_for_stubs(
    workspace: Path,
    changed_files: Iterable[str],
    *,
    adapter_stub_patterns: Optional[list[str]] = None,
    max_findings: int = 25,
) -> list[StubFinding]:
    """Return stub findings for the changed files.

    Python files are scanned via AST regardless of which adapter is
    active (Python AST is in the stdlib and gives the most precise
    signal). For other languages, the adapter's regex patterns are used
    as a heuristic — sufficient to flag the obvious cases without taking
    a tree-sitter dependency.
    """
    findings: list[StubFinding] = []
    workspace = Path(workspace)
    changed_file_list = list(changed_files)
    protocol_classes, subclass_overrides = _collect_python_interface_context(
        workspace,
        changed_file_list,
    )
    for rel in changed_file_list:
        if not rel:
            continue
        path = workspace / rel
        if not path.exists() or not path.is_file():
            continue
        suffix = path.suffix.lower()
        # Skip test files — stubs in tests are intentional sometimes.
        if _is_test_file(rel, path):
            continue
        if suffix == ".py":
            findings.extend(
                _scan_python(
                    path,
                    protocol_classes=protocol_classes,
                    subclass_overrides=subclass_overrides,
                )
            )
        else:
            language = _LANGUAGE_BY_SUFFIX.get(suffix)
            if language is None:
                continue
            # Prefer tree-sitter (function-scoped, far fewer false
            # positives) when available; fall back to whole-file regex
            # when tree-sitter isn't installed for this language.
            ts_results = _scan_with_tree_sitter(path, suffix)
            if ts_results is not None:
                findings.extend(ts_results)
            else:
                findings.extend(_scan_with_patterns(path, adapter_stub_patterns or []))
        if len(findings) >= max_findings:
            return findings[:max_findings]
    return findings


def scan_repo_for_stub_surface(
    workspace: Path,
    *,
    adapter: Optional[Any] = None,
    adapter_stub_patterns: Optional[list[str]] = None,
    max_files: int = 100000,
) -> list[str]:
    """Return the repo-relative paths whose public surface is still stubbed.

    Applies the existing AST stub detector (the same one
    :func:`scan_files_for_stubs` uses) repo-wide by rglob-ing every source
    file, minus test paths, and keeping a file iff it has at least one public
    stub finding. This is the "whole unimplemented surface" T2.1 needs for the
    decomposition partitioner — vs. the handful of localized focus files the
    planner normally feeds in.

    SIZE/STRUCTURE only: the caller decides whether a repo is
    decomposition-scale via :func:`scan_files_for_stubs`-derived counts; this
    helper is a pure scan with no repo/language conditional.
    """
    workspace = Path(workspace)
    if not workspace.is_dir():
        return []
    if adapter_stub_patterns is None and adapter is not None:
        try:
            adapter_stub_patterns = list(adapter.stub_patterns() or [])
        except Exception:
            adapter_stub_patterns = []
    adapter_stub_patterns = list(adapter_stub_patterns or [])

    stub_files: list[str] = []
    scanned = 0
    for path in sorted(workspace.rglob("*")):
        if scanned >= max_files:
            break
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix != ".py" and suffix not in _LANGUAGE_BY_SUFFIX:
            continue
        try:
            rel = str(path.relative_to(workspace))
        except ValueError:
            continue
        # Skip vendored / hidden / cache trees and test files.
        rel_posix = rel.replace("\\", "/")
        if any(
            part.startswith(".")
            or part in {"__pycache__", "node_modules", "site-packages", "dist-packages"}
            for part in Path(rel_posix).parts
        ):
            continue
        if _is_test_file(rel_posix, path):
            continue
        scanned += 1
        # Reuse the single-file detector with a high finding cap so a file
        # with many stubs still counts exactly once.
        findings = scan_files_for_stubs(
            workspace,
            [rel_posix],
            adapter_stub_patterns=adapter_stub_patterns,
            max_findings=1,
        )
        if findings:
            stub_files.append(rel_posix)
    return stub_files


def summarize_findings(findings: list[StubFinding], *, max_lines: int = 10) -> str:
    if not findings:
        return ""
    lines = ["Stub residue still present in changed files:"]
    for finding in findings[:max_lines]:
        lines.append(f"  - {finding.path}::{finding.symbol} — {finding.reason}")
    if len(findings) > max_lines:
        lines.append(f"  ... and {len(findings) - max_lines} more")
    return "\n".join(lines)
