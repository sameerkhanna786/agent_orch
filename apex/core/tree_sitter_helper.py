"""Optional tree-sitter wrapper for cross-language AST analysis.

Falls back gracefully when tree-sitter / tree-sitter-languages aren't
installed: callers see ``None`` from ``parse_source`` and skip the
AST-based path. The Python AST path remains the canonical implementation
for Python; this module is purely about JS / TS / Go / Rust / Java /
Ruby support for SWE-Bench Pro and similar polyglot benchmarks.

Two operations are exposed:

* :func:`top_level_public_symbols(source, language)` returns the set of
  ``(name, kind)`` tuples for top-level public declarations. Used by
  :mod:`apex.core.symbol_survival` to detect deletions.
* :func:`stub_function_findings(source, language)` returns the list of
  function declarations whose bodies look like unimplemented stubs
  (empty body, ``throw new Error("not implemented")`` /
  ``unimplemented!()`` / ``panic("not implemented")`` / etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

try:
    import tree_sitter_languages  # type: ignore[import]

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


_PARSER_CACHE: dict[str, object] = {}


_LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sc": "scala",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".cs": "c_sharp",
    ".php": "php",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
    ".bats": "bash",
}


def language_for_suffix(suffix: str) -> Optional[str]:
    return _LANGUAGE_BY_SUFFIX.get(suffix.lower())


def is_available() -> bool:
    return _AVAILABLE


def _parser_for(language: str) -> Optional[object]:
    if not _AVAILABLE:
        return None
    cached = _PARSER_CACHE.get(language)
    if cached is not None:
        return cached
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            parser = tree_sitter_languages.get_parser(language)
    except Exception:
        return None
    _PARSER_CACHE[language] = parser
    return parser


def parse_source(source: str, language: str):
    """Return a tree-sitter Tree, or ``None`` if parsing isn't possible."""
    parser = _parser_for(language)
    if parser is None:
        return None
    try:
        return parser.parse(source.encode("utf-8", errors="replace"))
    except Exception:
        return None


# Top-level declaration node types per language (covers the public API
# surface the agent might delete).
_TOP_LEVEL_DECL_NODES: dict[str, dict[str, str]] = {
    "javascript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "lexical_declaration": "assignment",
        "variable_declaration": "assignment",
    },
    "typescript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
        "enum_declaration": "enum",
        "lexical_declaration": "assignment",
        "variable_declaration": "assignment",
    },
    "tsx": {
        "function_declaration": "function",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
        "enum_declaration": "enum",
        "lexical_declaration": "assignment",
        "variable_declaration": "assignment",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_declaration": "type",
        "var_declaration": "assignment",
        "const_declaration": "assignment",
    },
    "rust": {
        "function_item": "function",
        "struct_item": "struct",
        "enum_item": "enum",
        "trait_item": "trait",
        "type_item": "type",
        "const_item": "assignment",
        "static_item": "assignment",
        "impl_item": "impl",
    },
    "java": {
        "method_declaration": "method",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
        "record_declaration": "record",
    },
    "ruby": {
        "method": "method",
        "class": "class",
        "module": "module",
    },
    "c": {
        "function_definition": "function",
        "declaration": "assignment",
        "type_definition": "type",
        "struct_specifier": "struct",
        "union_specifier": "union",
        "enum_specifier": "enum",
    },
    "cpp": {
        "function_definition": "function",
        "class_specifier": "class",
        "struct_specifier": "struct",
        "namespace_definition": "namespace",
        "template_declaration": "template",
        "type_definition": "type",
        "enum_specifier": "enum",
    },
    "c_sharp": {
        "method_declaration": "method",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "struct_declaration": "struct",
        "enum_declaration": "enum",
        "record_declaration": "record",
        "namespace_declaration": "namespace",
    },
    "kotlin": {
        "function_declaration": "function",
        "class_declaration": "class",
        "object_declaration": "object",
        "property_declaration": "property",
    },
    "scala": {
        "function_definition": "function",
        "class_definition": "class",
        "object_definition": "object",
        "trait_definition": "trait",
        "val_definition": "value",
    },
    "php": {
        "function_definition": "function",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "trait_declaration": "trait",
        "enum_declaration": "enum",
    },
    "lua": {
        "function_declaration": "function",
        "variable_declaration": "assignment",
    },
    "bash": {
        "function_definition": "function",
    },
}


def _node_field_text(node, field: str) -> str:
    try:
        child = node.child_by_field_name(field)
    except Exception:
        return ""
    if child is None:
        return ""
    return (child.text or b"").decode("utf-8", errors="replace")


def _is_public_in_language(node, language: str, name: str) -> bool:
    """Best-effort visibility check per language."""
    if language in ("javascript", "typescript", "tsx"):
        # JS uses `export` + leading-underscore convention. Walk up
        # looking for an export modifier on the parent.
        try:
            parent = node.parent
        except Exception:
            parent = None
        while parent is not None:
            if parent.type in ("export_statement",):
                return not name.startswith("_")
            parent_type = getattr(parent, "type", "")
            if parent_type == "program":
                # No export wrapper found: assume internal — top-level
                # but private to module. Still flag because tests may
                # import via `require()` without `export`.
                return not name.startswith("_")
            parent = getattr(parent, "parent", None)
        return not name.startswith("_")
    if language == "go":
        # Go convention: capitalized name == exported.
        return name[:1].isupper()
    if language == "rust":
        # Rust: look for `pub` modifier sibling.
        try:
            text = (node.text or b"").decode("utf-8", errors="replace")
        except Exception:
            text = ""
        return text.lstrip().startswith("pub ") or text.lstrip().startswith("pub(")
    if language == "java":
        try:
            text = (node.text or b"").decode("utf-8", errors="replace")
        except Exception:
            text = ""
        head = text[:200]
        return "public " in head or "protected " in head
    if language == "ruby":
        return not name.startswith("_")
    if language in ("c", "cpp"):
        # C/C++ has no convention; leading-underscore is reserved for
        # implementation but most public APIs don't use it. Treat all
        # top-level decls as public unless explicitly file-static.
        try:
            text = (node.text or b"").decode("utf-8", errors="replace")
        except Exception:
            text = ""
        return "static " not in text[:64]
    if language == "c_sharp":
        try:
            text = (node.text or b"").decode("utf-8", errors="replace")
        except Exception:
            text = ""
        head = text[:200]
        return "public " in head or "internal " in head or "protected " in head
    if language == "kotlin":
        try:
            text = (node.text or b"").decode("utf-8", errors="replace")
        except Exception:
            text = ""
        head = text[:120]
        # Kotlin defaults to public when no modifier is present.
        if "private " in head or "internal " in head:
            return False
        return not name.startswith("_")
    if language == "scala":
        try:
            text = (node.text or b"").decode("utf-8", errors="replace")
        except Exception:
            text = ""
        head = text[:120]
        if "private" in head:
            return False
        return not name.startswith("_")
    if language == "php":
        try:
            text = (node.text or b"").decode("utf-8", errors="replace")
        except Exception:
            text = ""
        head = text[:200]
        if "private " in head:
            return False
        return not name.startswith("_")
    if language in ("lua", "bash"):
        return not name.startswith("_")
    return not name.startswith("_")


def _extract_node_name(node, language: str) -> str:
    name = _node_field_text(node, "name")
    if name:
        return name
    # var / lexical declarations in JS contain declarators with names.
    for child in node.children:
        if child.type in ("variable_declarator", "init_declarator"):
            return _node_field_text(child, "name")
    # C / C++: function_definition wraps a function_declarator that
    # holds the identifier under the ``declarator`` field.
    if language in ("c", "cpp"):
        for child in node.children:
            if child.type == "function_declarator":
                ident = _node_field_text(child, "declarator")
                if ident:
                    return ident
                for grandchild in child.children:
                    if grandchild.type == "identifier":
                        return (grandchild.text or b"").decode("utf-8", errors="replace")
    # PHP method_declaration / function_definition: name field works
    # but some grammar versions wrap it in a name node.
    if language == "php":
        for child in node.children:
            if child.type == "name":
                return (child.text or b"").decode("utf-8", errors="replace")
    return ""


_NAMESPACE_LIKE_NODES = {
    "c_sharp": {"namespace_declaration", "file_scoped_namespace_declaration"},
    "cpp": {"namespace_definition"},
    "php": {"namespace_definition", "namespace_use_declaration"},
    "java": {"package_declaration"},
    "kotlin": {"package_header"},
    "scala": {"package_clause"},
}


def _walk_top_level(root, language: str):
    """Yield top-level decl nodes, descending into namespace/package wrappers."""
    namespace_nodes = _NAMESPACE_LIKE_NODES.get(language, set())
    stack = list(root.children)
    while stack:
        node = stack.pop(0)
        if node.type in namespace_nodes:
            # The body of a C# / C++ / PHP namespace block is its
            # declaration_list child; iterate that.
            body = None
            for field in ("body", "declarations"):
                try:
                    body = node.child_by_field_name(field)
                except Exception:
                    body = None
                if body is not None:
                    break
            if body is not None:
                stack.extend(body.children)
                continue
            # File-scoped namespace (C# 10+): everything after the
            # namespace declaration is part of it. Descend siblings.
            stack.extend(node.children)
            continue
        yield node


def top_level_public_symbols(source: str, language: str) -> set[tuple[str, str]]:
    """Return the public top-level ``(name, kind)`` symbols in ``source``."""
    if not _AVAILABLE:
        return set()
    decl_nodes = _TOP_LEVEL_DECL_NODES.get(language)
    if not decl_nodes:
        return set()
    tree = parse_source(source, language)
    if tree is None:
        return set()
    symbols: set[tuple[str, str]] = set()
    for child in _walk_top_level(tree.root_node, language):
        node_type = child.type
        # Some languages wrap decls in an export_statement etc.
        target = child
        if node_type in ("export_statement", "decorated_definition"):
            inner = next(
                (c for c in child.children if c.type in decl_nodes),
                None,
            )
            if inner is not None:
                target = inner
                node_type = inner.type
        kind = decl_nodes.get(node_type)
        if not kind:
            continue
        name = _extract_node_name(target, language)
        if not name:
            continue
        if _is_public_in_language(target, language, name):
            symbols.add((name, kind))
    return symbols


# Per-language stub-body queries: a function body that consists ONLY of
# the listed node types / contains the listed text patterns counts as a
# stub. Tree-sitter queries would be cleaner but the API differs across
# tree-sitter versions; we use simpler text-on-body matching.
_STUB_BODY_PATTERNS: dict[str, list[str]] = {
    "javascript": [
        r'^\s*throw\s+new\s+Error\s*\(\s*[\'"`].*not.{0,5}implemented.*[\'"`]\s*\)\s*;?\s*$',
        r'^\s*throw\s+new\s+Error\s*\(\s*[\'"`].*TODO.*[\'"`]\s*\)\s*;?\s*$',
        r"^\s*return\s+undefined\s*;?\s*$",
        r"^\s*return\s+null\s*;?\s*//.*TODO.*$",
        r"^\s*return\s*;?\s*$",
        r"^\s*//\s*TODO.*$",
    ],
    "typescript": None,  # filled in below as alias
    "tsx": None,
    "go": [
        r'^\s*panic\s*\(\s*[\'"`].*not.{0,5}implemented.*[\'"`]\s*\)\s*$',
        r'^\s*panic\s*\(\s*[\'"`].*TODO.*[\'"`]\s*\)\s*$',
        r"^\s*//\s*TODO.*$",
        r"^\s*return\s+nil\s*$",
    ],
    "rust": [
        r"^\s*unimplemented!\s*\(",
        r"^\s*todo!\s*\(",
        r'^\s*panic!\s*\(\s*[\'"`].*not.{0,5}implemented.*[\'"`]',
        r"^\s*//\s*TODO.*$",
    ],
    "java": [
        r"^\s*throw\s+new\s+UnsupportedOperationException\s*\(",
        r'^\s*throw\s+new\s+RuntimeException\s*\(\s*[\'"`].*not.{0,5}implemented.*[\'"`]',
        r"^\s*throw\s+new\s+NotImplementedError\s*\(",
        r"^\s*//\s*TODO.*$",
        r"^\s*return\s+null\s*;\s*//.*TODO",
    ],
    "ruby": [
        r"^\s*raise\s+NotImplementedError",
        r"^\s*#\s*TODO",
    ],
}
_STUB_BODY_PATTERNS["typescript"] = _STUB_BODY_PATTERNS["javascript"]
_STUB_BODY_PATTERNS["tsx"] = _STUB_BODY_PATTERNS["javascript"]
_STUB_BODY_PATTERNS["c"] = [
    r'^\s*assert\s*\(\s*(0|false)\s*&&\s*[\'"`].*not.{0,5}implemented.*[\'"`]\s*\)\s*;?\s*$',
    r"^\s*abort\s*\(\s*\)\s*;?\s*//.*not.{0,5}implemented",
    r"^\s*//\s*TODO[: ]",
    r"^\s*/\*\s*TODO",
    r"^\s*return\s+(NULL|0)\s*;\s*//.*TODO",
]
_STUB_BODY_PATTERNS["cpp"] = list(_STUB_BODY_PATTERNS["c"]) + [
    r'^\s*throw\s+std::runtime_error\s*\(\s*[\'"`].*not.{0,5}implemented.*[\'"`]\s*\)\s*;?\s*$',
    r'^\s*throw\s+std::logic_error\s*\(\s*[\'"`].*not.{0,5}implemented.*[\'"`]\s*\)\s*;?\s*$',
    r'^\s*static_assert\s*\(\s*false\s*,\s*[\'"`].*not.{0,5}implemented.*[\'"`]',
]
_STUB_BODY_PATTERNS["c_sharp"] = [
    r"^\s*throw\s+new\s+NotImplementedException\s*\(",
    r'^\s*throw\s+new\s+NotSupportedException\s*\(\s*[\'"`].*TODO.*[\'"`]\s*\)\s*;?\s*$',
    r"^\s*//\s*TODO[: ]",
    r"^\s*return\s+null\s*;\s*//.*TODO",
    r"^\s*return\s+default\s*;\s*//.*TODO",
]
_STUB_BODY_PATTERNS["kotlin"] = [
    r"^\s*TODO\s*\(",
    r"^\s*throw\s+NotImplementedError\s*\(",
    r"^\s*//\s*TODO[: ]",
]
_STUB_BODY_PATTERNS["scala"] = [
    r"^\s*\?\?\?\s*$",  # Scala's idiomatic "not implemented" sentinel
    r"^\s*throw\s+new\s+NotImplementedError\s*\(?",
    r"^\s*//\s*TODO[: ]",
]
_STUB_BODY_PATTERNS["php"] = [
    r'^\s*throw\s+new\s+\\?LogicException\s*\(\s*[\'"`].*not.{0,5}implemented.*[\'"`]\s*\)\s*;?\s*$',
    r'^\s*throw\s+new\s+\\?RuntimeException\s*\(\s*[\'"`].*not.{0,5}implemented.*[\'"`]\s*\)\s*;?\s*$',
    r"^\s*throw\s+new\s+\\?BadMethodCallException\s*\(",
    r"^\s*//\s*TODO[: ]",
    r"^\s*#\s*TODO[: ]",
]
_STUB_BODY_PATTERNS["lua"] = [
    r'^\s*error\s*\(\s*[\'"`].*not.{0,5}implemented.*[\'"`]\s*\)\s*$',
    r"^\s*--\s*TODO[: ]",
    r"^\s*return\s+nil\s*--.*TODO",
]
_STUB_BODY_PATTERNS["bash"] = [
    r'^\s*echo\s+[\'"`].*not.{0,5}implemented.*[\'"`]\s*$',
    r"^\s*return\s+1\s*#\s*TODO",
    r"^\s*#\s*TODO[: ]",
    r"^\s*:\s*$",  # bash no-op
]


@dataclass(frozen=True)
class StubLocation:
    name: str
    line: int
    reason: str


def _function_node_types(language: str) -> set[str]:
    return {
        "javascript": {
            "function_declaration",
            "method_definition",
            "function_expression",
            "arrow_function",
        },
        "typescript": {
            "function_declaration",
            "method_definition",
            "function_expression",
            "arrow_function",
        },
        "tsx": {
            "function_declaration",
            "method_definition",
            "function_expression",
            "arrow_function",
        },
        "go": {"function_declaration", "method_declaration"},
        "rust": {"function_item"},
        "java": {"method_declaration"},
        "ruby": {"method"},
        "c": {"function_definition"},
        "cpp": {"function_definition"},
        "c_sharp": {"method_declaration", "constructor_declaration", "local_function_statement"},
        "kotlin": {"function_declaration"},
        "scala": {"function_definition"},
        "php": {"function_definition", "method_declaration"},
        "lua": {"function_declaration", "function_definition_statement"},
        "bash": {"function_definition"},
    }.get(language, set())


def stub_function_findings(source: str, language: str) -> list[StubLocation]:
    """Return locations of suspected stub function bodies.

    Pure-text body inspection over tree-sitter-found function nodes.
    Less precise than a structured tree-sitter query but works across
    the language's node-type variations without per-version API churn.
    """
    if not _AVAILABLE:
        return []
    targets = _function_node_types(language)
    if not targets:
        return []
    patterns = _STUB_BODY_PATTERNS.get(language) or []
    if not patterns:
        return []
    tree = parse_source(source, language)
    if tree is None:
        return []
    compiled = []
    for pat in patterns:
        try:
            compiled.append(re.compile(pat, re.MULTILINE | re.IGNORECASE))
        except re.error:
            continue
    findings: list[StubLocation] = []

    def _body_inner(text: str) -> str:
        # Strip outer braces (brace-languages: JS / Go / Rust / Java) so
        # patterns can match the actual body content.
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            stripped = stripped[1:-1]
        # Ruby methods have `def ... end` shape; tree-sitter's body
        # field already excludes the `def`/`end` keywords.
        return stripped.strip()

    def visit(node) -> None:
        if node.type in targets:
            name = _extract_node_name(node, language) or "<anonymous>"
            if not name.startswith("_"):
                body_node = None
                for field in ("body", "block"):
                    try:
                        body_node = node.child_by_field_name(field)
                    except Exception:
                        body_node = None
                    if body_node is not None:
                        break
                if body_node is not None:
                    body_text = (body_node.text or b"").decode("utf-8", errors="replace")
                    inner = _body_inner(body_text)
                    if not inner or inner in ("()", "() => {}"):
                        findings.append(StubLocation(name, node.start_point[0] + 1, "empty body"))
                    else:
                        for cre in compiled:
                            if cre.search(inner):
                                findings.append(
                                    StubLocation(
                                        name,
                                        node.start_point[0] + 1,
                                        f"matches stub pattern /{cre.pattern}/",
                                    )
                                )
                                break
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return findings
