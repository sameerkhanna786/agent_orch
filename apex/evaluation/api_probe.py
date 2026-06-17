"""Read-only API-surface probes for grounded test generation."""

from __future__ import annotations

import ast
import difflib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ApiSymbol:
    name: str
    kind: str
    signature: str = ""
    docstring: str = ""
    raises: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ApiProbeResult:
    language: str
    focal_path: str
    symbols: list[ApiSymbol] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    usage_examples: list[str] = field(default_factory=list)
    existing_test_usage: list[str] = field(default_factory=list)
    repo_context: dict[str, Any] = field(default_factory=dict)
    parse_error: str = ""

    @property
    def public_names(self) -> set[str]:
        if self.exports:
            return {name for name in self.exports if name}
        return {symbol.name.split(".", 1)[0] for symbol in self.symbols}

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "focal_path": self.focal_path,
            "symbols": [symbol.to_dict() for symbol in self.symbols],
            "exports": list(self.exports),
            "usage_examples": list(self.usage_examples),
            "existing_test_usage": list(self.existing_test_usage),
            "repo_context": dict(self.repo_context),
            "parse_error": self.parse_error,
        }

    def get_signature(self, symbol_name: str) -> str | None:
        target = str(symbol_name or "")
        for symbol in self.symbols:
            if symbol.name == target or symbol.name.rsplit(".", 1)[-1] == target:
                return symbol.signature or None
        return None

    def remaining_symbols(
        self, covered_names: set[str] | list[str] | tuple[str, ...] | None = None
    ) -> list[str]:
        covered = {str(name) for name in (covered_names or [])}
        return [
            symbol.name for symbol in self.symbols if symbol.name and symbol.name not in covered
        ]


def probe_api_surface(
    *,
    focal_source: str,
    focal_path: str,
    existing_test_source: str = "",
    repo_root: Path | None = None,
    language: str = "python",
) -> ApiProbeResult:
    normalized = (language or "").lower()
    if normalized in {"python", "py", "python3"}:
        result = probe_python_api_surface(
            focal_source=focal_source,
            focal_path=focal_path,
            existing_test_source=existing_test_source,
            repo_root=repo_root,
        )
        return result
    return ApiProbeResult(
        language=normalized or "unknown",
        focal_path=focal_path,
        existing_test_usage=harvest_existing_test_usage(existing_test_source),
    )


def probe_python_api_surface(
    *,
    focal_source: str,
    focal_path: str,
    existing_test_source: str = "",
    repo_root: Path | None = None,
) -> ApiProbeResult:
    try:
        tree = ast.parse(focal_source or "")
    except SyntaxError as exc:
        return ApiProbeResult(
            language="python",
            focal_path=focal_path,
            existing_test_usage=harvest_existing_test_usage(existing_test_source),
            parse_error=str(exc),
        )

    symbols: list[ApiSymbol] = []
    exports = _extract_python_exports(tree)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(node.name):
            symbols.append(
                ApiSymbol(
                    name=node.name,
                    kind="function",
                    signature=f"{node.name}{_python_signature(node)}",
                    docstring=_first_doc_line(ast.get_docstring(node)),
                    raises=_raised_exception_names(node),
                )
            )
        elif isinstance(node, ast.ClassDef) and _is_public(node.name):
            methods: list[str] = []
            raises: list[str] = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(
                    child.name
                ):
                    methods.append(f"{node.name}.{child.name}{_python_signature(child)}")
                    raises.extend(_raised_exception_names(child))
            symbols.append(
                ApiSymbol(
                    name=node.name,
                    kind="class",
                    signature=f"class {node.name}",
                    docstring=_first_doc_line(ast.get_docstring(node)),
                    raises=sorted(set(raises)),
                )
            )
            for method in methods[:16]:
                method_name = method.split("(", 1)[0]
                symbols.append(
                    ApiSymbol(
                        name=method_name,
                        kind="method",
                        signature=method,
                    )
                )
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and _is_public(target.id):
                    symbols.append(ApiSymbol(name=target.id, kind="constant"))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                name = alias.asname or alias.name
                if _is_public(name):
                    symbols.append(ApiSymbol(name=name, kind="reexport"))

    usage_examples = _harvest_repo_usage_examples(
        repo_root=repo_root,
        focal_path=focal_path,
        public_names=set(exports)
        if exports
        else {symbol.name.split(".", 1)[0] for symbol in symbols},
    )
    repo_context_payload: dict[str, Any] = {}
    try:
        from .repo_context import probe_repo_context

        repo_context_payload = probe_repo_context(
            repo_root,
            existing_test_source=existing_test_source,
        ).to_dict()
    except Exception:
        repo_context_payload = {}
    return ApiProbeResult(
        language="python",
        focal_path=focal_path,
        symbols=symbols[:80],
        exports=exports,
        usage_examples=usage_examples,
        existing_test_usage=harvest_existing_test_usage(existing_test_source),
        repo_context=repo_context_payload,
    )


def render_api_surface_prompt_block(probe: ApiProbeResult) -> str:
    lines: list[str] = []
    if probe.symbols:
        lines.extend(["## Verified API surface"])
        for symbol in probe.symbols[:30]:
            entry = f"- {symbol.signature or symbol.name}"
            details: list[str] = []
            if symbol.raises:
                details.append("raises " + ", ".join(symbol.raises[:4]))
            if symbol.docstring:
                details.append(symbol.docstring)
            if details:
                entry += "  # " + "; ".join(details)
            lines.append(entry)
    if probe.existing_test_usage:
        if lines:
            lines.append("")
        lines.extend(
            [
                "## Known-good usage from existing tests",
                *[f"- `{item}`" for item in probe.existing_test_usage[:12]],
            ]
        )
    if probe.usage_examples:
        if lines:
            lines.append("")
        lines.extend(
            [
                "## Known-good usage from repository call sites",
                *[f"- `{item}`" for item in probe.usage_examples[:8]],
            ]
        )
    return "\n".join(lines)


def harvest_existing_test_usage(source: str) -> list[str]:
    text = source or ""
    patterns = [
        r"\b[A-Z][A-Za-z0-9_]*\s*\([^()\n]{0,120}\)",
        r"\b[a-z_][A-Za-z0-9_]*\s*\([^()\n]{0,120}\)",
        r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\s*\([^()\n]{0,120}\)",
    ]
    usage: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            snippet = " ".join(match.group(0).strip().split())
            if snippet.startswith(("assert", "pytest.", "self.assert", "print", "super(")):
                continue
            if snippet not in seen:
                usage.append(snippet)
                seen.add(snippet)
            if len(usage) >= 24:
                return usage
    return usage


def find_missing_public_imports(
    *,
    test_source: str,
    focal_module: str,
    public_names: set[str],
) -> dict[str, str]:
    """Return imported focal names that do not exist, with closest matches."""

    detailed = find_missing_public_imports_detailed(
        test_source=test_source,
        focal_module=focal_module,
        public_names=public_names,
    )
    return {name: str(issue.get("closest") or "") for name, issue in detailed.items()}


def find_missing_public_imports_detailed(
    *,
    test_source: str,
    focal_module: str,
    public_names: set[str],
) -> dict[str, dict[str, str]]:
    """Return missing focal imports with suggestions and severity."""

    try:
        tree = ast.parse(test_source or "")
    except SyntaxError:
        return {}
    missing: dict[str, dict[str, str]] = {}
    public_name_set = {str(name) for name in public_names if str(name)}
    focal_parts = [part for part in str(focal_module or "").split(".") if part]
    focal_leaf = focal_parts[-1] if focal_parts else ""
    focal_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                if module != focal_module:
                    continue
                focal_aliases.add(alias.asname or module.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
            if focal_parts and module == ".".join(focal_parts[:-1]):
                for alias in node.names:
                    if alias.name == focal_leaf:
                        focal_aliases.add(alias.asname or alias.name)
        if isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
            if module != focal_module:
                continue
            for alias in node.names:
                name = alias.name
                if name == "*" or name in public_name_set:
                    continue
                suggestion = _closest_public_names(name, public_name_set)
                if _looks_like_external_module_import(
                    module=module,
                    imported_name=name,
                    focal_module=focal_module,
                    suggestion=suggestion,
                ):
                    continue
                missing[name] = {
                    "closest": suggestion,
                    "severity": "private" if name.startswith("_") else "error",
                }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        chain = _attribute_chain(node)
        if len(chain) < 2:
            continue
        attr = chain[-1]
        is_focal_alias_access = chain[0] in focal_aliases
        is_focal_qualified_access = focal_parts and chain[: len(focal_parts)] == focal_parts
        if not is_focal_alias_access and not is_focal_qualified_access:
            continue
        if attr in public_name_set:
            continue
        missing[".".join(chain)] = {
            "closest": _closest_public_names(attr, public_name_set),
            "severity": "private" if attr.startswith("_") else "error",
        }
    return missing


def _closest_public_names(name: str, public_names: set[str]) -> str:
    matches = difflib.get_close_matches(
        str(name or ""),
        sorted(public_names),
        n=3,
        cutoff=0.6,
    )
    return ", ".join(matches)


def _looks_like_external_module_import(
    *,
    module: str,
    imported_name: str,
    focal_module: str,
    suggestion: str,
) -> bool:
    if "." in str(focal_module or ""):
        return False
    if suggestion:
        return False
    return str(module or "") == str(focal_module or "") and str(imported_name or "").islower()


def find_unreferenced_public_symbols(
    *,
    test_source: str,
    focal_module: str,
    public_names: set[str],
) -> list[str]:
    """Return public focal symbols that generated tests do not mention."""

    if not public_names:
        return []
    try:
        tree = ast.parse(test_source or "")
    except SyntaxError:
        return sorted(public_names)
    focal_parts = [part for part in str(focal_module or "").split(".") if part]
    focal_leaf = focal_parts[-1] if focal_parts else ""
    focal_aliases: set[str] = set()
    imported_names: set[str] = set()
    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == focal_module or node.module.endswith("." + focal_leaf):
                for alias in node.names:
                    if alias.name != "*":
                        imported_names.add(alias.asname or alias.name)
                        if alias.name in public_names:
                            referenced.add(alias.name)
            elif focal_parts and node.module == ".".join(focal_parts[:-1]):
                for alias in node.names:
                    if alias.name == focal_leaf:
                        focal_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == focal_module:
                    focal_aliases.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.Name):
            if node.id in public_names:
                referenced.add(node.id)
            if node.id in imported_names and node.id in public_names:
                referenced.add(node.id)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        chain = _attribute_chain(node)
        if len(chain) < 2:
            continue
        attr = chain[-1]
        if attr not in public_names:
            continue
        if chain[0] in focal_aliases:
            referenced.add(attr)
        elif focal_parts and chain[: len(focal_parts)] == focal_parts:
            referenced.add(attr)
    return sorted(public_names - referenced)


def _python_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args: list[str] = []
    positional = list(node.args.posonlyargs) + list(node.args.args)
    defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
    for arg, default in zip(positional, defaults):
        if arg.arg == "self":
            continue
        rendered = arg.arg
        if default is not None:
            rendered += "=" + _safe_unparse(default)
        args.append(rendered)
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        rendered = arg.arg
        if default is not None:
            rendered += "=" + _safe_unparse(default)
        args.append(rendered)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    return f"({', '.join(args)})"


def _raised_exception_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Raise) or child.exc is None:
            continue
        if isinstance(child.exc, ast.Call):
            name = _expr_name(child.exc.func)
        else:
            name = _expr_name(child.exc)
        if name and name not in names:
            names.append(name)
    return names


def _harvest_repo_usage_examples(
    *,
    repo_root: Path | None,
    focal_path: str,
    public_names: set[str],
) -> list[str]:
    if repo_root is None or not public_names:
        return []
    root = Path(repo_root)
    if not root.exists():
        return []
    examples: list[str] = []
    seen: set[str] = set()
    focal_name = Path(focal_path).stem
    for candidate in list(root.rglob("*.py"))[:800]:
        if str(candidate.relative_to(root)).replace("\\", "/") == focal_path.replace("\\", "/"):
            continue
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name in sorted(public_names):
            for match in re.finditer(
                rf"\b(?:{re.escape(name)}|{re.escape(focal_name)}\.{re.escape(name)})\s*\([^()\n]{{0,120}}\)",
                text,
            ):
                snippet = " ".join(match.group(0).strip().split())
                if snippet not in seen:
                    examples.append(snippet)
                    seen.add(snippet)
                    if len(examples) >= 8:
                        return examples
    return examples


def _expr_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expr_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _attribute_chain(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [*_attribute_chain(node.value), node.attr]
    return []


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "..."


def _extract_python_exports(tree: ast.Module) -> list[str]:
    exports: list[str] = []
    seen: set[str] = set()
    for node in tree.body:
        value: ast.AST | None = None
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            continue
        for name in _literal_string_sequence(value):
            if name not in seen:
                exports.append(name)
                seen.add(name)
    return exports


def _literal_string_sequence(node: ast.AST | None) -> list[str]:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values: list[str] = []
        for item in node.elts:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                values.append(item.value)
        return values
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    return []


def _first_doc_line(docstring: str | None) -> str:
    return (docstring or "").strip().splitlines()[0][:160] if docstring else ""


def _is_public(name: str) -> bool:
    return bool(name) and not name.startswith("_")
