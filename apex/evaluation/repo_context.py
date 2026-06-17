"""Repository test-context probing for generated-test prompts."""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Fixture:
    name: str
    scope: str = "function"
    file: str = ""
    signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RepoContext:
    fixtures: list[Fixture] = field(default_factory=list)
    conftest_imports: list[str] = field(default_factory=list)
    setup_teardown_pattern: str = "none"
    isolation_required: bool = False
    isolation_marker: str = ""
    skip_markers: set[str] = field(default_factory=set)
    forbidden_generated_names: set[str] = field(default_factory=set)
    focal_symbols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixtures": [fixture.to_dict() for fixture in self.fixtures],
            "conftest_imports": list(self.conftest_imports),
            "setup_teardown_pattern": self.setup_teardown_pattern,
            "isolation_required": self.isolation_required,
            "isolation_marker": self.isolation_marker,
            "skip_markers": sorted(self.skip_markers),
            "forbidden_generated_names": sorted(self.forbidden_generated_names),
            "focal_symbols": list(self.focal_symbols),
        }


@dataclass(frozen=True)
class NamespaceFinding:
    name: str
    kind: str
    line: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NamespaceValidationResult:
    status: str
    findings: list[NamespaceFinding] = field(default_factory=list)
    parse_error: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "findings": [finding.to_dict() for finding in self.findings],
            "parse_error": self.parse_error,
        }


@dataclass(frozen=True)
class ContextSnippet:
    path: str
    kind: str
    score: float
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "score": round(float(self.score or 0.0), 4),
            "content": self.content,
        }


@dataclass(frozen=True)
class TestgenContextPack:
    snippets: list[ContextSnippet] = field(default_factory=list)
    focal_symbols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snippets": [snippet.to_dict() for snippet in self.snippets],
            "focal_symbols": list(self.focal_symbols),
        }


def probe_repo_context(
    repo_root: Path | None,
    *,
    existing_test_source: str = "",
    existing_test_path: str = "",
    focal_source: str = "",
) -> RepoContext:
    if repo_root is None:
        return _context_from_sources([existing_test_source], focal_source=focal_source)
    root = Path(repo_root)
    sources: list[tuple[str, str]] = []
    if existing_test_source:
        sources.append((existing_test_path or "<existing>", existing_test_source))
    if root.exists():
        for candidate in list(root.rglob("conftest.py"))[:20]:
            sources.append((_rel(candidate, root), _read(candidate)))
        for candidate in list(root.rglob("test*.py"))[:80]:
            sources.append((_rel(candidate, root), _read(candidate)))
    return _context_from_sources(
        [text for _, text in sources],
        source_files=sources,
        focal_source=focal_source,
    )


def render_repo_context_block(context: RepoContext | dict[str, Any]) -> str:
    payload = context.to_dict() if isinstance(context, RepoContext) else dict(context or {})
    lines = ["## Repository test context"]
    fixtures = list(payload.get("fixtures") or [])
    if fixtures:
        lines.append(
            "- Fixtures: " + ", ".join(str(item.get("name") or "") for item in fixtures[:16])
        )
    setup = str(payload.get("setup_teardown_pattern") or "none")
    lines.append(f"- Setup/teardown: {setup}")
    if payload.get("isolation_required"):
        lines.append(f"- Isolation marker: {payload.get('isolation_marker') or 'required'}")
    skip = list(payload.get("skip_markers") or [])
    if skip:
        lines.append("- Observed skip markers: " + ", ".join(skip[:8]))
    forbidden = list(payload.get("forbidden_generated_names") or [])
    if forbidden:
        lines.append(
            "- Do not redefine existing top-level test helpers/classes: "
            + ", ".join(forbidden[:16])
        )
    focal_symbols = list(payload.get("focal_symbols") or [])
    if focal_symbols:
        lines.append("- Focal symbols available: " + ", ".join(focal_symbols[:16]))
    return "\n".join(lines)


def validate_generated_namespace(
    source: str,
    context: RepoContext | dict[str, Any] | None = None,
    *,
    forbidden_generated_names: Iterable[str] | None = None,
    required_focal_symbols: Iterable[str] | None = None,
) -> NamespaceValidationResult:
    """Fail generated tests that collide with repo-owned names.

    This is a static, language-adapter-friendly policy surface. Python uses
    AST extraction here; other language adapters can provide the same finding
    shape without importing this implementation.
    """

    text = str(source or "")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return NamespaceValidationResult(status="skipped", parse_error=f"SyntaxError: {exc}")

    payload = context.to_dict() if isinstance(context, RepoContext) else dict(context or {})
    forbidden = set(str(name) for name in (forbidden_generated_names or ()))
    forbidden.update(str(name) for name in payload.get("forbidden_generated_names") or ())
    required = set(str(name) for name in (required_focal_symbols or ()))
    available = set(str(name) for name in payload.get("focal_symbols") or ())

    findings: list[NamespaceFinding] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name in forbidden:
            findings.append(
                NamespaceFinding(
                    name=node.name,
                    kind="namespace_collision",
                    line=getattr(node, "lineno", 0),
                    reason=f"generated top-level symbol collides with existing repo test symbol {node.name!r}",
                )
            )
    for symbol in sorted(required - available):
        findings.append(
            NamespaceFinding(
                name=symbol,
                kind="missing_focal_symbol",
                reason=f"required focal symbol {symbol!r} is not available in repo context",
            )
        )
    return NamespaceValidationResult(
        status="fail" if findings else "pass",
        findings=findings,
    )


def retrieve_testgen_context(
    repo_root: Path | None,
    *,
    focal_path: str = "",
    focal_source: str = "",
    existing_test_path: str = "",
    existing_test_source: str = "",
    max_snippets: int = 6,
    max_chars_per_snippet: int = 1400,
) -> TestgenContextPack:
    """Retrieve compact, benchmark-agnostic context for test generation.

    The scorer is intentionally capability-based: prefer existing tests,
    fixtures, path-nearby tests, and snippets mentioning focal symbols. Dataset
    adapters can pass whatever repo root they have; this function does not care
    which benchmark supplied the task.
    """

    focal_symbols = _focal_symbol_names(focal_source)
    root = Path(repo_root) if repo_root is not None else None
    candidates: list[ContextSnippet] = []
    if existing_test_source.strip():
        candidates.append(
            ContextSnippet(
                path=existing_test_path or "<existing_test>",
                kind="existing_test",
                score=1000.0,
                content=_compact_source(existing_test_source, max_chars_per_snippet),
            )
        )
    if root is not None and root.exists():
        focal_parts = [part for part in Path(focal_path or "").parts if part]
        focal_dir = "/".join(focal_parts[:-1])
        for path in _iter_context_files(root):
            rel = _rel(path, root)
            if rel == existing_test_path:
                continue
            text = _read(path)
            if not text.strip():
                continue
            score, kind = _score_context_file(
                rel,
                text,
                focal_path=focal_path,
                focal_dir=focal_dir,
                focal_symbols=focal_symbols,
            )
            if score <= 0:
                continue
            candidates.append(
                ContextSnippet(
                    path=rel,
                    kind=kind,
                    score=score,
                    content=_compact_source(text, max_chars_per_snippet),
                )
            )
    ranked = sorted(candidates, key=lambda item: (item.score, -len(item.content)), reverse=True)
    deduped: list[ContextSnippet] = []
    seen_paths: set[str] = set()
    for snippet in ranked:
        if snippet.path in seen_paths:
            continue
        seen_paths.add(snippet.path)
        deduped.append(snippet)
        if len(deduped) >= max(0, int(max_snippets or 0)):
            break
    return TestgenContextPack(snippets=deduped, focal_symbols=focal_symbols)


def render_testgen_context_pack(pack: TestgenContextPack | dict[str, Any]) -> str:
    payload = pack.to_dict() if isinstance(pack, TestgenContextPack) else dict(pack or {})
    snippets = [item for item in list(payload.get("snippets") or []) if isinstance(item, dict)]
    if not snippets:
        return ""
    lines = [
        "## Retrieved repository context",
        "Use these snippets for fixtures, imports, helper factories, and style. Prefer these repo-native patterns over inventing new setup.",
    ]
    for snippet in snippets[:8]:
        content = str(snippet.get("content") or "").strip()
        if not content:
            continue
        lines.extend(
            [
                "",
                f"### {snippet.get('path') or '<context>'} ({snippet.get('kind') or 'context'})",
                "```python",
                content,
                "```",
            ]
        )
    return "\n".join(lines)


def _context_from_sources(
    sources: list[str],
    *,
    source_files: list[tuple[str, str]] | None = None,
    focal_source: str = "",
) -> RepoContext:
    fixtures: list[Fixture] = []
    conftest_imports: list[str] = []
    setup = "none"
    isolation = False
    isolation_marker = ""
    skip_markers: set[str] = set()
    forbidden_names: set[str] = set()
    files = source_files or [(f"<source-{i}>", source) for i, source in enumerate(sources)]
    for file_name, source in files:
        text = source or ""
        if "@pytest.mark.django_db" in text or "pytest.mark.django_db" in text:
            isolation = True
            isolation_marker = "@pytest.mark.django_db"
        for marker in re.findall(r"@pytest\.mark\.(skipif|skip)\b", text):
            skip_markers.add(marker)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        forbidden_names.update(_top_level_symbol_names(tree))
        if file_name.endswith("conftest.py"):
            conftest_imports.extend(_top_level_import_lines(text))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "setup_module":
                    setup = "module_setup"
                fixture_scope = _fixture_scope(node)
                if fixture_scope:
                    fixtures.append(
                        Fixture(
                            name=node.name,
                            scope=fixture_scope,
                            file=file_name,
                            signature=f"{node.name}({_arg_names(node)})",
                        )
                    )
            elif isinstance(node, ast.ClassDef):
                method_names = {
                    child.name
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                if {"setUp", "tearDown"} & method_names:
                    setup = "unittest"
    return RepoContext(
        fixtures=fixtures,
        conftest_imports=conftest_imports,
        setup_teardown_pattern=setup,
        isolation_required=isolation,
        isolation_marker=isolation_marker,
        skip_markers=skip_markers,
        forbidden_generated_names=forbidden_names,
        focal_symbols=_focal_symbol_names(focal_source),
    )


def _top_level_symbol_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("test_"):
                continue
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    names.add(target.id)
    return names


def _iter_context_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.py"):
        rel_parts = path.relative_to(root).parts
        if any(
            part in {".git", ".hg", ".venv", "venv", "node_modules", "__pycache__"}
            for part in rel_parts
        ):
            continue
        if any(part.endswith(".egg-info") or part == "site-packages" for part in rel_parts):
            continue
        name = path.name
        if name == "conftest.py" or name.startswith("test") or name.endswith("_test.py"):
            files.append(path)
        if len(files) >= 160:
            break
    return files


def _score_context_file(
    rel_path: str,
    text: str,
    *,
    focal_path: str,
    focal_dir: str,
    focal_symbols: list[str],
) -> tuple[float, str]:
    rel = rel_path.replace("\\", "/")
    score = 0.0
    kind = "test"
    if rel.endswith("conftest.py"):
        score += 300.0
        kind = "fixture"
    if rel.startswith(focal_dir.rstrip("/") + "/") and focal_dir:
        score += 120.0
    if focal_path and Path(rel).parent == Path(focal_path).parent:
        score += 80.0
    symbol_hits = sum(1 for symbol in focal_symbols if re.search(rf"\b{re.escape(symbol)}\b", text))
    score += min(symbol_hits, 8) * 40.0
    focal_leaf = Path(focal_path or "").stem
    if focal_leaf and re.search(rf"\b{re.escape(focal_leaf)}\b", text):
        score += 30.0
    if "@pytest.fixture" in text:
        score += 80.0
        kind = "fixture"
    if "factory" in text.lower() or "fixture" in text.lower():
        score += 20.0
    return score, kind


def _compact_source(source: str, max_chars: int) -> str:
    text = source or ""
    if len(text) <= max_chars:
        return text.strip()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text[:max_chars].rstrip() + "\n# ... truncated ..."
    chunks: list[str] = []
    imports = _top_level_import_lines(text)
    if imports:
        chunks.append("\n".join(imports[:20]))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            segment = ast.get_source_segment(text, node) or ""
            if not segment.strip():
                continue
            if len(segment) > max_chars // 2:
                header = segment.splitlines()[0] if segment.splitlines() else ""
                segment = header + "\n    # ... body truncated ..."
            chunks.append(segment.strip())
            if len("\n\n".join(chunks)) >= max_chars:
                break
    compact = "\n\n".join(chunks).strip() or text[:max_chars].rstrip()
    if len(compact) > max_chars:
        compact = compact[:max_chars].rstrip() + "\n# ... truncated ..."
    return compact


def _focal_symbol_names(source: str) -> list[str]:
    try:
        tree = ast.parse(source or "")
    except SyntaxError:
        return []
    names: list[str] = []
    for node in tree.body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ) and not node.name.startswith("_"):
            names.append(node.name)
    return names[:40]


def _fixture_scope(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    for decorator in node.decorator_list:
        text = ast.unparse(decorator)
        if text.startswith("pytest.fixture"):
            match = re.search(r"scope=['\"]([^'\"]+)['\"]", text)
            return match.group(1) if match else "function"
    return ""


def _arg_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    return ", ".join(arg.arg for arg in node.args.args)


def _top_level_import_lines(source: str) -> list[str]:
    return [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ][:40]


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)
