"""Mine existing tests in the repo to use as few-shot exemplars.

Today the test_writer agent improvises test scaffolding from scratch
every iteration: import paths, fixture style, decorator usage, assertion
idioms — all guessed, often wrong. Strong testgen systems (arXiv
2602.12256, "Automated Test Suite Enhancement w/ Few-shot") consistently
show that injecting similar existing tests as exemplars wins, because
the agent then COPIES the project's idioms instead of inventing them.

This module scans the repo's existing test files, scores each by
similarity to the agent's focus files, and returns the top-K most
similar tests as compact snippets ready to inject into the prompt.

Design notes:
    * Pure helper — no engine / orchestrator dependency. The same code
      runs for benchmark tasks, IDE plugins, and CI integrations.
    * Cost-bounded by default (200 file cap, 30-line snippet cap).
    * Similarity is structural (imports, decorators, target symbols),
      not embedding-based — keeps it fast and dependency-free.
    * Defensive: returns [] cleanly when the repo has no tests, the
      AST parse fails, or focus_files is empty.

Public API:
    extract_test_exemplars(repo_path, focus_files, ...)
        Returns ``list[TestExemplar]`` ranked by similarity score.
    render_exemplars_prompt_block(exemplars)
        Renders a markdown section ready to drop into a prompt.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


_DEFAULT_MAX_FILES_SCANNED = 200
_DEFAULT_MAX_SNIPPET_LINES = 30
_DEFAULT_TOP_K = 3
_TEST_FILE_GLOBS = (
    "test_*.py",
    "*_test.py",
    "*.test.js",
    "*.spec.js",
    "*.test.jsx",
    "*.spec.jsx",
    "*.test.ts",
    "*.spec.ts",
    "*.test.tsx",
    "*.spec.tsx",
    "*Test.js",
    "*Spec.js",
    "*Test.jsx",
    "*Spec.jsx",
    "*Test.ts",
    "*Spec.ts",
    "*Test.tsx",
    "*Spec.tsx",
    "*_test.go",
    "*Test.java",
    "*Tests.java",
    "*IT.java",
    "*Test.kt",
    "*Tests.kt",
    "*Test.php",
    "*Tests.swift",
)


@dataclass
class TestExemplar:
    """One example test snippet from the repo, scored for similarity."""

    # The dataclass name happens to start with "Test" but it's not a
    # pytest test class — suppress the collection warning.
    __test__ = False

    path: str  # repo-relative
    snippet: str  # source text of one test function
    imports: list[str] = field(default_factory=list)
    target_symbols: list[str] = field(default_factory=list)
    score: float = 0.0
    reason: str = ""
    language: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "snippet": self.snippet,
            "imports": list(self.imports),
            "target_symbols": list(self.target_symbols),
            "score": round(self.score, 4),
            "reason": self.reason,
            "language": self.language,
        }


def extract_test_exemplars(
    *,
    repo_path: str | Path,
    focus_files: list[str] | None,
    issue_keywords: list[str] | None = None,
    top_k: int = _DEFAULT_TOP_K,
    max_files_scanned: int = _DEFAULT_MAX_FILES_SCANNED,
    max_snippet_lines: int = _DEFAULT_MAX_SNIPPET_LINES,
) -> list[TestExemplar]:
    """Mine the repo's existing tests; return the top-K most similar
    to the agent's targets.

    Similarity is structural and additive — each signal contributes a
    bounded weight:

      * +1.0   if the test imports from a module that contains a focus
               file (strongest signal — same module is being exercised)
      * +0.5   if the test references a symbol whose name appears in
               an issue keyword (e.g. issue says 'add()', test calls
               ``add``)
      * +0.3   if the test's path basename matches a focus file's
               module name (test_<module>.py for module.py)
      * +0.1   per fixture / parametrize decorator (signals project
               idioms worth copying)

    Returns [] cleanly on missing repo, no test files found, no focus
    files, or AST parse failures.
    """
    repo = Path(repo_path)
    if not repo.exists():
        return []
    focus = list(focus_files or [])
    if not focus:
        return []
    keywords = {str(k or "").strip().lower() for k in (issue_keywords or []) if k}

    # Build the focus-symbol set: module names from focus_files plus
    # any issue keywords that look like Python identifiers.
    # For __init__.py files, the meaningful module name is the PARENT
    # dir (mathlib/__init__.py → mathlib), not the stem (__init__).
    focus_modules: set[str] = set()
    focus_basenames: set[str] = set()
    for fp in focus:
        path = Path(fp)
        if path.name == "__init__.py":
            module_name = path.parent.name.lower()
        else:
            module_name = path.stem.lower()
        focus_modules.add(module_name)
        focus_basenames.add(module_name)

    candidates: list[TestExemplar] = []
    files_scanned = 0
    test_files: list[Path] = []
    for glob in _TEST_FILE_GLOBS:
        try:
            test_files.extend(repo.rglob(glob))
        except (OSError, PermissionError):
            continue
    # Dedupe and cap
    seen: set[str] = set()
    deduped: list[Path] = []
    for tf in test_files:
        s = str(tf)
        if s in seen:
            continue
        seen.add(s)
        deduped.append(tf)
    test_files = deduped[:max_files_scanned]

    for test_path in test_files:
        files_scanned += 1
        try:
            source = test_path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, ValueError):
            continue
        language = _language_for_test_path(test_path)
        rel_path = _safe_relative_path(test_path, repo)
        path_basename_score = (
            0.3 if _test_path_matches_focus_basename(test_path, focus_basenames) else 0.0
        )

        try:
            snippets = (
                _extract_python_test_snippets(source, max_lines=max_snippet_lines)
                if language == "python"
                else _extract_non_python_test_snippets(
                    source,
                    language=language,
                    max_lines=max_snippet_lines,
                )
            )
        except (SyntaxError, ValueError):
            continue

        for snippet, imports, target_symbols, idiom_count in snippets:
            if not snippet:
                continue

            # Compute similarity score
            score = path_basename_score
            reason_parts: list[str] = []
            if path_basename_score > 0:
                reason_parts.append(f"basename matches focus module {test_path.stem}")

            # Module-import overlap
            for imp in imports:
                imp_lower = imp.lower()
                for module in focus_modules:
                    if module and module in imp_lower:
                        score += 1.0
                        reason_parts.append(f"imports from {imp}")
                        break

            # Symbol-keyword overlap
            for sym in target_symbols:
                sym_lower = sym.lower()
                if sym_lower in keywords:
                    score += 0.5
                    reason_parts.append(f"calls {sym}")
                    break  # only credit once per test

            if idiom_count > 0:
                score += 0.1 * min(idiom_count, 3)
                reason_parts.append(f"{idiom_count} test idiom marker(s)")

            if score <= 0:
                continue
            candidates.append(
                TestExemplar(
                    path=rel_path,
                    snippet=snippet,
                    imports=imports,
                    target_symbols=target_symbols,
                    score=score,
                    reason="; ".join(reason_parts) or "structural similarity",
                    language=language,
                )
            )

    # Sort by score desc, then by path for determinism, take top_k
    candidates.sort(key=lambda c: (-c.score, c.path))
    return candidates[:top_k]


def _language_for_test_path(path: Path) -> str:
    suffixes = "".join(path.suffixes[-2:]).lower()
    suffix = path.suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix in {".js", ".jsx", ".mjs", ".cjs"}:
        return "javascript"
    if suffix in {".ts", ".tsx", ".mts", ".cts"}:
        return "typescript"
    if suffix == ".go":
        return "go"
    if suffix == ".java":
        return "java"
    if suffix == ".kt":
        return "kotlin"
    if suffix == ".php":
        return "php"
    if suffix == ".swift":
        return "swift"
    return suffixes.lstrip(".") or suffix.lstrip(".") or "text"


def _test_path_matches_focus_basename(
    test_path: Path,
    focus_basenames: set[str],
) -> bool:
    stem = test_path.stem.lower()
    candidates = {
        stem,
        stem.removeprefix("test_"),
        stem.removesuffix("_test"),
        stem.removesuffix("test"),
        stem.removesuffix("tests"),
        stem.removesuffix("spec"),
    }
    dotted_stem = stem
    for marker in (".test", ".spec"):
        if marker in dotted_stem:
            candidates.add(dotted_stem.split(marker, 1)[0])
    return any(candidate in focus_basenames for candidate in candidates if candidate)


def _extract_python_test_snippets(
    source: str,
    *,
    max_lines: int,
) -> list[tuple[str, list[str], list[str], int]]:
    tree = ast.parse(source)
    snippets: list[tuple[str, list[str], list[str], int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith("test_"):
            continue
        snippet, imports, target_symbols = _extract_test_function(
            tree=tree,
            func_node=node,
            source=source,
            max_lines=max_lines,
        )
        snippets.append((snippet, imports, target_symbols, len(list(node.decorator_list))))
    return snippets


def _extract_non_python_test_snippets(
    source: str,
    *,
    language: str,
    max_lines: int,
) -> list[tuple[str, list[str], list[str], int]]:
    imports = _extract_non_python_import_lines(source, language=language)
    block_starts = _non_python_test_block_starts(source, language=language)
    snippets: list[tuple[str, list[str], list[str], int]] = []
    for start in block_starts[:8]:
        snippet = _extract_balanced_block_snippet(source, start, max_lines=max_lines)
        if not snippet.strip():
            continue
        target_symbols = _extract_non_python_target_symbols(snippet, language=language)
        idiom_count = _count_non_python_test_idioms(snippet, language=language)
        snippets.append((snippet, imports, target_symbols, idiom_count))
    return snippets


def _extract_non_python_import_lines(source: str, *, language: str) -> list[str]:
    imports: list[str] = []
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if language in {"javascript", "typescript"} and (
            line.startswith("import ") or "require(" in line
        ):
            imports.append(line[:160])
        elif language == "go" and (line.startswith("import ") or line.startswith('"')):
            imports.append(line[:160])
        elif language in {"java", "kotlin"} and line.startswith("import "):
            imports.append(line[:160])
        elif language == "php" and (line.startswith("use ") or line.startswith("require")):
            imports.append(line[:160])
        elif language == "swift" and line.startswith("import "):
            imports.append(line[:160])
        if len(imports) >= 6:
            break
    return imports


def _non_python_test_block_starts(source: str, *, language: str) -> list[int]:
    if language in {"javascript", "typescript"}:
        pattern = re.compile(
            r"\b(?:describe|it|test)\s*(?:\.(?:only|skip))?\s*\("
            r"|\bo\s*\.\s*spec\s*\("
            r"|\bo\s*\(\s*['\"]"
        )
    elif language == "go":
        pattern = re.compile(r"\bfunc\s+Test[A-Za-z0-9_]*\s*\(")
    elif language in {"java", "kotlin"}:
        pattern = re.compile(r"@\s*Test\b|(?:public\s+)?void\s+test[A-Za-z0-9_]*\s*\(")
    elif language == "php":
        pattern = re.compile(r"\bfunction\s+test[A-Za-z0-9_]*\s*\(")
    elif language == "swift":
        pattern = re.compile(r"\bfunc\s+test[A-Za-z0-9_]*\s*\(")
    else:
        return []
    return [match.start() for match in pattern.finditer(source or "")]


def _extract_balanced_block_snippet(
    source: str,
    start: int,
    *,
    max_lines: int,
) -> str:
    lines = source[start:].splitlines()
    if not lines:
        return ""
    selected: list[str] = []
    brace_depth = 0
    paren_depth = 0
    seen_open = False
    for line in lines:
        selected.append(line.rstrip())
        for char in line:
            if char in "{(":
                seen_open = True
                if char == "{":
                    brace_depth += 1
                else:
                    paren_depth += 1
            elif char in "})":
                if char == "}" and brace_depth > 0:
                    brace_depth -= 1
                elif char == ")" and paren_depth > 0:
                    paren_depth -= 1
        if len(selected) >= max_lines:
            selected.append("// ... (truncated)")
            break
        if seen_open and brace_depth <= 0 and paren_depth <= 0 and len(selected) > 1:
            break
    return "\n".join(selected).strip()


def _extract_non_python_target_symbols(source: str, *, language: str) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", source or ""):
        name = match.group(1)
        if name in {
            "if",
            "for",
            "while",
            "switch",
            "function",
            "func",
            "describe",
            "it",
            "test",
            "spec",
            "o",
            "assert",
            "expect",
            "verify",
        }:
            continue
        if name not in seen:
            seen.add(name)
            symbols.append(name)
        if len(symbols) >= 8:
            break
    return symbols


def _count_non_python_test_idioms(source: str, *, language: str) -> int:
    patterns = [
        r"\bexpect\s*\(",
        r"\bassert(?:\.\w+)?\s*\(",
        r"\brequire\.\w+\s*\(",
        r"\bo\s*\(",
        r"\bt\.(?:Error|Fatal|Fail)",
        r"@\s*Test\b",
        r"\bXCTAssert\w*\s*\(",
    ]
    return sum(1 for pattern in patterns if re.search(pattern, source or ""))


def _extract_test_function(
    *,
    tree: ast.Module,
    func_node: ast.FunctionDef,
    source: str,
    max_lines: int,
) -> tuple[str, list[str], list[str]]:
    """Pull the source text, imports, and target-symbol references from
    one test function.

    Returns (snippet_or_empty, imports, target_symbols).
    """
    snippet = ast.get_source_segment(source, func_node) or ""
    if not snippet:
        return "", [], []
    if snippet.count("\n") > max_lines:
        # Truncate to first N lines + a marker so the agent knows it's clipped.
        lines = snippet.splitlines()
        snippet = "\n".join(lines[:max_lines]) + "\n    # ... (truncated)"

    # Module-level imports (file-wide)
    imports: list[str] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom):
            module = stmt.module or ""
            imports.append(f"from {module} import ...")
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                imports.append(f"import {alias.name}")
    imports = imports[:6]

    # Target symbols = identifiers called inside this test function
    target_symbols: list[str] = []
    seen: set[str] = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            target = _call_target_name(node.func)
            if target and target not in seen and not target.startswith("_"):
                seen.add(target)
                target_symbols.append(target)
    target_symbols = target_symbols[:8]

    return snippet, imports, target_symbols


def _call_target_name(node: ast.AST) -> Optional[str]:
    """Return the rightmost name of a call expression's callee."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _safe_relative_path(test_path: Path, repo: Path) -> str:
    try:
        return str(test_path.relative_to(repo))
    except ValueError:
        return str(test_path)


def render_exemplars_prompt_block(
    exemplars: list[TestExemplar],
    *,
    header: str = "Existing tests in this repository (use as style exemplars)",
) -> str:
    """Render exemplars as a Markdown prompt section.

    Returns "" when no exemplars are available so the prompt doesn't
    carry an empty section.
    """
    if not exemplars:
        return ""
    lines = [
        f"## {header}",
        "",
        (
            "These tests already exist in the repo and use the project's "
            "idioms (import paths, fixture style, assertion shape). Copy "
            "the SHAPE of these patterns when writing your new tests — "
            "do NOT copy their assertions blindly. Each one is annotated "
            "with WHY it was selected as similar to the bug surface."
        ),
        "",
    ]
    for ex in exemplars:
        lines.append(f"### `{ex.path}` (score {ex.score:.2f})")
        lines.append(f"_Selected because: {ex.reason}_")
        if ex.imports:
            lines.append("Module-level imports: " + ", ".join(ex.imports[:3]))
        language = str(ex.language or "").strip().lower()
        fence_language = {
            "javascript": "javascript",
            "typescript": "typescript",
            "go": "go",
            "java": "java",
            "kotlin": "kotlin",
            "php": "php",
            "swift": "swift",
        }.get(language, "python")
        lines.append(f"```{fence_language}")
        lines.append(ex.snippet)
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
