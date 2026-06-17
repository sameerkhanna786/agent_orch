"""Public-symbol survival check.

After the agent submits a patch, walk the AST of every edited Python
file in (a) the baseline state and (b) the candidate state, and report
top-level public symbols (def / class / module-level assignment) that
existed in baseline but disappeared in the candidate.

Public means: not leading-underscore, not dunder. The check helps catch
the most common form of test-collection breakage — the agent removes a
function the tests / conftest import. Non-Python files are skipped here
because accurate cross-language symbol tracking would require
tree-sitter; the language-agnostic safety net (stub scanner +
infrastructure gate) covers the same failure mode at coarser grain.
"""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class SymbolLoss:
    path: str
    symbol: str
    kind: str  # "function" | "class" | "assignment"


def _public_top_level_symbols(source: str) -> set[tuple[str, str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    symbols: set[tuple[str, str]] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                symbols.add((node.name, "function"))
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                symbols.add((node.name, "class"))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    symbols.add((target.id, "assignment"))
    return symbols


def _read_baseline_source(
    workspace: Path,
    rel_path: str,
    *,
    baseline_ref: str = "apex-base",
) -> Optional[str]:
    """Fetch the baseline-state contents of ``rel_path`` from git.

    Falls back to ``HEAD`` when the canonical apex-base ref isn't
    present (e.g. test fixtures that don't set it up).
    """
    for ref in (baseline_ref, "HEAD"):
        try:
            result = subprocess.run(
                ["git", "show", f"{ref}:{rel_path}"],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        if result.returncode == 0:
            return result.stdout
    return None


_TEST_PATH_MARKERS = (
    "/tests/",
    "/test/",
    "/__tests__/",
    "/spec/",
)
_TEST_FILE_SUFFIX_MARKERS = (
    "_test.py",
    "_test.go",
    "_spec.rb",
    ".test.js",
    ".test.jsx",
    ".test.ts",
    ".test.tsx",
    ".spec.js",
    ".spec.jsx",
    ".spec.ts",
    ".spec.tsx",
    "Test.java",  # Conventional JUnit test class suffix
)


def _is_test_path(rel: str) -> bool:
    rel_lower = rel.lower()
    if rel_lower.startswith(("tests/", "test/", "__tests__/", "spec/")):
        return True
    if any(marker in rel_lower for marker in _TEST_PATH_MARKERS):
        return True
    base = rel.rsplit("/", 1)[-1]
    if base.startswith("test_") or base.startswith("Test"):
        return True
    if any(rel.endswith(s) for s in _TEST_FILE_SUFFIX_MARKERS):
        return True
    return False


def _symbols_for_source(source: str, suffix: str) -> set[tuple[str, str]]:
    """Pick the right AST/parser for the language and return public symbols."""
    if suffix == ".py":
        return _public_top_level_symbols(source)
    try:
        from . import tree_sitter_helper
    except ImportError:
        return set()
    if not tree_sitter_helper.is_available():
        return set()
    language = tree_sitter_helper.language_for_suffix(suffix)
    if language is None:
        return set()
    return tree_sitter_helper.top_level_public_symbols(source, language)


def detect_public_symbol_losses(
    workspace: Path,
    changed_files: Iterable[str],
    *,
    baseline_ref: str = "apex-base",
    max_findings: int = 25,
) -> list[SymbolLoss]:
    """Compare baseline vs current public symbols across edited files.

    Python uses the stdlib ``ast`` module (always available); JS / TS /
    Go / Rust / Java / Ruby use tree-sitter when installed and silently
    fall back to "no losses detected" otherwise. Test files are skipped
    by language-aware path heuristics (``tests/``, ``__tests__/``,
    ``*_test.go``, ``*Test.java``, etc.).
    """
    workspace = Path(workspace)
    losses: list[SymbolLoss] = []
    for rel in changed_files:
        suffix = ""
        if "." in rel.rsplit("/", 1)[-1]:
            suffix = "." + rel.rsplit(".", 1)[-1].lower()
        if not suffix:
            continue
        # Only process languages we can actually parse.
        if suffix != ".py":
            try:
                from . import tree_sitter_helper

                if (
                    not tree_sitter_helper.is_available()
                    or tree_sitter_helper.language_for_suffix(suffix) is None
                ):
                    continue
            except ImportError:
                continue
        if _is_test_path(rel):
            continue
        path = workspace / rel
        if not path.exists() or not path.is_file():
            # File was deleted entirely — that's a much louder loss; flag
            # the whole module so the followup knows.
            baseline_source = _read_baseline_source(workspace, rel, baseline_ref=baseline_ref)
            if baseline_source is not None:
                for symbol, kind in _symbols_for_source(baseline_source, suffix):
                    losses.append(SymbolLoss(rel, symbol, kind))
                    if len(losses) >= max_findings:
                        return losses
            continue
        try:
            current_source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        baseline_source = _read_baseline_source(workspace, rel, baseline_ref=baseline_ref)
        if baseline_source is None:
            # New file: nothing was lost.
            continue
        baseline_symbols = _symbols_for_source(baseline_source, suffix)
        current_symbols = _symbols_for_source(current_source, suffix)
        for symbol, kind in sorted(baseline_symbols - current_symbols):
            losses.append(SymbolLoss(rel, symbol, kind))
            if len(losses) >= max_findings:
                return losses
    return losses


def summarize_losses(losses: list[SymbolLoss], *, max_lines: int = 10) -> str:
    if not losses:
        return ""
    lines = ["Baseline public symbols missing from the candidate patch:"]
    for loss in losses[:max_lines]:
        lines.append(f"  - {loss.path}::{loss.symbol} ({loss.kind})")
    if len(losses) > max_lines:
        lines.append(f"  ... and {len(losses) - max_lines} more")
    return "\n".join(lines)
