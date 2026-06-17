"""Post-splice validation helpers for generated test artifacts.

Benchmark harnesses often do not execute the raw model artifact exactly as
emitted. They splice it into an existing test file, append it to a checkout, or
apply it as a patch first. Static gates should inspect that post-splice shape.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from .test_style import TestStyleProfile


class SpliceMode(str, Enum):
    REPLACE = "replace"
    APPEND = "append"
    NEW_FILE = "new_file"
    PATCH = "patch"


class SpliceSimulator(Protocol):
    splice_mode: SpliceMode

    def synthesize_post_splice(
        self,
        *,
        original_test_source: str,
        artifact_text: str,
        style: TestStyleProfile,
    ) -> str:
        """Return the source text that the benchmark harness will validate."""


@dataclass(frozen=True)
class TestGenEvalSpliceSimulator:
    """Best-effort mirror of TestGenEval's generated-test insertion shape."""

    separator: str = "\n\n# Apex generated tests\n\n"
    splice_mode: SpliceMode = SpliceMode.REPLACE
    __test__ = False

    def synthesize_post_splice(
        self,
        *,
        original_test_source: str,
        artifact_text: str,
        style: TestStyleProfile,
    ) -> str:
        if self.splice_mode == SpliceMode.REPLACE:
            artifact = str(artifact_text or "").strip()
            return artifact + ("\n" if artifact else "")
        original = str(original_test_source or "").rstrip()
        artifact = str(artifact_text or "").strip()
        if not original:
            return artifact + ("\n" if artifact else "")
        if not artifact:
            return original + "\n"
        return original + self.separator + artifact + "\n"


@dataclass(frozen=True)
class AppendSpliceSimulator(TestGenEvalSpliceSimulator):
    """Append-mode simulator for benchmarks that extend an existing test file."""

    splice_mode: SpliceMode = SpliceMode.APPEND


@dataclass(frozen=True)
class SpliceInvariantReport:
    parse_ok: bool = True
    diagnostic: str = ""
    duplicate_symbols: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.parse_ok and not self.duplicate_symbols

    def to_dict(self) -> dict[str, object]:
        return {
            "parse_ok": self.parse_ok,
            "diagnostic": self.diagnostic,
            "duplicate_symbols": list(self.duplicate_symbols),
        }


def validate_splice_invariants(
    *,
    original_test_source: str,
    artifact_text: str,
    post_splice_source: str,
    style: TestStyleProfile,
    splice_mode: SpliceMode,
) -> SpliceInvariantReport:
    """Validate cheap invariants on the synthesized post-splice file."""

    language = (style.language or "").lower()
    if language not in {"python", "py", "python3"}:
        return SpliceInvariantReport()
    try:
        ast.parse(post_splice_source or "")
    except SyntaxError as exc:
        return SpliceInvariantReport(
            parse_ok=False,
            diagnostic=f"post-splice SyntaxError: {exc}",
        )
    if splice_mode == SpliceMode.APPEND:
        duplicates = _python_duplicate_top_level_symbols(
            original_test_source=original_test_source,
            artifact_text=artifact_text,
        )
        if duplicates:
            return SpliceInvariantReport(
                duplicate_symbols=duplicates,
                diagnostic="duplicate top-level symbol(s) after splice: " + ", ".join(duplicates),
            )
    return SpliceInvariantReport()


def _python_duplicate_top_level_symbols(
    *,
    original_test_source: str,
    artifact_text: str,
) -> list[str]:
    original = _python_top_level_symbols(original_test_source)
    generated = _python_top_level_symbols(artifact_text)
    duplicates = sorted(original & generated)
    # Baseline-name preservation intentionally keeps some test_* selectors for
    # harness filters. Treat duplicate test functions as valid anchors; still
    # reject helper/class collisions that can change import-time behavior.
    return [name for name in duplicates if not name.startswith("test_")]


def _python_top_level_symbols(source: str) -> set[str]:
    try:
        tree = ast.parse(source or "")
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = list(getattr(node, "targets", [])) or [getattr(node, "target", None)]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names
