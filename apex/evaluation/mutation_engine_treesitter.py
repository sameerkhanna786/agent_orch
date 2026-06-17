"""Tree-sitter-backed mutation engine for non-Python languages.

Phase I.5 — multi-language mutation. The Python AST-backed
mutator in ``mutation_engine`` only handles ``.py`` sources. This
module provides an equivalent generator powered by tree-sitter,
so JS / TS / Go / Java / Rust / etc. mutation scoring can drive
the same in-loop sensitivity feedback the Python path enjoys.

Core idea: tree-sitter gives us a CST with byte ranges per node;
we walk the tree, identify mutation sites (binary operators,
boolean / number literals, return statements, conditional
predicates), and emit byte-level edits. The single ``Mutant``
shape is reused so downstream code (mutation_engine.evaluate_*,
test_minimizer, iteration_feedback.derive_mutation_sensitivity)
sees one uniform record type regardless of language.

Supported languages (anything tree-sitter-languages exposes):
    javascript, typescript, tsx, go, rust, java, kotlin, swift,
    csharp, ruby, php, scala, c, cpp.

The same operator vocabulary works across all of them because
the binary-operator tokens (``<``, ``<=``, ``==``, ``!=``, ``+``,
``-``, ``*``, ``/``) are textually identical. Boolean literals
differ by language (``true`` / ``false`` vs ``True`` / ``False``)
so we detect both shapes.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Language name normalization
# ---------------------------------------------------------------------------

_TREESITTER_LANGUAGE_ALIASES: dict[str, str] = {
    "js": "javascript",
    "javascript": "javascript",
    "ts": "typescript",
    "typescript": "typescript",
    "tsx": "tsx",
    "go": "go",
    "golang": "go",
    "rs": "rust",
    "rust": "rust",
    "java": "java",
    "kt": "kotlin",
    "kotlin": "kotlin",
    "swift": "swift",
    "cs": "csharp",
    "csharp": "csharp",
    "rb": "ruby",
    "ruby": "ruby",
    "php": "php",
    "scala": "scala",
    "c": "c",
    "cpp": "cpp",
    "cc": "cpp",
    "h": "c",
    "hpp": "cpp",
}


# Operator-swap rules. Same vocabulary works across the supported
# languages because the literal operator tokens are identical.
_OPERATOR_SWAPS: dict[str, str] = {
    "<": "<=",
    "<=": "<",
    ">": ">=",
    ">=": ">",
    "==": "!=",
    "!=": "==",
    "+": "-",
    "-": "+",
    "*": "/",
    "/": "*",
}


# Boolean literal flips. Works across {true,false} (JS/Go/Rust/...)
# and {True,False} (Python — handled here too for tree-sitter Python).
_BOOLEAN_FLIPS: dict[str, str] = {
    "true": "false",
    "false": "true",
    "True": "False",
    "False": "True",
}


# Cap on bytes a single edit can touch. Keeps mutations localized
# (we never want to delete the body of a function via the
# statement-deletion operator on a multi-line statement).
_MAX_EDIT_BYTES = 64


def _resolve_treesitter_language(language: str) -> Optional[str]:
    return _TREESITTER_LANGUAGE_ALIASES.get((language or "").lower())


def _treesitter_parser(language: str) -> Optional[Any]:
    """Lazy-load a tree-sitter parser; returns None if tree-sitter is
    not installed or the language is unsupported."""
    normalized = _resolve_treesitter_language(language)
    if normalized is None:
        return None
    try:
        from tree_sitter_languages import get_parser  # type: ignore
    except Exception:  # pragma: no cover — defensive
        logger.debug("tree_sitter_languages not installed")
        return None
    try:
        return get_parser(normalized)
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("get_parser(%s) failed: %s", normalized, exc)
        return None


# ---------------------------------------------------------------------------
# Mutation generation
# ---------------------------------------------------------------------------


def _byte_replace(source_bytes: bytes, start: int, end: int, repl: bytes) -> bytes:
    return source_bytes[:start] + repl + source_bytes[end:]


def _make_mutant(
    *,
    operator: str,
    source_path: str,
    source_text: str,
    mutated_source_text: str,
    line: int,
    col: int,
    original_snippet: str,
    mutated_snippet: str,
) -> Any:
    """Construct a Mutant in the shape mutation_engine expects.

    Imports ``Mutant`` lazily so this module doesn't have an
    import-time dependency on the AST-backed engine (avoids cycles
    when callers wire dispatch).
    """
    from .mutation_engine import Mutant

    return Mutant(
        operator=operator,
        source_path=source_path,
        line=line,
        col=col,
        original_snippet=original_snippet,
        mutated_snippet=mutated_snippet,
        mutated_source=mutated_source_text,
    )


def _walk(node: Any) -> Any:
    """Pre-order walk over a tree-sitter CST."""
    yield node
    for child in node.children:
        yield from _walk(child)


def _operator_mutants_for_node(
    node: Any, source_bytes: bytes, source_text: str, source_path: str
) -> list[Any]:
    """If ``node`` is an operator token whose text is one of the
    swap targets, emit one Mutant for the swap.

    Tree-sitter exposes operators as anonymous nodes whose ``type``
    equals the literal token text (e.g., ``<``, ``<=``, ``==``).
    """
    out: list[Any] = []
    if node.child_count != 0:
        return out
    if not node.is_named:
        token = node.type or ""
        if token in _OPERATOR_SWAPS:
            replacement = _OPERATOR_SWAPS[token]
            mutated_bytes = _byte_replace(
                source_bytes, node.start_byte, node.end_byte, replacement.encode("utf-8")
            )
            try:
                mutated_text = mutated_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return out
            line = node.start_point[0] + 1  # tree-sitter is 0-indexed
            col = node.start_point[1]
            family = "boundary" if token in {"<", "<=", ">", ">=", "==", "!="} else "arith"
            operator_name = f"{family}_{token}_to_{replacement}"
            out.append(
                _make_mutant(
                    operator=operator_name,
                    source_path=source_path,
                    source_text=source_text,
                    mutated_source_text=mutated_text,
                    line=line,
                    col=col,
                    original_snippet=token,
                    mutated_snippet=replacement,
                )
            )
    return out


def _boolean_mutants_for_node(
    node: Any, source_bytes: bytes, source_text: str, source_path: str
) -> list[Any]:
    out: list[Any] = []
    if node.child_count != 0:
        return out
    text = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")
    if text in _BOOLEAN_FLIPS:
        replacement = _BOOLEAN_FLIPS[text]
        mutated_bytes = _byte_replace(
            source_bytes, node.start_byte, node.end_byte, replacement.encode("utf-8")
        )
        try:
            mutated_text = mutated_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return out
        line = node.start_point[0] + 1
        col = node.start_point[1]
        out.append(
            _make_mutant(
                operator=f"constant_{text}_to_{replacement}",
                source_path=source_path,
                source_text=source_text,
                mutated_source_text=mutated_text,
                line=line,
                col=col,
                original_snippet=text,
                mutated_snippet=replacement,
            )
        )
    return out


def _number_mutants_for_node(
    node: Any, source_bytes: bytes, source_text: str, source_path: str
) -> list[Any]:
    """Mutate integer literals: n → n+1, n → n-1, n → 0 (when n != 0).

    Tree-sitter's number node type varies by language — the most
    common names are ``number``, ``integer_literal``, ``int_literal``,
    ``float_literal``. We accept all and require that the body parses
    as an int (otherwise skip).
    """
    out: list[Any] = []
    if node.type not in {
        "number",
        "integer_literal",
        "int_literal",
        "decimal_integer_literal",
    }:
        return out
    text = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore").strip()
    try:
        value = int(text)
    except ValueError:
        return out
    line = node.start_point[0] + 1
    col = node.start_point[1]
    targets: list[tuple[str, str]] = []
    targets.append(("plus_one", str(value + 1)))
    targets.append(("minus_one", str(value - 1)))
    if value != 0:
        targets.append(("to_zero", "0"))
    for tag, replacement in targets:
        mutated_bytes = _byte_replace(
            source_bytes, node.start_byte, node.end_byte, replacement.encode("utf-8")
        )
        try:
            mutated_text = mutated_bytes.decode("utf-8")
        except UnicodeDecodeError:
            continue
        out.append(
            _make_mutant(
                operator=f"constant_replacement_{tag}",
                source_path=source_path,
                source_text=source_text,
                mutated_source_text=mutated_text,
                line=line,
                col=col,
                original_snippet=text,
                mutated_snippet=replacement,
            )
        )
    return out


def _return_value_mutants_for_node(
    node: Any, source_bytes: bytes, source_text: str, source_path: str
) -> list[Any]:
    """For a ``return_statement`` whose value is a single named
    expression, replace it with the language's null token so the
    function returns nothing meaningful.

    Mutations large enough to span a multi-line return body are
    skipped via _MAX_EDIT_BYTES so we don't accidentally delete a
    full block.
    """
    out: list[Any] = []
    if node.type != "return_statement":
        return out
    value_children = [c for c in node.children if c.is_named]
    if not value_children:
        return out
    value = value_children[-1]  # last named child = expression
    span = value.end_byte - value.start_byte
    if span <= 0 or span > _MAX_EDIT_BYTES:
        return out
    # Pick a null token suitable for any of our supported languages.
    # `null` works in JS / Go / Java / etc.; Rust uses None which we
    # don't want here since it requires Option<>; skip Rust.
    null_token = "null"
    mutated_bytes = _byte_replace(
        source_bytes, value.start_byte, value.end_byte, null_token.encode("utf-8")
    )
    try:
        mutated_text = mutated_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return out
    line = value.start_point[0] + 1
    col = value.start_point[1]
    original_snippet = source_bytes[value.start_byte : value.end_byte].decode(
        "utf-8", errors="ignore"
    )
    out.append(
        _make_mutant(
            operator="return_value_to_null",
            source_path=source_path,
            source_text=source_text,
            mutated_source_text=mutated_text,
            line=line,
            col=col,
            original_snippet=original_snippet[:40],
            mutated_snippet=null_token,
        )
    )
    return out


def generate_mutants_treesitter(
    *,
    source_path: str | Path,
    source_text: Optional[str] = None,
    language: str,
    max_mutants: int = 32,
    seed: int = 0,
) -> list[Any]:
    """Generate mutants for a non-Python source via tree-sitter.

    Returns ``[]`` when:
        * tree-sitter is not installed,
        * the language is unsupported,
        * the source can't be read or parsed.

    Caller-side dispatch in ``mutation_engine.generate_mutants``
    routes Python sources to the AST-backed engine and everything
    else to this function.
    """
    parser = _treesitter_parser(language)
    if parser is None:
        return []
    src_path = Path(source_path)
    if source_text is None:
        try:
            source_text = src_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []
    source_bytes = source_text.encode("utf-8")
    try:
        tree = parser.parse(source_bytes)
    except Exception as exc:  # pragma: no cover — defensive
        # Audit H12: log so a tree-sitter failure is visible to the
        # operator instead of silently producing zero mutants — that
        # latter case is indistinguishable from "no mutation candidates
        # in this file" and distorts the mutation score.
        logger = __import__("logging").getLogger(__name__)
        logger.warning(
            "mutation_engine_treesitter: tree-sitter parse failed for %r "
            "(%s: %s); 0 mutants produced",
            str(source_path),
            type(exc).__name__,
            exc,
        )
        return []
    repo_relative_path = str(source_path)

    candidates: list[Any] = []
    for node in _walk(tree.root_node):
        candidates.extend(
            _operator_mutants_for_node(node, source_bytes, source_text, repo_relative_path)
        )
        candidates.extend(
            _boolean_mutants_for_node(node, source_bytes, source_text, repo_relative_path)
        )
        candidates.extend(
            _number_mutants_for_node(node, source_bytes, source_text, repo_relative_path)
        )
        candidates.extend(
            _return_value_mutants_for_node(node, source_bytes, source_text, repo_relative_path)
        )

    # Deduplicate by (operator, line, col, mutated_snippet) — same
    # rule as the Python engine.
    seen: set[tuple[str, int, int, str]] = set()
    unique: list[Any] = []
    for m in candidates:
        key = (m.operator, m.line, m.col, m.mutated_snippet)
        if key in seen:
            continue
        seen.add(key)
        unique.append(m)

    if len(unique) <= max_mutants:
        return unique
    rng = random.Random(seed)
    sampled = rng.sample(unique, max_mutants)
    sampled.sort(key=lambda m: (m.line, m.col, m.operator))
    return sampled
