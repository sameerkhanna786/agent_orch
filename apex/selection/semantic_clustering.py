"""Decisive-Edge B.3 — semantic patch clustering.

The legacy clustering layer (``apex/selection/selector.py``) bucketed
candidates by exact AST equality (the "fingerprint" pass) and then ran
agglomerative single-linkage clustering using a Jaccard-style mix of
file-set + normalized-line + payload string similarity. That mix
collapsed semantically very different patches that happened to touch
the same files into the same cluster — e.g. three patches that all edit
``pkg/foo.py`` but flip the comparison operator from ``>`` to ``<`` to
``==`` would cluster as one (high file_similarity, high line
similarity from the surrounding context), even though only one of them
can be the true fix.

This module exposes a *semantic* signature for each patch and a
weighted distance between two signatures. The selector still uses the
exact-AST fingerprint as its first pass (so semantically identical
patches always cluster together), but the second pass — single-linkage
agglomerative clustering — now operates on ``semantic_distance``
rather than the legacy text similarity, with the legacy text-similarity
preserved as a final fallback when AST parsing fails on every file in
the patch.

Public API:

  * :class:`SemanticSignature` — frozen dataclass capturing the
    semantic delta of a patch.
  * :func:`compute_semantic_signature` — extract the signature from
    a unified-diff blob.
  * :func:`semantic_distance` — weighted distance ∈ [0, 1] between
    two signatures (0 = semantically identical; 1 = completely
    different).
  * :func:`semantic_similarity` — convenience: ``1.0 - semantic_distance``.

The implementation is pure stdlib (``ast``, ``re``, ``difflib``,
``dataclasses``, ``pathlib``) — no ``tree-sitter`` dependency.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

logger = logging.getLogger("apex.selection.semantic_clustering")


# ---------------------------------------------------------------------------
# Signature dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticSignature:
    """Structured semantic delta of a patch.

    All tuple fields are sorted + de-duplicated so two signatures with
    the same conceptual contents compare equal regardless of input
    order.
    """

    changed_function_names: Tuple[str, ...] = field(default_factory=tuple)
    """Functions whose body was edited (added, removed, or modified)."""

    changed_call_sites: Tuple[str, ...] = field(default_factory=tuple)
    """Call expressions added/removed/modified by the patch (callee
    names; e.g. ``foo``, ``self.bar``, ``module.baz``)."""

    added_imports: Tuple[str, ...] = field(default_factory=tuple)
    """Imports introduced by the patch (``import foo`` →  ``foo``;
    ``from foo import bar`` → ``foo.bar``)."""

    removed_imports: Tuple[str, ...] = field(default_factory=tuple)
    """Imports removed by the patch."""

    modified_control_flow: Tuple[str, ...] = field(default_factory=tuple)
    """Control-flow node *types* that appear in the edit region (e.g.
    ``If``, ``For``, ``While``, ``Try``, ``With``, ``Match``)."""

    modified_data_structures: Tuple[str, ...] = field(default_factory=tuple)
    """Data-structure node *types* that appear in the edit region (e.g.
    ``ClassDef``, ``Dict``, ``List``, ``Set``, ``Tuple``)."""

    file_set_normalized: Tuple[str, ...] = field(default_factory=tuple)
    """Canonicalized file paths touched by the patch."""

    operator_kinds: Tuple[str, ...] = field(default_factory=tuple)
    """Comparison / boolean / binary / unary operator node kinds present
    in the edit region (e.g. ``Lt``, ``Gt``, ``Eq``, ``And``, ``Add``).
    Critical for distinguishing patches that differ only in the
    operator they use (``x > 0`` vs ``x < 0`` vs ``x == 0``)."""

    constant_signature: Tuple[str, ...] = field(default_factory=tuple)
    """Stringified constant literals appearing in the edit region.
    De-duplicated and length-bounded so the comparison stays cheap;
    used by :func:`semantic_distance` only as a tiebreaker for
    operator-flip and constant-flip cases."""

    # Diagnostics — not part of the equality / distance computation.
    parse_failed_files: Tuple[str, ...] = field(default_factory=tuple)
    non_python_files: Tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Diff parsing
# ---------------------------------------------------------------------------


_FILE_HEADER_RE = re.compile(r"^diff --git a/(?P<a>\S+) b/(?P<b>\S+)")
_OLD_FILE_RE = re.compile(r"^---\s+(?:a/)?(?P<path>\S+)")
_NEW_FILE_RE = re.compile(r"^\+\+\+\s+(?:b/)?(?P<path>\S+)")


def _normalize_path(raw: str) -> str:
    """Normalize a diff-supplied path.

    Strips leading ``a/`` / ``b/`` markers, leading ``./``, and a few
    common variants of ``/dev/null``. Returns empty string for paths
    that should be ignored.
    """
    text = (raw or "").strip()
    if not text or text in {"/dev/null", "dev/null"}:
        return ""
    if text.startswith("a/") or text.startswith("b/"):
        text = text[2:]
    if text.startswith("./"):
        text = text[2:]
    return text


@dataclass
class _PerFileDiff:
    """Internal: parsed view of one file's hunks within a unified diff."""

    path: str
    added_lines: list[str] = field(default_factory=list)
    removed_lines: list[str] = field(default_factory=list)
    is_python: bool = False
    parse_failed: bool = False


def _parse_unified_diff(patch_text: str) -> list[_PerFileDiff]:
    """Split a unified diff into per-file added/removed line buckets.

    Robust to git-format diffs (``diff --git a/... b/...``), plain
    ``--- a/x``/``+++ b/x`` headers, and binary-file markers (skipped).
    """
    if not patch_text:
        return []
    files: list[_PerFileDiff] = []
    current: Optional[_PerFileDiff] = None

    def _flush() -> None:
        nonlocal current
        if current is not None and current.path:
            files.append(current)
        current = None

    for raw_line in patch_text.splitlines():
        line = raw_line.rstrip("\r")
        if line.startswith("diff --git"):
            _flush()
            match = _FILE_HEADER_RE.match(line)
            if match:
                path = _normalize_path(match.group("b") or match.group("a"))
                current = _PerFileDiff(path=path)
            else:
                current = _PerFileDiff(path="")
            continue
        if line.startswith("--- "):
            # ``---`` arrives before ``+++`` for non-git diffs (no
            # ``diff --git`` line). If we don't have a current entry,
            # start one with the old path; ``+++`` will overwrite.
            if current is None:
                current = _PerFileDiff(path="")
            old_match = _OLD_FILE_RE.match(line)
            if old_match and not current.path:
                current.path = _normalize_path(old_match.group("path"))
            continue
        if line.startswith("+++ "):
            if current is None:
                current = _PerFileDiff(path="")
            new_match = _NEW_FILE_RE.match(line)
            if new_match:
                new_path = _normalize_path(new_match.group("path"))
                if new_path:
                    current.path = new_path
            continue
        if line.startswith("Binary files"):
            # Skip binary-file diffs entirely.
            if current is not None:
                current.parse_failed = True
            continue
        if current is None:
            continue
        # Ignore hunk headers; collect added/removed body lines.
        if line.startswith("@@"):
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            current.added_lines.append(line[1:])
        elif line.startswith("-"):
            current.removed_lines.append(line[1:])
        # Context lines are deliberately discarded — semantic deltas
        # only care about adds/removes.
    _flush()

    # Annotate language.
    for entry in files:
        entry.is_python = entry.path.endswith(".py")
    return files


# ---------------------------------------------------------------------------
# AST extraction
# ---------------------------------------------------------------------------


def _safe_parse(snippet: str) -> Optional[ast.AST]:
    """Best-effort parse of a code snippet.

    Diff-fragment parsing is fundamentally lossy — added/removed lines
    rarely form a syntactically complete file by themselves. We try
    multiple wrappers (function body, class body, raw module) before
    giving up. Returns ``None`` when nothing parses.
    """
    candidates = [
        snippet,
        # Wrap in a fake function so leading-indent fragments parse.
        "def __apex_wrap__():\n" + "\n".join("    " + line for line in snippet.splitlines()),
        # Wrap inside a class for class-body fragments.
        "class __ApexWrap__:\n" + "\n".join("    " + line for line in snippet.splitlines()),
    ]
    for candidate in candidates:
        try:
            return ast.parse(candidate)
        except SyntaxError:
            continue
        except Exception:  # noqa: BLE001 — defensive; ast.parse can raise odd things
            continue
    return None


def _collect_call_names(tree: ast.AST) -> set[str]:
    """Extract callee names from every ``ast.Call`` in a tree."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            try:
                names.add(_attr_chain(func))
            except Exception:
                names.add(func.attr)
        elif isinstance(func, ast.Call):
            # Curried/chained calls — record the inner attribute / name.
            inner = func.func
            if isinstance(inner, ast.Name):
                names.add(inner.id)
            elif isinstance(inner, ast.Attribute):
                names.add(inner.attr)
    return names


def _attr_chain(node: ast.Attribute) -> str:
    """Render an attribute chain like ``a.b.c`` from its AST."""
    parts: list[str] = []
    current: Any = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    elif isinstance(current, ast.Call):
        inner = current.func
        if isinstance(inner, ast.Name):
            parts.append(inner.id)
        elif isinstance(inner, ast.Attribute):
            parts.append(inner.attr)
    return ".".join(reversed(parts))


def _collect_function_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def _collect_imports(tree: ast.AST) -> set[str]:
    """Extract import targets, normalizing to ``module.name`` form."""
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if module:
                    imports.add(f"{module}.{alias.name}")
                else:
                    imports.add(alias.name)
    return imports


_CONTROL_FLOW_TYPES: tuple[type, ...] = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.Raise,
    ast.Return,
    ast.Yield,
    ast.YieldFrom,
    ast.Break,
    ast.Continue,
)
# Match (Python 3.10+); guard with getattr so older interpreters skip it.
_MATCH_NODE = getattr(ast, "Match", None)
if _MATCH_NODE is not None:
    _CONTROL_FLOW_TYPES = _CONTROL_FLOW_TYPES + (_MATCH_NODE,)

_DATA_STRUCTURE_TYPES: tuple[type, ...] = (
    ast.ClassDef,
    ast.Dict,
    ast.List,
    ast.Set,
    ast.Tuple,
    ast.DictComp,
    ast.ListComp,
    ast.SetComp,
    ast.GeneratorExp,
)


def _collect_node_kinds(tree: ast.AST, node_types: Iterable[type]) -> set[str]:
    types_tuple = tuple(node_types)
    kinds: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, types_tuple):
            kinds.add(type(node).__name__)
    return kinds


def _collect_operator_kinds(tree: ast.AST) -> set[str]:
    """Extract comparison / boolean / binary / unary operator node kinds.

    Critical for distinguishing operator-flip patches: ``x > 0`` vs
    ``x < 0`` vs ``x == 0`` look identical at the function-name /
    call-site level but differ here.
    """
    kinds: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for op in node.ops:
                kinds.add(type(op).__name__)
        elif isinstance(node, ast.BoolOp):
            kinds.add(type(node.op).__name__)
        elif isinstance(node, ast.BinOp):
            kinds.add(type(node.op).__name__)
        elif isinstance(node, ast.UnaryOp):
            kinds.add(type(node.op).__name__)
        elif isinstance(node, ast.AugAssign):
            kinds.add("Aug:" + type(node.op).__name__)
    return kinds


_MAX_CONSTANT_REPR_LEN = 32


def _collect_constants(tree: ast.AST) -> set[str]:
    """Stringify constant literals (numbers, strings, None, True/False)
    appearing in a tree, length-bounded so giant string literals don't
    blow up the signature."""
    constants: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        value = node.value
        try:
            rendered = repr(value)
        except Exception:  # noqa: BLE001 - defensive
            continue
        if len(rendered) > _MAX_CONSTANT_REPR_LEN:
            rendered = rendered[:_MAX_CONSTANT_REPR_LEN] + "..."
        constants.add(rendered)
    return constants


# ---------------------------------------------------------------------------
# Non-Python heuristic extraction
# ---------------------------------------------------------------------------


# Function/method definition patterns common to JS / Go / Java / Rust.
# Each yields the function name in group(1).
_NON_PY_FUNC_PATTERNS: tuple[re.Pattern[str], ...] = (
    # JS / TS: ``function foo(`` and ``foo = function(``
    re.compile(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\("),
    re.compile(r"\b([A-Za-z_$][\w$]*)\s*=\s*function\b"),
    # Arrow function: ``const foo = (`` / ``let foo = (``
    re.compile(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\("),
    # Go: ``func foo(`` and ``func (r Recv) foo(``
    re.compile(r"\bfunc\s+(?:\([^)]*\)\s+)?([A-Za-z_][\w]*)\s*\("),
    # Java/C#-style: ``public void foo(`` (best-effort)
    re.compile(
        r"\b(?:public|private|protected|static|final|virtual|override)\s+[A-Za-z_<>\[\],\s]+?\s+([A-Za-z_][\w]*)\s*\("
    ),
    # Rust: ``fn foo(`` and ``pub fn foo(``
    re.compile(r"\bfn\s+([A-Za-z_][\w]*)\s*\("),
)
_NON_PY_CALL_PATTERN = re.compile(r"\b([A-Za-z_$][\w$]*)\s*\(")
_NON_PY_IMPORT_PATTERNS: tuple[re.Pattern[str], ...] = (
    # JS: ``import foo from 'bar'`` / ``import { x } from 'bar'``
    re.compile(r"\bimport\b[^;'\"\n]*from\s+['\"]([^'\"]+)['\"]"),
    # JS: ``require('bar')``
    re.compile(r"\brequire\(\s*['\"]([^'\"]+)['\"]\s*\)"),
    # Go: ``import \"bar\"`` (single)
    re.compile(r"\bimport\s+['\"]([^'\"]+)['\"]"),
    # Java: ``import foo.bar.Baz;``
    re.compile(r"\bimport\s+(?:static\s+)?([\w.$]+)\s*;"),
)


def _extract_non_python_signature_pieces(
    snippet: str,
) -> dict[str, set[str]]:
    """Best-effort heuristic extraction for non-Python languages.

    Returns a dict with keys matching the signature field names so the
    caller can merge them with the Python AST results.
    """
    if not snippet.strip():
        return {
            "function_names": set(),
            "call_sites": set(),
            "imports": set(),
        }
    function_names: set[str] = set()
    for pattern in _NON_PY_FUNC_PATTERNS:
        for match in pattern.finditer(snippet):
            name = match.group(1)
            if name:
                function_names.add(name)
    call_sites: set[str] = set()
    for match in _NON_PY_CALL_PATTERN.finditer(snippet):
        name = match.group(1)
        # Filter out language keywords that look like calls.
        if name in {
            "if",
            "for",
            "while",
            "switch",
            "return",
            "function",
            "func",
            "fn",
            "import",
            "require",
            "catch",
            "throw",
            "new",
            "delete",
            "typeof",
            "instanceof",
            "sizeof",
            "do",
        }:
            continue
        # And filter out names that exactly match a function definition
        # in the same snippet — avoids double-counting ``foo`` from
        # ``function foo(...)``.
        if name in function_names:
            continue
        call_sites.add(name)
    imports: set[str] = set()
    for pattern in _NON_PY_IMPORT_PATTERNS:
        for match in pattern.finditer(snippet):
            target = match.group(1)
            if target:
                imports.add(target)
    return {
        "function_names": function_names,
        "call_sites": call_sites,
        "imports": imports,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_semantic_signature(
    patch_text: str,
    repo_root: Optional[Path] = None,
) -> SemanticSignature:
    """Parse the diff and extract a semantic signature.

    ``repo_root`` is accepted for API compatibility / future expansion
    (e.g. to walk the post-patch worktree for fuller context); the
    current implementation derives the signature purely from the diff
    text, which keeps the function pure and side-effect-free.

    Returns a partial signature when a file fails to parse — handle
    parse errors gracefully so one bad fragment doesn't drop the entire
    patch out of clustering.
    """
    if not patch_text or not str(patch_text).strip():
        return SemanticSignature()

    files = _parse_unified_diff(patch_text)
    if not files:
        return SemanticSignature()

    function_names: set[str] = set()
    call_sites_added: set[str] = set()
    call_sites_removed: set[str] = set()
    added_imports: set[str] = set()
    removed_imports: set[str] = set()
    control_flow: set[str] = set()
    data_structures: set[str] = set()
    operator_kinds: set[str] = set()
    constants: set[str] = set()
    file_set: set[str] = set()
    parse_failed_files: set[str] = set()
    non_python_files: set[str] = set()

    for entry in files:
        if entry.path:
            file_set.add(_canonicalize_file_path(entry.path))

        added_blob = "\n".join(entry.added_lines)
        removed_blob = "\n".join(entry.removed_lines)

        if not entry.is_python:
            if entry.path:
                non_python_files.add(entry.path)
            for blob, call_bucket, import_bucket in (
                (added_blob, call_sites_added, added_imports),
                (removed_blob, call_sites_removed, removed_imports),
            ):
                pieces = _extract_non_python_signature_pieces(blob)
                function_names.update(pieces["function_names"])
                call_bucket.update(pieces["call_sites"])
                import_bucket.update(pieces["imports"])
            continue

        # Python: AST parse both the added and removed blobs and
        # collect the union of node kinds. SyntaxError on a fragment
        # is expected (most diff fragments aren't standalone-parseable);
        # _safe_parse tries multiple wrappers before giving up.
        added_tree = _safe_parse(added_blob) if added_blob.strip() else None
        removed_tree = _safe_parse(removed_blob) if removed_blob.strip() else None
        added_ok = added_tree is not None or not added_blob.strip()
        removed_ok = removed_tree is not None or not removed_blob.strip()
        if not added_ok and not removed_ok:
            parse_failed_files.add(entry.path)
            # Fall back to non-Python heuristic on Python source so the
            # patch still contributes *some* signal.
            for blob, call_bucket, import_bucket in (
                (added_blob, call_sites_added, added_imports),
                (removed_blob, call_sites_removed, removed_imports),
            ):
                pieces = _extract_non_python_signature_pieces(blob)
                function_names.update(pieces["function_names"])
                call_bucket.update(pieces["call_sites"])
                import_bucket.update(pieces["imports"])
            continue

        if added_tree is not None:
            function_names.update(_collect_function_names(added_tree))
            call_sites_added.update(_collect_call_names(added_tree))
            added_imports.update(_collect_imports(added_tree))
            control_flow.update(_collect_node_kinds(added_tree, _CONTROL_FLOW_TYPES))
            data_structures.update(_collect_node_kinds(added_tree, _DATA_STRUCTURE_TYPES))
            operator_kinds.update(_collect_operator_kinds(added_tree))
            constants.update(_collect_constants(added_tree))
        if removed_tree is not None:
            function_names.update(_collect_function_names(removed_tree))
            call_sites_removed.update(_collect_call_names(removed_tree))
            removed_imports.update(_collect_imports(removed_tree))
            control_flow.update(_collect_node_kinds(removed_tree, _CONTROL_FLOW_TYPES))
            data_structures.update(_collect_node_kinds(removed_tree, _DATA_STRUCTURE_TYPES))
            operator_kinds.update(_collect_operator_kinds(removed_tree))
            constants.update(_collect_constants(removed_tree))

    return SemanticSignature(
        changed_function_names=tuple(sorted(function_names)),
        changed_call_sites=tuple(sorted(call_sites_added | call_sites_removed)),
        added_imports=tuple(sorted(added_imports - removed_imports)),
        removed_imports=tuple(sorted(removed_imports - added_imports)),
        modified_control_flow=tuple(sorted(control_flow)),
        modified_data_structures=tuple(sorted(data_structures)),
        file_set_normalized=tuple(sorted(file_set)),
        operator_kinds=tuple(sorted(operator_kinds)),
        constant_signature=tuple(sorted(constants)),
        parse_failed_files=tuple(sorted(parse_failed_files)),
        non_python_files=tuple(sorted(non_python_files)),
    )


# Weights chosen to emphasise *what* the patch changes (function /
# call-site identity) over *where* (file set), since two patches that
# touch the same file but flip different operators must cluster apart.
# Sums to 1.0 so the returned distance lives on [0, 1].
_DISTANCE_WEIGHTS: dict[str, float] = {
    "function_names": 0.25,
    "call_sites": 0.25,
    "control_flow": 0.20,
    "imports": 0.10,
    "data_structures": 0.10,
    "file_set": 0.10,
}


def _jaccard_distance(left: Iterable[str], right: Iterable[str]) -> float:
    """1.0 minus Jaccard similarity. Empty-vs-empty → 0.0."""
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 0.0
    union = left_set | right_set
    if not union:
        return 0.0
    intersection = left_set & right_set
    return 1.0 - (len(intersection) / len(union))


def semantic_distance(a: SemanticSignature, b: SemanticSignature) -> float:
    """Weighted Jaccard distance between two semantic signatures.

    Returns a float on ``[0, 1]``: ``0.0`` means semantically identical
    (every component matches), ``1.0`` means completely different
    (no overlap on any component).

    Components and weights (sum to 1.0):

      * ``function_names``  (0.25) — functions whose body was edited.
      * ``call_sites``      (0.25) — callees added/removed/modified.
      * ``control_flow``    (0.20) — if/while/for/try/etc. node kinds
        plus operator kinds (Lt/Gt/Eq/Add/...) appearing in the edit
        region. Operator kinds are pooled in here so a patch that
        flips ``>`` → ``<`` increments distance even when the rest of
        the signature is identical.
      * ``imports``         (0.10) — Jaccard over the symmetric
        difference (added ∪ removed).
      * ``data_structures`` (0.10) — class/dict/list/etc. node kinds
        plus stringified constants. Constants are pooled in here so
        a patch that changes ``return 0`` → ``return 1`` shifts
        distance even when AST shape is identical.
      * ``file_set``        (0.10) — canonicalized touched-file paths.
    """
    # Pool operator_kinds into control_flow and constants into
    # data_structures to keep the spec'd weight allocation while
    # making the signature sensitive to operator/constant flips.
    cf_a = list(a.modified_control_flow) + list(a.operator_kinds)
    cf_b = list(b.modified_control_flow) + list(b.operator_kinds)
    ds_a = list(a.modified_data_structures) + list(a.constant_signature)
    ds_b = list(b.modified_data_structures) + list(b.constant_signature)

    distance = 0.0
    distance += _DISTANCE_WEIGHTS["function_names"] * _jaccard_distance(
        a.changed_function_names, b.changed_function_names
    )
    distance += _DISTANCE_WEIGHTS["call_sites"] * _jaccard_distance(
        a.changed_call_sites, b.changed_call_sites
    )
    distance += _DISTANCE_WEIGHTS["control_flow"] * _jaccard_distance(cf_a, cf_b)
    distance += _DISTANCE_WEIGHTS["imports"] * _jaccard_distance(
        list(a.added_imports) + list(a.removed_imports),
        list(b.added_imports) + list(b.removed_imports),
    )
    distance += _DISTANCE_WEIGHTS["data_structures"] * _jaccard_distance(ds_a, ds_b)
    distance += _DISTANCE_WEIGHTS["file_set"] * _jaccard_distance(
        a.file_set_normalized, b.file_set_normalized
    )
    # Clamp for floating-point safety.
    if distance < 0.0:
        return 0.0
    if distance > 1.0:
        return 1.0
    return distance


def semantic_similarity(a: SemanticSignature, b: SemanticSignature) -> float:
    """Convenience: ``1.0 - semantic_distance(a, b)``.

    The selector wants a *similarity* score (higher = more alike) to
    plug into the existing single-linkage clustering loop, which
    merges pairs whose similarity exceeds the threshold.
    """
    return 1.0 - semantic_distance(a, b)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _canonicalize_file_path(raw: str) -> str:
    """Normalize a file path for the ``file_set_normalized`` field.

    Lowercases the path, collapses redundant separators, and strips any
    leading ``./``. Keeps the relative-to-repo orientation so
    cross-platform diffs compare consistently.
    """
    if not raw:
        return ""
    text = str(raw).replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    # Collapse double slashes.
    while "//" in text:
        text = text.replace("//", "/")
    return text.lower()


def fallback_text_similarity(left_text: str, right_text: str) -> float:
    """Final-fallback text similarity used when AST parsing fails on
    every file in a candidate.

    Exposed for the selector's own fallback path; uses
    ``difflib.SequenceMatcher`` on the raw blobs so the signal is at
    least non-zero when both diffs share boilerplate.
    """
    return SequenceMatcher(None, left_text or "", right_text or "").ratio()


__all__ = [
    "SemanticSignature",
    "compute_semantic_signature",
    "semantic_distance",
    "semantic_similarity",
    "fallback_text_similarity",
]
