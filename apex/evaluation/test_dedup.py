"""Cross-candidate test-level deduplication (P1 step 7).

Multi-agent ensembles often produce near-duplicate tests because the
agents share training distribution. The V5 cross-candidate voter picks
ONE candidate file but doesn't deduplicate the tests inside the bundle —
so when three candidates each carry the same six tests plus one unique
test, the voter sees three "candidates of size seven" and the bundle's
real diversity is misjudged.

This module hashes each test function by its AST-shape signature
(variable names normalized, literals normalized, call-call structure
preserved). When two candidates carry tests with the same shape we keep
the test on the candidate with the most distinct surviving tests (or
break ties by deterministic candidate-id ordering) and drop it from the
others. Diagnostics record the per-candidate drop count so post-mortem
can confirm the dedup pass actually reduced overlap.

The rewriter only EVER drops test functions; it does not touch imports,
helpers, fixtures, or module-level code. Empty candidates after dedup
are kept in the bundle (the downstream voter / acceptance gate will
decide what to do with them) — the goal here is to surface honest
diversity, not to prune candidates outright.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional


@dataclass(frozen=True)
class TestHash:
    """Shape signature for one test function."""

    # Tell pytest not to collect this dataclass as a test class.
    __test__ = False

    test_name: str
    shape_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {"test_name": self.test_name, "shape_hash": self.shape_hash}


@dataclass
class CandidateDedupReport:
    """Per-candidate dedup outcome."""

    candidate_id: str
    test_id: str
    original_test_count: int
    surviving_test_count: int
    dropped_tests: list[str] = field(default_factory=list)
    parse_error: Optional[str] = None

    @property
    def changed(self) -> bool:
        return bool(self.dropped_tests) or self.parse_error is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "test_id": self.test_id,
            "original_test_count": self.original_test_count,
            "surviving_test_count": self.surviving_test_count,
            "dropped_tests": list(self.dropped_tests),
            "parse_error": self.parse_error,
        }


@dataclass
class DedupResult:
    """Aggregate dedup pass result."""

    candidates: list[dict[str, Any]] = field(default_factory=list)
    reports: list[CandidateDedupReport] = field(default_factory=list)
    total_dropped: int = 0
    duplicate_groups: int = 0

    @property
    def changed(self) -> bool:
        return self.total_dropped > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_dropped": self.total_dropped,
            "duplicate_groups": self.duplicate_groups,
            "reports": [r.to_dict() for r in self.reports],
        }


def dedup_candidate_bundle(
    candidates: Iterable[Mapping[str, Any]],
) -> DedupResult:
    """Deduplicate test functions across the candidate bundle.

    Each candidate is a dict with at least:
      * ``candidate_id``  — stable id for tie-breaking
      * ``test_id``       — descriptive id for diagnostics (often equal
        to ``candidate_id`` when each candidate is one file)
      * ``artifact_content`` — the test file source

    Returns a fresh list of candidate dicts (deep-copied with possibly
    rewritten ``artifact_content``) plus a per-candidate diagnostic.
    Candidates whose source fails to AST-parse are returned untouched
    with a ``parse_error`` recorded — biased toward false-negatives so
    we never accidentally drop tests because of a transient parse hiccup.
    """

    materialized = [dict(c) for c in candidates]
    if len(materialized) < 2:
        return DedupResult(
            candidates=materialized,
            reports=[],
        )

    # Step 1: hash every test function in every candidate.
    # per_candidate: candidate_index → list[(test_name, shape_hash, ast_node_index)]
    per_candidate: list[list[TestHash]] = []
    parse_errors: list[Optional[str]] = []
    for candidate in materialized:
        source = str(candidate.get("artifact_content") or "")
        hashes, parse_error = _hash_tests_in_source(source)
        per_candidate.append(hashes)
        parse_errors.append(parse_error)

    # Step 2: bucket (shape_hash → [(candidate_index, test_name)]).
    buckets: dict[str, list[tuple[int, str]]] = {}
    for c_index, hashes in enumerate(per_candidate):
        for h in hashes:
            buckets.setdefault(h.shape_hash, []).append((c_index, h.test_name))

    # Step 3: pick the keeper for every multi-occupant bucket.
    drops_per_candidate: dict[int, set[str]] = {i: set() for i in range(len(materialized))}
    duplicate_groups = 0
    for shape_hash, occupants in buckets.items():
        if len(occupants) < 2:
            continue
        duplicate_groups += 1
        keeper_index = _pick_keeper(occupants, materialized)
        for c_index, test_name in occupants:
            if c_index == keeper_index:
                continue
            drops_per_candidate[c_index].add(test_name)

    # Step 4: rewrite each candidate's artifact, build per-candidate report.
    reports: list[CandidateDedupReport] = []
    rewritten: list[dict[str, Any]] = []
    total_dropped = 0
    for c_index, candidate in enumerate(materialized):
        original_count = len(per_candidate[c_index])
        drops = drops_per_candidate[c_index]
        parse_error = parse_errors[c_index]
        if parse_error is not None:
            reports.append(
                CandidateDedupReport(
                    candidate_id=str(candidate.get("candidate_id") or ""),
                    test_id=str(candidate.get("test_id") or ""),
                    original_test_count=original_count,
                    surviving_test_count=original_count,
                    parse_error=parse_error,
                )
            )
            rewritten.append(candidate)
            continue
        if not drops:
            reports.append(
                CandidateDedupReport(
                    candidate_id=str(candidate.get("candidate_id") or ""),
                    test_id=str(candidate.get("test_id") or ""),
                    original_test_count=original_count,
                    surviving_test_count=original_count,
                )
            )
            rewritten.append(candidate)
            continue
        new_source = _drop_tests_from_source(str(candidate.get("artifact_content") or ""), drops)
        new_candidate = dict(candidate)
        new_candidate["artifact_content"] = new_source
        rewritten.append(new_candidate)
        surviving = original_count - len(drops)
        total_dropped += len(drops)
        reports.append(
            CandidateDedupReport(
                candidate_id=str(candidate.get("candidate_id") or ""),
                test_id=str(candidate.get("test_id") or ""),
                original_test_count=original_count,
                surviving_test_count=surviving,
                dropped_tests=sorted(drops),
            )
        )

    return DedupResult(
        candidates=rewritten,
        reports=reports,
        total_dropped=total_dropped,
        duplicate_groups=duplicate_groups,
    )


def _hash_tests_in_source(source: str) -> tuple[list[TestHash], Optional[str]]:
    """Return ``(per-test hashes, parse_error)`` for *source*.

    Empty source returns an empty list and no error.
    """

    if not source.strip():
        return [], None
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [], f"SyntaxError: {exc}"
    out: list[TestHash] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("test_"):
                continue
            shape = _normalize_test_node(node)
            digest = hashlib.sha1(shape.encode("utf-8")).hexdigest()
            out.append(TestHash(test_name=node.name, shape_hash=digest))
        elif isinstance(node, ast.ClassDef):
            # ``TestX`` classes — hash each test_ method too. Use
            # ``ClassName.method_name`` as the dedup id so different
            # classes are treated as distinct surfaces.
            for child in node.body:
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and child.name.startswith("test_"):
                    shape = _normalize_test_node(child)
                    digest = hashlib.sha1(shape.encode("utf-8")).hexdigest()
                    out.append(
                        TestHash(
                            test_name=f"{node.name}.{child.name}",
                            shape_hash=digest,
                        )
                    )
    return out, None


def _normalize_test_node(node: ast.AST) -> str:
    """Return a stable string capturing the test's call-shape.

    Strips:
      * variable names (replaced with ``_``)
      * literal values (replaced with the type sentinel)
      * docstrings
      * line numbers / col offsets

    Preserves:
      * call-target attribute chains (``mock.patch`` stays distinct from
        ``monkeypatch.setattr``)
      * comparison / boolean operators
      * raise type names
      * with-statement context-manager qualnames
    """

    return _NormalizingVisitor().normalize(node)


class _NormalizingVisitor:
    """Render an AST node into a normalized string for shape hashing."""

    def normalize(self, node: ast.AST) -> str:
        # Drop function name, args, decorators, return type — only the
        # body shape matters for diversity.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = [b for b in node.body if not _is_docstring(b)]
            return f"FN[{','.join(self._render(b) for b in body)}]"
        return self._render(node)

    def _render(self, node: Any) -> str:
        if node is None:
            return "_"
        if isinstance(node, list):
            return f"[{','.join(self._render(item) for item in node)}]"
        if not isinstance(node, ast.AST):
            return f"<{type(node).__name__}>"
        cls_name = type(node).__name__
        # Special cases that need to preserve identity beyond bare class name
        if isinstance(node, ast.Constant):
            return f"C<{type(node.value).__name__}>"
        if isinstance(node, ast.Name):
            return "N"
        if isinstance(node, ast.arg):
            return "A"
        if isinstance(node, ast.Attribute):
            return f"AT<{node.attr}>({self._render(node.value)})"
        if isinstance(node, ast.Call):
            return (
                f"CL<{self._render_func(node.func)}>("
                f"{','.join(self._render(a) for a in node.args)},"
                f"{','.join(f'{kw.arg}={self._render(kw.value)}' for kw in node.keywords)}"
                ")"
            )
        if isinstance(node, ast.Compare):
            ops = ",".join(type(op).__name__ for op in node.ops)
            comparators = ",".join(self._render(c) for c in node.comparators)
            return f"CMP<{ops}>({self._render(node.left)},{comparators})"
        if isinstance(node, ast.BoolOp):
            return f"BO<{type(node.op).__name__}>({','.join(self._render(v) for v in node.values)})"
        if isinstance(node, ast.UnaryOp):
            return f"UO<{type(node.op).__name__}>({self._render(node.operand)})"
        if isinstance(node, ast.BinOp):
            return (
                f"BIN<{type(node.op).__name__}>("
                f"{self._render(node.left)},{self._render(node.right)})"
            )
        if isinstance(node, ast.Raise):
            exc_part = self._render(node.exc) if node.exc else "_"
            return f"RAISE({exc_part})"
        if isinstance(node, ast.Assert):
            msg_part = self._render(node.msg) if node.msg else "_"
            return f"ASSERT({self._render(node.test)},{msg_part})"
        # Default: walk children in field order, tagged with class name
        fields = []
        for field_name, value in ast.iter_fields(node):
            if field_name in {"lineno", "col_offset", "end_lineno", "end_col_offset"}:
                continue
            fields.append(f"{field_name}={self._render(value)}")
        return f"{cls_name}({','.join(fields)})"

    def _render_func(self, node: ast.AST) -> str:
        # Preserve the dotted call target so ``mock.patch`` ≠ ``monkeypatch.setattr``
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{self._render_func(node.value)}.{node.attr}"
        return self._render(node)


def _is_docstring(stmt: ast.AST) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _pick_keeper(
    occupants: list[tuple[int, str]],
    candidates: list[Mapping[str, Any]],
) -> int:
    """Pick which candidate keeps a duplicated test.

    Heuristic: the candidate with the largest artifact (more code = more
    context) wins. Ties break on candidate_id alphabetical order so the
    dedup pass is fully deterministic across runs.
    """

    def sort_key(entry: tuple[int, str]) -> tuple[int, str]:
        c_index, _ = entry
        candidate = candidates[c_index]
        size = len(str(candidate.get("artifact_content") or ""))
        cid = str(candidate.get("candidate_id") or "")
        # Negative size so larger sorts first; cid ASC for stable tiebreak.
        return (-size, cid)

    return min(occupants, key=sort_key)[0]


def _drop_tests_from_source(source: str, drops: set[str]) -> str:
    """Remove the named test functions from *source* and return new source.

    Operates via AST round-trip so the rewritten file remains
    syntactically valid. Class-method drops use ``Class.method`` keys —
    when dropping a class method we strip just the method, leaving the
    class shell intact (other methods may still be live).
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    new_body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in drops:
                continue
            new_body.append(node)
        elif isinstance(node, ast.ClassDef):
            new_class_body: list[ast.stmt] = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualname = f"{node.name}.{child.name}"
                    if qualname in drops:
                        continue
                new_class_body.append(child)
            if not new_class_body:
                # Class becomes empty — keep a ``pass`` so the file still parses.
                new_class_body = [ast.Expr(value=ast.Constant(value=Ellipsis))]
            node.body = new_class_body
            new_body.append(node)
        else:
            new_body.append(node)
    tree.body = new_body
    try:
        return ast.unparse(tree).rstrip() + "\n"
    except Exception:  # pragma: no cover - defensive
        return source
