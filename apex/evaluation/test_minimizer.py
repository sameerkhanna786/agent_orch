"""Greedy minimization of an agent-generated test suite.

After Stage 4 (mutation discrimination), each candidate test file in the
agent's portfolio has a *measured* contribution: how many F2P transitions
it produced and how many mutants it killed. This module greedily picks
the smallest subset of files that still covers the union of F2P kills
and mutation kills, and discards the rest.

Why minimize at all:
    * Reward-hacking pressure produces "shotgun" suites (10-20 trivial
      assertions) that the comparison metric counts as wins. The Apr 27
      validate run had `generated_artifact_count` = 7-11 per ansible task
      while the gold suite covered the same contract in 1-3 tests.
    * Smaller suites are reviewable. SWE-Bench Pro maintainers manually
      audit failing PRs and trim non-load-bearing tests; a pre-trimmed
      suite is more likely to land.
    * Cumulative regression suites grow per-milestone. Pruning per
      milestone keeps the long-horizon working set bounded.

Non-goals (deferred):
    * Function-level minimization. V1 operates at the file level — drop
      whole files but never edit a kept file. Function-level requires
      AST-rewriting the test source, which is a large surface for bugs.
    * Mutation-survival re-verification post-minimization. We trust the
      coverage map computed before minimization; if the dropped tests
      were truly redundant the kept ones still kill the same mutants.

Public API:
    minimize_suite(test_artifacts, f2p_payload, mutation_report)
        -> (kept_artifacts, MinimizationReport)
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class MinimizationReport:
    original_count: int
    minimized_count: int
    dropped_paths: list[str] = field(default_factory=list)
    kept_paths: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: Optional[str] = None
    f2p_total: int = 0
    f2p_covered_by_kept: int = 0
    mutation_total: int = 0
    mutation_covered_by_kept: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_count": self.original_count,
            "minimized_count": self.minimized_count,
            "dropped_paths": list(self.dropped_paths),
            "kept_paths": list(self.kept_paths),
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "f2p_total": self.f2p_total,
            "f2p_covered_by_kept": self.f2p_covered_by_kept,
            "mutation_total": self.mutation_total,
            "mutation_covered_by_kept": self.mutation_covered_by_kept,
        }


def minimize_suite(
    *,
    test_artifacts: list[dict[str, Any]],
    f2p_payload: Optional[dict[str, Any]] = None,
    mutation_report: Optional[dict[str, Any]] = None,
) -> tuple[list[dict[str, Any]], MinimizationReport]:
    """Drop test files whose coverage is fully subsumed by other kept files.

    Coverage is the union of:
        * nodeids that transitioned fail->pass under the F2P oracle
          (``f2p_payload["transitions"][nodeid]["f2p"] is True``)
        * nodeids listed in any ``mutation_report["per_mutant"][i]["killing_tests"]``

    Files mapped to *no* covered nodeids — typically conftests, helpers,
    or shotgun assertions that didn't catch anything — are preserved if
    we have no signal at all (defensive: all artifacts kept), and dropped
    only when at least one peer file *does* contribute coverage.

    Always keeps at least one artifact: an empty suite is never an
    improvement on a non-empty one for downstream consumers.
    """
    artifacts = list(test_artifacts or [])
    n = len(artifacts)
    if n <= 1:
        return artifacts, MinimizationReport(
            original_count=n,
            minimized_count=n,
            kept_paths=[_artifact_path(a) for a in artifacts if _artifact_path(a)],
            skipped=True,
            skip_reason="too_few_artifacts",
        )

    # Build the (path -> covered-nodeid-set) map.
    f2p_kills_by_path: dict[str, set[str]] = {}
    if f2p_payload:
        transitions = dict(f2p_payload.get("transitions") or {})
        for nodeid, info in transitions.items():
            if not isinstance(info, dict) or not info.get("f2p"):
                continue
            path = _path_from_nodeid(str(nodeid))
            if not path:
                continue
            f2p_kills_by_path.setdefault(path, set()).add(str(nodeid))

    mutation_kills_by_path: dict[str, set[str]] = {}
    if mutation_report:
        for entry in mutation_report.get("per_mutant") or []:
            if not isinstance(entry, dict):
                continue
            # Marker identifies the *mutant* (operator + line), NOT the
            # killing nodeid. When two test files both kill the same
            # mutant, they share a marker — exactly the redundancy we
            # want greedy set-cover to detect and drop.
            marker = f"mut::{entry.get('operator')}@{entry.get('line')}"
            for nodeid in entry.get("killing_tests") or []:
                path = _path_from_nodeid(str(nodeid))
                if not path:
                    continue
                mutation_kills_by_path.setdefault(path, set()).add(marker)

    if not f2p_kills_by_path and not mutation_kills_by_path:
        # No signal — minimization would be unprincipled. Keep everything.
        return artifacts, MinimizationReport(
            original_count=n,
            minimized_count=n,
            kept_paths=[_artifact_path(a) for a in artifacts if _artifact_path(a)],
            skipped=True,
            skip_reason="no_coverage_signal",
        )

    # Greedy set cover, broken by (f2p_kills, mutation_kills, -loc).
    f2p_universe: set[str] = (
        set().union(*f2p_kills_by_path.values()) if f2p_kills_by_path else set()
    )
    mutation_universe: set[str] = (
        set().union(*mutation_kills_by_path.values()) if mutation_kills_by_path else set()
    )
    target_universe = f2p_universe | mutation_universe

    paths_in_artifacts = [_artifact_path(a) for a in artifacts]
    artifact_by_path: dict[str, dict[str, Any]] = {}
    for path, artifact in zip(paths_in_artifacts, artifacts):
        if path:
            artifact_by_path[path] = artifact

    contributing_paths = [
        path
        for path in paths_in_artifacts
        if path and (path in f2p_kills_by_path or path in mutation_kills_by_path)
    ]

    if not contributing_paths:
        # We have signal in the universe but no artifact-path matched it
        # (likely test_writer wrote tests that reference fixture-loaded
        # paths which collected under a different rootdir). Keep all to
        # avoid silently dropping useful files.
        return artifacts, MinimizationReport(
            original_count=n,
            minimized_count=n,
            kept_paths=[p for p in paths_in_artifacts if p],
            skipped=True,
            skip_reason="no_artifact_path_matched_coverage",
        )

    kept: list[str] = []
    covered: set[str] = set()
    remaining = set(contributing_paths)
    while remaining and covered != target_universe:
        best_path: Optional[str] = None
        best_key: tuple[int, int, int, int] = (-1, -1, -1, 0)
        for path in remaining:
            f2p_set = f2p_kills_by_path.get(path, set())
            mut_set = mutation_kills_by_path.get(path, set())
            new_f2p = len(f2p_set - covered)
            new_mut = len(mut_set - covered)
            if new_f2p == 0 and new_mut == 0:
                continue  # Adds nothing — never selectable
            artifact = artifact_by_path.get(path) or {}
            content = str(artifact.get("content") or "")
            # Larger files break ties in favor of the smaller — proxy for
            # readability and for the "shotgun assertion" anti-pattern.
            loc_negative = -len(content.splitlines())
            key = (new_f2p, new_mut, loc_negative, -len(path))
            if key > best_key:
                best_key = key
                best_path = path
        if best_path is None:
            break
        kept.append(best_path)
        covered.update(f2p_kills_by_path.get(best_path, set()))
        covered.update(mutation_kills_by_path.get(best_path, set()))
        remaining.discard(best_path)

    if not kept:
        # Defensive — preserve at least one file even if greedy selected
        # nothing (e.g. all artifacts had empty contribution sets).
        first = next((p for p in paths_in_artifacts if p), None)
        if first:
            kept.append(first)

    # Preserve files we didn't classify as test files at all (conftest.py,
    # fixture helpers) — dropping them would silently break the kept tests.
    auxiliary = [
        p for p in paths_in_artifacts if p and not _looks_like_test_file(p) and p not in kept
    ]
    final_paths_set = set(kept) | set(auxiliary)
    minimized = [
        artifact
        for artifact, path in zip(artifacts, paths_in_artifacts)
        if path and path in final_paths_set
    ]
    dropped = [p for p in paths_in_artifacts if p and p not in final_paths_set]

    return minimized, MinimizationReport(
        original_count=n,
        minimized_count=len(minimized),
        kept_paths=[p for p in paths_in_artifacts if p and p in final_paths_set],
        dropped_paths=dropped,
        f2p_total=len(f2p_universe),
        f2p_covered_by_kept=len(covered & f2p_universe),
        mutation_total=len(mutation_universe),
        mutation_covered_by_kept=len(covered & mutation_universe),
    )


def minimize_to_passing_subset(
    *,
    artifact_text: str,
    tier_3_run: dict[str, Any],
    style: Any,
    keep_minimum: int = 3,
) -> tuple[str, list[str]]:
    """Drop failing Python test definitions from a partially passing file.

    This is the low-risk repair path for TestGenEval-style suites where the
    harness filtered to "one good test plus broken siblings". It removes only
    tests whose nodeid is explicitly reported as fail/error and keeps the
    original artifact when too few tests would remain.
    """

    if (getattr(style, "language", "python") or "").lower() not in {"python", "py", "python3"}:
        return artifact_text, []
    source = str(artifact_text or "")
    failing = _failing_test_names(tier_3_run)
    if not failing:
        return source, []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, []
    all_tests = _test_names_in_tree(tree)
    remaining = [name for name in all_tests if name not in failing]
    if len(remaining) < max(1, int(keep_minimum or 1)):
        return source, []
    unused_helpers = _helpers_referenced_only_by_dropped_tests(
        tree=tree,
        dropped_tests=failing,
        remaining_tests=set(remaining),
    )
    transformer = _DropFailingTests(failing)
    new_tree = transformer.visit(tree)
    if unused_helpers and isinstance(new_tree, ast.Module):
        new_tree = _DropUnusedTopLevelHelpers(unused_helpers).visit(new_tree)
    if isinstance(new_tree, ast.Module) and not new_tree.body:
        return source, []
    ast.fix_missing_locations(new_tree)
    if not transformer.dropped:
        return source, []
    try:
        rendered = ast.unparse(new_tree).strip() + "\n"
    except Exception:
        return source, []
    # Strict W3 gate: parse + compile.
    from .final_acceptance_gate import strict_syntax_check

    syntax_ok, _ = strict_syntax_check(rendered)
    if not syntax_ok:
        return source, []
    return rendered, sorted(transformer.dropped)


def test_names_in_artifact(artifact_text: str) -> list[str]:
    """Return test function/method names found in a Python artifact."""

    try:
        tree = ast.parse(str(artifact_text or ""))
    except SyntaxError:
        return []
    return _test_names_in_tree(tree)


def drop_tests_from_artifact(
    artifact_text: str,
    names: set[str] | list[str] | tuple[str, ...],
    *,
    keep_minimum: int = 1,
) -> str:
    """Public wrapper around the AST drop transformer."""

    rendered, _ = drop_tests_from_artifact_with_report(
        artifact_text,
        names,
        keep_minimum=keep_minimum,
    )
    return rendered


def drop_tests_from_artifact_with_report(
    artifact_text: str,
    names: set[str] | list[str] | tuple[str, ...],
    *,
    keep_minimum: int = 1,
) -> tuple[str, list[str]]:
    """Drop named Python tests and return ``(new_source, dropped_names)``."""

    source = str(artifact_text or "")
    failing = {str(name) for name in names if str(name)}
    if not failing:
        return source, []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, []
    all_tests = _test_names_in_tree(tree)
    remaining = [name for name in all_tests if name not in failing]
    if len(remaining) < max(0, int(keep_minimum or 0)):
        return source, []
    unused_helpers = _helpers_referenced_only_by_dropped_tests(
        tree=tree,
        dropped_tests=failing,
        remaining_tests=set(remaining),
    )
    transformer = _DropFailingTests(failing)
    new_tree = transformer.visit(tree)
    if unused_helpers and isinstance(new_tree, ast.Module):
        new_tree = _DropUnusedTopLevelHelpers(unused_helpers).visit(new_tree)
    if isinstance(new_tree, ast.Module) and not new_tree.body:
        return source, []
    ast.fix_missing_locations(new_tree)
    if not transformer.dropped:
        return source, []
    try:
        rendered = ast.unparse(new_tree).strip() + "\n"
    except Exception:
        return source, []
    # Strict W3 gate: parse + compile. Catches the gate-leak shapes
    # (return at module scope, async-await outside async fn, duplicate
    # kwargs) that ``ast.parse`` accepts but ``compile`` rejects.
    from .final_acceptance_gate import strict_syntax_check

    syntax_ok, _ = strict_syntax_check(rendered)
    if not syntax_ok:
        return source, []
    return rendered, sorted(transformer.dropped)


def _artifact_path(artifact: dict[str, Any]) -> str:
    return str((artifact or {}).get("path") or "").strip()


def _path_from_nodeid(nodeid: str) -> str:
    head, _, _ = nodeid.partition("::")
    return head.strip()


def _looks_like_test_file(path: str) -> bool:
    """Whether the basename matches the pytest test-file convention.

    Only the basename matters here: `tests/conftest.py` is NOT a test file
    even though it lives under tests/, and `mypkg/test_inline.py` IS a
    test file even though it doesn't live under tests/. The auxiliary-file
    preservation logic relies on this — dropping a conftest because the
    parent dir matched would silently break every kept test that depends
    on its fixtures.
    """
    basename = path.lower().rsplit("/", 1)[-1]
    return basename.startswith("test_") or basename.endswith("_test.py")


def _failing_test_names(tier_3_run: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    per_test = dict((tier_3_run or {}).get("per_test_status") or {})
    for nodeid, raw_status in per_test.items():
        if str(raw_status or "").lower() not in {"fail", "error"}:
            continue
        names.update(_test_names_from_nodeid(str(nodeid or "")))
    names.update(detect_isolation_offenders(tier_3_run or {}))
    return names


def detect_isolation_offenders(tier_3_run: dict[str, Any]) -> set[str]:
    """Return tests that pass alone but fail in the combined generated suite."""

    offenders: set[str] = set()
    for raw in tier_3_run.get("isolation_offenders") or []:
        offenders.update(_test_names_from_nodeid(str(raw or "")))
    isolated = _status_map(
        tier_3_run.get("isolated_per_test_status")
        or tier_3_run.get("per_test_status_isolated")
        or tier_3_run.get("isolated_status")
    )
    combined = _status_map(
        tier_3_run.get("combined_per_test_status")
        or tier_3_run.get("suite_per_test_status")
        or tier_3_run.get("per_test_status_combined")
    )
    for nodeid, isolated_status in isolated.items():
        if isolated_status != "pass":
            continue
        if combined.get(nodeid) in {"fail", "error"}:
            offenders.update(_test_names_from_nodeid(nodeid))
    return offenders


def _status_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(status or "").lower() for key, status in value.items()}


def _test_names_from_nodeid(nodeid: str) -> set[str]:
    parts = str(nodeid or "").split("::")
    parts = [parts[0]] if len(parts) == 1 else parts[1:]
    names: set[str] = set()
    for part in parts:
        clean = part.split("[", 1)[0].strip()
        if clean.startswith("test_"):
            names.add(clean)
    return names


def _test_names_in_tree(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            names.append(node.name)
    return names


class _DropFailingTests(ast.NodeTransformer):
    def __init__(self, failing: set[str]) -> None:
        self.failing = set(failing)
        self.dropped: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST | None:
        if node.name in self.failing:
            self.dropped.add(node.name)
            return None
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST | None:
        if node.name in self.failing:
            self.dropped.add(node.name)
            return None
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST | None:
        node = self.generic_visit(node)
        if isinstance(node, ast.ClassDef) and not node.body:
            node.body = [ast.Pass()]
        return node


class _DropUnusedTopLevelHelpers(ast.NodeTransformer):
    def __init__(self, unused: set[str]) -> None:
        self.unused = set(unused)

    def visit_Module(self, node: ast.Module) -> ast.Module:
        body: list[ast.stmt] = []
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if child.name in self.unused and not child.name.startswith("test_"):
                    continue
            elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                names = _assigned_top_level_names(child)
                if names and names <= self.unused:
                    continue
            body.append(child)
        node.body = body
        return node


def _helpers_referenced_only_by_dropped_tests(
    *,
    tree: ast.Module,
    dropped_tests: set[str],
    remaining_tests: set[str],
) -> set[str]:
    top_level_helpers = _top_level_helper_names(tree)
    if not top_level_helpers:
        return set()
    helper_refs = _top_level_helper_refs(tree)
    dropped_refs: set[str] = set()
    remaining_refs: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        refs = _load_name_refs(node)
        if node.name in dropped_tests:
            dropped_refs.update(refs)
        elif node.name in remaining_tests:
            remaining_refs.update(refs)
    dropped_refs = _expand_helper_refs(dropped_refs, helper_refs, top_level_helpers)
    remaining_refs = _expand_helper_refs(remaining_refs, helper_refs, top_level_helpers)
    return (dropped_refs - remaining_refs) & top_level_helpers


def _top_level_helper_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("test_"):
                names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            names.update(_assigned_top_level_names(node))
    return names


def _top_level_helper_refs(tree: ast.Module) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("test_"):
                refs[node.name] = _load_name_refs(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for name in _assigned_top_level_names(node):
                refs[name] = _load_name_refs(node)
    return refs


def _expand_helper_refs(
    refs: set[str],
    helper_refs: dict[str, set[str]],
    helpers: set[str],
) -> set[str]:
    expanded = set(refs)
    changed = True
    while changed:
        changed = False
        for name in list(expanded & helpers):
            for ref in helper_refs.get(name, set()):
                if ref not in expanded:
                    expanded.add(ref)
                    changed = True
    return expanded


def _assigned_top_level_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
    targets = list(getattr(node, "targets", [])) or [getattr(node, "target", None)]
    return {
        target.id
        for target in targets
        if isinstance(target, ast.Name) and not target.id.startswith("_")
    }


def _load_name_refs(node: ast.AST) -> set[str]:
    refs: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            refs.add(child.id)
    return refs
