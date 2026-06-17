"""Cross-task persistent testgen memory (Phase I.7).

Bridges per-task in-loop signals (F2P, mutation, axis coverage,
edge predictions) into the cross-run RepoMemoryStore. The stored
insights then surface as priors at the START of new testgen rollouts
on the same repo, so APEX gets strictly better the more it runs.

Insight types stored across solves on the same repo:

    TESTGEN_FOCUS_FILE_HOTSPOT
        File paths repeatedly targeted by past testgen tasks. Use
        case: warm-start the test_writer's exemplar miner.

    TESTGEN_F2P_BUG_PATTERN
        Specific (file, line) loci where a past task's tests
        successfully F2P'd. Use case: hint the agent to stress test
        the same surface — bug fixes often cluster.

    TESTGEN_RESISTANT_MUTATION_CLASS
        Mutation operator classes that historically SURVIVE the
        agent's tests in this repo. Use case: nudge the agent toward
        stricter assertions for those classes.

    TESTGEN_KILLED_MUTATION_CLASS
        Mutation operator classes that historically GET KILLED in
        this repo — confirms the test_writer is already strong here.

    TESTGEN_LOW_COVERAGE_HOTSPOT
        File paths where past coverage gaps were repeatedly the same.
        Use case: pre-seed the next agent with the recurring blind
        spot.

    TESTGEN_AXIS_COVERAGE_HOTSPOT
        Contract axes that past generated tests missed for a target.
        Use case: nudge the agent to include explicit positive,
        malformed, boundary, and ordering assertions when the same
        repo surface recurs.

These insights are stored via RepoMemoryStore so they share the
existing decay / merge / cap machinery. The store auto-decays
confidence on load, so stale insights fade naturally.

All extraction helpers are defensive: missing keys, type-coerced
values, malformed payloads → empty list (with a debug log).
Generalizes outside benchmarks because every signal is rooted in
either F2P / mutation / coverage payloads, all of which the modes
API exposes.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from .repo_memory import PersistedInsight, RepoMemoryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Insight type constants
# ---------------------------------------------------------------------------

INSIGHT_TYPE_TESTGEN_FOCUS_FILE_HOTSPOT = "TESTGEN_FOCUS_FILE_HOTSPOT"
INSIGHT_TYPE_TESTGEN_F2P_BUG_PATTERN = "TESTGEN_F2P_BUG_PATTERN"
INSIGHT_TYPE_TESTGEN_RESISTANT_MUTATION_CLASS = "TESTGEN_RESISTANT_MUTATION_CLASS"
INSIGHT_TYPE_TESTGEN_KILLED_MUTATION_CLASS = "TESTGEN_KILLED_MUTATION_CLASS"
INSIGHT_TYPE_TESTGEN_LOW_COVERAGE_HOTSPOT = "TESTGEN_LOW_COVERAGE_HOTSPOT"
INSIGHT_TYPE_TESTGEN_AXIS_COVERAGE_HOTSPOT = "TESTGEN_AXIS_COVERAGE_HOTSPOT"

ALL_TESTGEN_INSIGHT_TYPES = (
    INSIGHT_TYPE_TESTGEN_FOCUS_FILE_HOTSPOT,
    INSIGHT_TYPE_TESTGEN_F2P_BUG_PATTERN,
    INSIGHT_TYPE_TESTGEN_RESISTANT_MUTATION_CLASS,
    INSIGHT_TYPE_TESTGEN_KILLED_MUTATION_CLASS,
    INSIGHT_TYPE_TESTGEN_LOW_COVERAGE_HOTSPOT,
    INSIGHT_TYPE_TESTGEN_AXIS_COVERAGE_HOTSPOT,
)


# Default confidences when extracting fresh insights from a single
# task's run. Confidences blend toward higher values when the same
# insight is re-observed on later tasks via RepoMemoryStore's
# support-weighted convex combination.
_CONFIDENCE_FOCUS_FILE = 0.7
_CONFIDENCE_F2P_BUG = 0.85  # F2P'ing is high-precision evidence
_CONFIDENCE_MUTATION_CLASS = 0.7
_CONFIDENCE_COVERAGE_HOTSPOT = 0.65
_CONFIDENCE_AXIS_COVERAGE = 0.75


def _norm_paths(values: Optional[Iterable[Any]], *, cap: int = 6) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        s = str(v or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= cap:
            break
    return out


def _operator_class_from_signature(sig: str) -> Optional[str]:
    """Extract the mutation OPERATOR FAMILY (e.g., 'boundary',
    'arithmetic', 'constant_replacement') from a `operator@path:line`
    signature.

    The operator naming follows the mutation_engine convention:
        boundary_<_to_<=, boundary_==_to_!=, ...
        arith_+_to_-, arith_*_to_/, ...
        constant_True_to_False, constant_replacement_5_to_1, ...
        return_x_to_None, ...
        statement_deletion, ...
        conditional_negation, ...

    We collapse to the family prefix so cross-task insights aggregate
    across specific line numbers / values.
    """
    head, _, _ = (sig or "").partition("@")
    head = head.strip().lower()
    if not head:
        return None
    if head.startswith("boundary"):
        return "boundary"
    if head.startswith("arith"):
        return "arithmetic"
    if head.startswith("constant"):
        return "constant_replacement"
    if head.startswith("return"):
        return "return_value"
    if head.startswith("statement"):
        return "statement_deletion"
    if head.startswith("conditional") or head.startswith("negate"):
        return "conditional_negation"
    return head.partition("_")[0] or None


def _norm_symbols(values: Optional[Iterable[Any]], *, cap: int = 10) -> list[str]:
    return _norm_paths(values, cap=cap)


# ---------------------------------------------------------------------------
# Public extraction API
# ---------------------------------------------------------------------------


def extract_testgen_insights_from_run_summary(
    *,
    focus_files: Optional[Iterable[str]] = None,
    f2p_summary: Optional[dict[str, Any]] = None,
    mutation_summary: Optional[dict[str, Any]] = None,
    coverage_gap_summary: Optional[dict[str, Any]] = None,
    axis_coverage_summary: Optional[dict[str, Any]] = None,
    target_comparison: Optional[Iterable[dict[str, Any]]] = None,
    stage_name: str = "test_writer",
) -> list[PersistedInsight]:
    """Convert a completed testgen run's per-task summaries into the
    list of cross-run PersistedInsight records to merge into the store.

    All inputs are optional — pass whatever the caller has. Each input
    independently produces 0..N insights; the union is returned.
    Defensive about missing/malformed fields so callers can pass raw
    dicts straight from JSON without preflight cleanup.
    """
    insights: list[PersistedInsight] = []
    focus_paths = _norm_paths(focus_files)

    # Hotspot files: every focus_file gets an insight with weight 1.
    # Cross-task re-observation will boost support_count.
    for path in focus_paths:
        insights.append(
            PersistedInsight(
                insight_type=INSIGHT_TYPE_TESTGEN_FOCUS_FILE_HOTSPOT,
                description=f"Repeatedly targeted in past testgen tasks: {path}",
                confidence=_CONFIDENCE_FOCUS_FILE,
                file_paths=[path],
                stage_name=stage_name,
            )
        )

    # F2P bug patterns: file paths where the agent's tests successfully
    # F2P'd. The supporting test nodeids end up in test_ids so the
    # signature stays unique per (file, test) pair — re-observation
    # of the SAME (file, test) bumps support; new tests on the same
    # file create new insights that share file_paths.
    if isinstance(f2p_summary, dict) and f2p_summary.get("any_f2p"):
        f2p_tests_raw = f2p_summary.get("f2p_tests")
        if isinstance(f2p_tests_raw, list) and f2p_tests_raw:
            files_for_f2p: dict[str, list[str]] = {}
            for nodeid in f2p_tests_raw:
                node_str = str(nodeid or "").strip()
                if not node_str:
                    continue
                file_part, _, _ = node_str.partition("::")
                files_for_f2p.setdefault(file_part or "?", []).append(node_str)
            for file_part, nodeids in files_for_f2p.items():
                insights.append(
                    PersistedInsight(
                        insight_type=INSIGHT_TYPE_TESTGEN_F2P_BUG_PATTERN,
                        description=(
                            f"Tests in `{file_part}` previously F2P'd "
                            f"({len(nodeids)} test(s)). Re-target this surface."
                        ),
                        confidence=_CONFIDENCE_F2P_BUG,
                        file_paths=[file_part] if file_part != "?" else [],
                        test_ids=nodeids[:8],
                        stage_name=stage_name,
                    )
                )
        else:
            # No per-test detail but any_f2p is true — surface the
            # candidate test paths as the F2P locus.
            for path in _norm_paths(f2p_summary.get("candidate_test_paths")):
                insights.append(
                    PersistedInsight(
                        insight_type=INSIGHT_TYPE_TESTGEN_F2P_BUG_PATTERN,
                        description=(
                            f"Test path `{path}` produced F2P transitions in a prior run."
                        ),
                        confidence=_CONFIDENCE_F2P_BUG,
                        file_paths=[path],
                        stage_name=stage_name,
                    )
                )

    # Mutation classes: bucket killed vs survived signatures by family.
    if isinstance(mutation_summary, dict):
        per_mutant = mutation_summary.get("per_mutant")
        killed_classes: dict[str, int] = {}
        survived_classes: dict[str, int] = {}
        if isinstance(per_mutant, list):
            for entry in per_mutant:
                if not isinstance(entry, dict):
                    continue
                operator = str(entry.get("operator") or "").strip()
                family = _operator_class_from_signature(operator + "@x:1")
                if not family:
                    continue
                status = str(entry.get("status") or "").lower()
                if status == "killed":
                    killed_classes[family] = killed_classes.get(family, 0) + 1
                elif status == "survived":
                    survived_classes[family] = survived_classes.get(family, 0) + 1
        # Killed mutant signatures sometimes only appear in the in-loop
        # MutationSensitivityFeedback shape via killed_mutant_signatures
        for sig in mutation_summary.get("killed_mutant_signatures") or []:
            family = _operator_class_from_signature(str(sig))
            if family:
                killed_classes[family] = killed_classes.get(family, 0) + 1
        for family, count in killed_classes.items():
            insights.append(
                PersistedInsight(
                    insight_type=INSIGHT_TYPE_TESTGEN_KILLED_MUTATION_CLASS,
                    description=(
                        f"Mutation class `{family}` killed by tests in a "
                        f"prior run ({count} mutant(s))."
                    ),
                    confidence=_CONFIDENCE_MUTATION_CLASS,
                    symbols=[family],
                    stage_name=stage_name,
                    support_count=max(1, count),
                )
            )
        for family, count in survived_classes.items():
            insights.append(
                PersistedInsight(
                    insight_type=INSIGHT_TYPE_TESTGEN_RESISTANT_MUTATION_CLASS,
                    description=(
                        f"Mutation class `{family}` SURVIVED tests in a "
                        f"prior run ({count} mutant(s)). Tighten assertions."
                    ),
                    confidence=_CONFIDENCE_MUTATION_CLASS,
                    symbols=[family],
                    stage_name=stage_name,
                    negative=True,
                    support_count=max(1, count),
                )
            )

    # Coverage hotspots: files where unexercised line ranges were
    # significant. We only surface files where >0 ranges were missing.
    if isinstance(coverage_gap_summary, dict):
        per_file = coverage_gap_summary.get("per_file_uncovered_ranges")
        if isinstance(per_file, dict):
            for path, ranges in per_file.items():
                if not ranges:
                    continue
                missing_lines = sum(
                    max(0, int(r[1]) - int(r[0]) + 1)
                    for r in ranges
                    if isinstance(r, (list, tuple)) and len(r) >= 2
                )
                if missing_lines == 0:
                    continue
                insights.append(
                    PersistedInsight(
                        insight_type=INSIGHT_TYPE_TESTGEN_LOW_COVERAGE_HOTSPOT,
                        description=(
                            f"`{path}` had {missing_lines} unexercised line(s) in a prior run."
                        ),
                        confidence=_CONFIDENCE_COVERAGE_HOTSPOT,
                        file_paths=[str(path)],
                        stage_name=stage_name,
                    )
                )

    # Contract-axis hotspots: the SWE-Bench Pro comparator reports both
    # per-target missing axes and task-level required-axis holes. Persist
    # them separately from raw line coverage: axis coverage is about missing
    # behavioral dimensions, not statement execution.
    if target_comparison:
        for entry in target_comparison:
            if not isinstance(entry, dict):
                continue
            target = str(entry.get("target") or "").strip()
            missing_axes = _norm_symbols(entry.get("missing_axes"))
            if not target or not missing_axes:
                continue
            insights.append(
                PersistedInsight(
                    insight_type=INSIGHT_TYPE_TESTGEN_AXIS_COVERAGE_HOTSPOT,
                    description=(
                        f"Prior generated tests missed contract axes for "
                        f"`{target}`: {', '.join(missing_axes)}. Add explicit "
                        "observable assertions for these dimensions."
                    ),
                    confidence=_CONFIDENCE_AXIS_COVERAGE,
                    file_paths=focus_paths,
                    symbols=[target, *missing_axes],
                    stage_name=stage_name,
                    negative=True,
                    support_count=max(1, len(missing_axes)),
                )
            )

    axis_summary = axis_coverage_summary
    if axis_summary is None and isinstance(coverage_gap_summary, dict):
        axis_summary = coverage_gap_summary
    if isinstance(axis_summary, dict):
        missing_required_axes = _norm_symbols(axis_summary.get("missing_required_axes"))
        if missing_required_axes:
            insights.append(
                PersistedInsight(
                    insight_type=INSIGHT_TYPE_TESTGEN_AXIS_COVERAGE_HOTSPOT,
                    description=(
                        "Prior generated tests missed required contract axes: "
                        f"{', '.join(missing_required_axes)}."
                    ),
                    confidence=_CONFIDENCE_AXIS_COVERAGE,
                    file_paths=focus_paths,
                    symbols=missing_required_axes,
                    stage_name=stage_name,
                    negative=True,
                    support_count=max(1, len(missing_required_axes)),
                )
            )

    return insights


def persist_testgen_insights_for_repo(
    *,
    repo_path: str,
    insights: Iterable[PersistedInsight],
    store: Optional[RepoMemoryStore] = None,
    directory: Optional[str] = None,
) -> dict[str, Any]:
    """Merge the supplied insights into the repo's RepoMemoryStore.

    Caller can pass a pre-constructed ``store`` (handy for tests) or
    let the helper build one. Returns the store's merge summary so
    callers can attach it to their own run report.
    """
    target = store or RepoMemoryStore(repo_path, directory=directory)
    insights_list = [i for i in insights if isinstance(i, PersistedInsight)]
    return target.merge_and_persist(insights_list)


def query_prior_testgen_insights_for_focus_files(
    *,
    repo_path: str,
    focus_files: Iterable[str],
    insight_types: Optional[Iterable[str]] = None,
    max_insights: int = 12,
    store: Optional[RepoMemoryStore] = None,
    directory: Optional[str] = None,
) -> list[PersistedInsight]:
    """Read prior cross-task testgen insights for the focus files.

    Returns a list ranked by (matches focus file, support_count,
    confidence) so the agent gets the most-supported, most-relevant
    priors first. Defensive about missing store / empty focus set.
    """
    focus_set = {str(f or "").strip() for f in focus_files if str(f or "").strip()}
    types_filter = set(str(t).upper() for t in (insight_types or [])) or set(
        ALL_TESTGEN_INSIGHT_TYPES
    )
    target = store or RepoMemoryStore(repo_path, directory=directory)
    try:
        all_insights = target.load()
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("Could not load repo memory: %s", exc)
        return []

    matched: list[tuple[int, int, float, PersistedInsight]] = []
    for insight in all_insights:
        if insight.insight_type.upper() not in types_filter:
            continue
        # Match score: 1 if any focus file overlaps, 0 otherwise.
        # Insights with no file_paths (e.g., mutation class) match
        # universally for this repo.
        if not insight.file_paths:
            match = 1
        else:
            match = 1 if (focus_set and any(p in focus_set for p in insight.file_paths)) else 0
        if match == 0 and focus_set:
            continue
        matched.append((match, int(insight.support_count), float(insight.confidence), insight))

    matched.sort(key=lambda t: (-t[0], -t[1], -t[2]))
    return [t[3] for t in matched[:max_insights]]


def render_prior_testgen_insights_prompt_block(
    insights: list[PersistedInsight],
) -> str:
    """Render prior cross-task testgen insights as a Markdown prompt
    section. Returns "" when the list is empty."""
    if not insights:
        return ""
    lines = [
        "## Prior cross-task testgen insights for this repo",
        "",
        (
            "These observations come from past APEX testgen runs on the "
            "SAME repo. They are not gold facts — confidence is decayed on "
            "load, and old beliefs may be stale — but they're a strong "
            "starting prior. Treat them as hints, not constraints:"
        ),
        "",
    ]
    for insight in insights:
        modifier = "(NEGATIVE) " if insight.negative else ""
        lines.append(
            f"  * {modifier}[{insight.insight_type}] "
            f"support={insight.support_count} "
            f"conf={insight.confidence:.2f}: {insight.description}"
        )
    return "\n".join(lines) + "\n"
