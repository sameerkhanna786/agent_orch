"""Closed-loop F2P-style feedback for the test_writer iteration loop.

Today the test_writer runs many iterations writing tests against the
broken (un-patched) worktree. Quick-verification runs inside each
iteration and reports per-test pass/fail on the broken state, but that
signal is not surfaced back to the agent in iteration N+1's prompt.
Result: the agent has no idea which of its tests are P2P-shaped
(passing on broken = useless) vs F2P-shaped (failing on broken = good).

This module turns the quick-verification payload into a structured
feedback object the next iteration's prompt can render.

Why this matters for SOTA:
    Tests that pass on the broken code cannot be fail-to-pass — by
    construction. A test_writer iteration loop that doesn't see this
    keeps writing P2P tests and the F2P oracle silently scores them
    as zero. The Apr 28-29 ansible smokes showed exactly this pattern:
    100% gold-target recall but 6-26% F2P rate.

Why this generalizes beyond benchmarks:
    The same logic applies to any TDD workflow. The "broken" state is
    just "the current code before the agent's fix" — there's no
    benchmark task object required. An IDE plugin, CI gate, or
    interactive agent can call this helper to give the model the
    same "your test is useless" signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Cap the number of nodeids we surface per category — the prompt has a
# finite budget and the agent only needs a few representative examples.
_MAX_REPORTED_NODEIDS = 8

# Canonical contract axes the test_writer is expected to cover.
# Mirrored from synthetic_test_analysis._CANONICAL_REQUIRED_AXES — kept
# inline here so the in-loop feedback path doesn't depend on the
# benchmark-shaped synthetic_test_analysis module.
CANONICAL_REQUIRED_AXES: tuple[str, ...] = (
    "positive_path",
    "missing_boundary",
    "negative_malformed",
    "multi_ordering",
)


@dataclass
class IterationFeedback:
    """Structured feedback derived from one test-writer iteration's
    quick-verification result."""

    useless_p2p_tests: list[str] = field(default_factory=list)
    likely_f2p_tests: list[str] = field(default_factory=list)
    infrastructure_failures: list[str] = field(default_factory=list)
    p2p_count: int = 0
    f2p_likely_count: int = 0
    infrastructure_failure_count: int = 0
    iteration_index: int = 0
    failure_classification_label: str = ""
    repair_hints: list[dict[str, str]] = field(default_factory=list)
    failure_excerpts: dict[str, str] = field(default_factory=dict)
    missing_modules: list[str] = field(default_factory=list)

    def is_actionable(self) -> bool:
        """Whether the feedback contains anything worth telling the agent.

        Empty feedback (no tests classified) shouldn't pollute the
        next iteration's prompt with a useless empty section.
        """
        return (
            self.p2p_count > 0
            or self.f2p_likely_count > 0
            or self.infrastructure_failure_count > 0
            or bool(self.repair_hints)
            or bool(self.failure_excerpts)
            or bool(self.missing_modules)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "useless_p2p_tests": list(self.useless_p2p_tests),
            "likely_f2p_tests": list(self.likely_f2p_tests),
            "infrastructure_failures": list(self.infrastructure_failures),
            "p2p_count": self.p2p_count,
            "f2p_likely_count": self.f2p_likely_count,
            "infrastructure_failure_count": self.infrastructure_failure_count,
            "iteration_index": self.iteration_index,
            "failure_classification_label": self.failure_classification_label,
            "repair_hints": [dict(item) for item in self.repair_hints],
            "failure_excerpts": dict(self.failure_excerpts),
            "missing_modules": list(self.missing_modules),
        }


@dataclass
class AxisCoverageFeedback:
    """Per-iteration coverage of the canonical contract axes.

    The test_writer agent is asked to declare ``contract_axes`` per
    artifact (positive_path, missing_boundary, negative_malformed,
    multi_ordering). Today the agent gets aggregate axis coverage in
    the post-rollout report but never per-iteration feedback. Surfacing
    "you covered X, still missing Y" between iterations directly steers
    the agent to fill gaps instead of accumulating more of the same.

    Cheap to derive: just walk the agent's own submission, no F2P or
    mutation oracle required. Generalizes outside benchmarks because
    it's rooted in what the agent declared, not in any gold suite.
    """

    covered_axes: list[str] = field(default_factory=list)
    missing_axes: list[str] = field(default_factory=list)
    artifact_count: int = 0
    iteration_index: int = 0

    def is_actionable(self) -> bool:
        """Skip the prompt block when there's nothing to nudge.

        Empty (no artifacts) and full (all axes covered) both render to
        nothing — the agent only needs the section when it has a
        specific axis to fill.
        """
        return bool(self.missing_axes) and self.artifact_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "covered_axes": list(self.covered_axes),
            "missing_axes": list(self.missing_axes),
            "artifact_count": self.artifact_count,
            "iteration_index": self.iteration_index,
        }


# Cross-rollout testgen insight types used by G.9 to broadcast / query
# the EpisodicMemoryBus. Naming convention: TESTGEN_* so the existing
# bus filters can include / exclude this surface cleanly.
INSIGHT_TYPE_TESTGEN_F2P_LIKELY = "TESTGEN_F2P_LIKELY"
INSIGHT_TYPE_TESTGEN_AXES_COVERED = "TESTGEN_AXES_COVERED"
INSIGHT_TYPE_TESTGEN_INSENSITIVE = "TESTGEN_INSENSITIVE"
# Phase I.2: positive-signal kills broadcast as `operator@path:line`
# strings in the symbols payload. Lets siblings see WHICH specific
# weakness classes are already covered and DIVERGE accordingly.
INSIGHT_TYPE_TESTGEN_MUTATION_KILLS = "TESTGEN_MUTATION_KILLS"


@dataclass
class CrossRolloutFeedback:
    """Per-iteration view of what OTHER parallel rollouts on the same
    task have discovered. Mined from the shared EpisodicMemoryBus.

    The feedback enables both (a) collaboration — agent picks up where
    a sibling rollout got traction — and (b) diversification — agent
    explicitly chooses a different angle when many siblings already
    converge on the same surface. Generalizes outside benchmarks: the
    EpisodicMemoryBus is shared across any parallel rollouts the
    orchestrator dispatches.
    """

    sibling_f2p_likely_paths: list[str] = field(default_factory=list)
    sibling_axes_covered: list[str] = field(default_factory=list)
    sibling_insensitive_paths: list[str] = field(default_factory=list)
    # Phase I.2: signatures of mutants siblings have already killed —
    # rendered as `operator@path:line` so the agent can target
    # uncovered weakness classes for diversity.
    sibling_killed_mutants: list[str] = field(default_factory=list)
    sibling_count: int = 0  # how many distinct other rollouts contributed
    iteration_index: int = 0

    def is_actionable(self) -> bool:
        return (
            bool(self.sibling_f2p_likely_paths)
            or bool(self.sibling_axes_covered)
            or bool(self.sibling_insensitive_paths)
            or bool(self.sibling_killed_mutants)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sibling_f2p_likely_paths": list(self.sibling_f2p_likely_paths),
            "sibling_axes_covered": list(self.sibling_axes_covered),
            "sibling_insensitive_paths": list(self.sibling_insensitive_paths),
            "sibling_killed_mutants": list(self.sibling_killed_mutants),
            "sibling_count": self.sibling_count,
            "iteration_index": self.iteration_index,
        }


def derive_cross_rollout_feedback(
    *,
    memory_bus: Any,
    rollout_id: int,
    iteration_index: int = 0,
    max_per_category: int = 6,
) -> CrossRolloutFeedback:
    """Query the shared EpisodicMemoryBus for testgen insights from
    OTHER rollouts and roll them up into a structured feedback object.

    Defensive about missing bus / missing methods — returns a non-
    actionable feedback so the prompt block doesn't render. The
    `min_confidence=0.5` filter matches the bus's default and excludes
    speculative discoveries from sibling rollouts.
    """
    feedback = CrossRolloutFeedback(iteration_index=iteration_index)
    if memory_bus is None or not hasattr(memory_bus, "query"):
        return feedback
    try:
        discoveries = memory_bus.query(
            exclude_rollout_id=rollout_id,
            insight_types=[
                INSIGHT_TYPE_TESTGEN_F2P_LIKELY,
                INSIGHT_TYPE_TESTGEN_AXES_COVERED,
                INSIGHT_TYPE_TESTGEN_INSENSITIVE,
                INSIGHT_TYPE_TESTGEN_MUTATION_KILLS,
            ],
            min_confidence=0.5,
            positive_only=False,
        )
    except Exception:  # pragma: no cover — defensive
        return feedback

    contributing_rollouts: set[int] = set()
    f2p_paths: list[str] = []
    axes: list[str] = []
    insensitive_paths: list[str] = []
    killed_mutants: list[str] = []
    seen_f2p: set[str] = set()
    seen_axes: set[str] = set()
    seen_insensitive: set[str] = set()
    seen_kills: set[str] = set()

    for d in discoveries:
        # Discovery has insight_type, file_paths, symbols, rollout_id
        contributing_rollouts.add(int(getattr(d, "rollout_id", 0)))
        itype = str(getattr(d, "insight_type", "") or "").upper()
        for path in list(getattr(d, "file_paths", []) or []):
            path_str = str(path or "").strip()
            if not path_str:
                continue
            if itype == INSIGHT_TYPE_TESTGEN_F2P_LIKELY and path_str not in seen_f2p:
                seen_f2p.add(path_str)
                f2p_paths.append(path_str)
            elif itype == INSIGHT_TYPE_TESTGEN_INSENSITIVE and path_str not in seen_insensitive:
                seen_insensitive.add(path_str)
                insensitive_paths.append(path_str)
        if itype == INSIGHT_TYPE_TESTGEN_AXES_COVERED:
            # Axes are surfaced in the symbols field
            for axis in list(getattr(d, "symbols", []) or []):
                axis_str = str(axis or "").strip().lower()
                if axis_str and axis_str not in seen_axes:
                    seen_axes.add(axis_str)
                    axes.append(axis_str)
        elif itype == INSIGHT_TYPE_TESTGEN_MUTATION_KILLS:
            # Phase I.2: mutant signatures `operator@path:line` are in symbols
            for sig in list(getattr(d, "symbols", []) or []):
                sig_str = str(sig or "").strip()
                if sig_str and sig_str not in seen_kills:
                    seen_kills.add(sig_str)
                    killed_mutants.append(sig_str)

    feedback.sibling_count = len(contributing_rollouts)
    feedback.sibling_f2p_likely_paths = f2p_paths[:max_per_category]
    feedback.sibling_axes_covered = sorted(axes)[:max_per_category]
    feedback.sibling_insensitive_paths = insensitive_paths[:max_per_category]
    feedback.sibling_killed_mutants = killed_mutants[:max_per_category]
    return feedback


def render_cross_rollout_prompt_block(
    feedback: CrossRolloutFeedback,
) -> str:
    """Render cross-rollout discoveries as a Markdown prompt section.

    Returns "" when not actionable. The block uses neutral framing —
    "siblings produced X" — so the agent can choose to converge or
    diverge based on what's most useful for THIS rollout's morph.
    """
    if not feedback.is_actionable():
        return ""
    lines = [
        "## Cross-rollout discoveries from siblings",
        "",
        (
            f"As of iteration {feedback.iteration_index}, "
            f"{feedback.sibling_count} other parallel rollout(s) on this "
            "same task have surfaced the following observations. Decide "
            "whether to CONVERGE (extend their work) or DIVERGE (try a "
            "different angle so the F2P-tuple selector has diverse "
            "candidates to choose from):"
        ),
        "",
    ]
    if feedback.sibling_f2p_likely_paths:
        lines.append(
            "Test paths siblings flagged as F2P-LIKELY (failed on broken — "
            "likely catching the bug):"
        )
        for path in feedback.sibling_f2p_likely_paths:
            lines.append(f"  * {path}")
    if feedback.sibling_axes_covered:
        lines.append(
            "Contract axes siblings have already covered: "
            + ", ".join(feedback.sibling_axes_covered)
            + ". Consider focusing on UN-covered axes for diversity."
        )
    if feedback.sibling_insensitive_paths:
        lines.append(
            "Test paths siblings flagged as INSENSITIVE (passed on broken AND on "
            "near-broken mutants — too loose to catch real bugs). Avoid this "
            "pattern in your portfolio:"
        )
        for path in feedback.sibling_insensitive_paths:
            lines.append(f"  * {path}")
    if feedback.sibling_killed_mutants:
        lines.append(
            "Mutation classes siblings have ALREADY KILLED (formatted "
            "`operator@path:line`). To diversify the rollout pool, target "
            "DIFFERENT weakness classes — e.g., if siblings cover "
            "boundary/conditional mutants, lean toward arithmetic, "
            "constant-replacement, or return-value mutants on this iteration:"
        )
        for sig in feedback.sibling_killed_mutants:
            lines.append(f"  * {sig}")
    return "\n".join(lines) + "\n"


def broadcast_iteration_testgen_insights(
    *,
    memory_bus: Any,
    rollout_id: int,
    iteration_index: int,
    f2p_feedback: "IterationFeedback | None",
    axis_feedback: "AxisCoverageFeedback | None",
    mutation_feedback: "MutationSensitivityFeedback | None",
) -> int:
    """Broadcast this rollout's iteration findings to the shared
    EpisodicMemoryBus so OTHER parallel rollouts can query them.

    Returns the count of insights broadcast. Defensive about missing
    bus / failed broadcasts — never raises into the iteration loop.
    """
    if memory_bus is None or not hasattr(memory_bus, "broadcast"):
        return 0
    broadcast_count = 0
    try:
        if f2p_feedback is not None and f2p_feedback.likely_f2p_tests:
            file_paths = sorted(
                {nodeid.partition("::")[0] for nodeid in f2p_feedback.likely_f2p_tests}
            )
            memory_bus.broadcast(
                rollout_id=rollout_id,
                insight_type=INSIGHT_TYPE_TESTGEN_F2P_LIKELY,
                description=(
                    f"rollout {rollout_id} iteration {iteration_index}: "
                    f"{f2p_feedback.f2p_likely_count} test(s) failed on broken "
                    "(F2P-shaped) at the listed paths"
                ),
                confidence=0.7,
                file_paths=file_paths[:6],
                stage_name="test_writer",
            )
            broadcast_count += 1
        if axis_feedback is not None and axis_feedback.covered_axes:
            memory_bus.broadcast(
                rollout_id=rollout_id,
                insight_type=INSIGHT_TYPE_TESTGEN_AXES_COVERED,
                description=(
                    f"rollout {rollout_id} iteration {iteration_index}: "
                    f"covered axes {','.join(axis_feedback.covered_axes)}"
                ),
                confidence=0.8,  # the agent's own axis declarations are reliable
                symbols=list(axis_feedback.covered_axes),
                stage_name="test_writer",
            )
            broadcast_count += 1
        if mutation_feedback is not None and mutation_feedback.insensitive_count > 0:
            # Insensitive tests flagged as a NEGATIVE pattern others should avoid
            memory_bus.broadcast(
                rollout_id=rollout_id,
                insight_type=INSIGHT_TYPE_TESTGEN_INSENSITIVE,
                description=(
                    f"rollout {rollout_id} iteration {iteration_index}: "
                    f"{mutation_feedback.insensitive_count} test(s) survived "
                    "near-broken mutations (too loose)"
                ),
                confidence=0.6,
                file_paths=list(mutation_feedback.target_source_paths)[:3],
                stage_name="test_writer",
                negative=True,
            )
            broadcast_count += 1
        # Phase I.2: positive-signal mutation kill broadcast.
        # Cap signatures at 12 so the bus doesn't accumulate noise on
        # high-killing rollouts; rollouts that beat the cap convey
        # plenty of signal already.
        if mutation_feedback is not None and mutation_feedback.killed_mutant_signatures:
            memory_bus.broadcast(
                rollout_id=rollout_id,
                insight_type=INSIGHT_TYPE_TESTGEN_MUTATION_KILLS,
                description=(
                    f"rollout {rollout_id} iteration {iteration_index}: "
                    f"killed {len(mutation_feedback.killed_mutant_signatures)} "
                    "mutant(s); siblings can DIVERGE toward uncovered "
                    "weakness classes"
                ),
                confidence=0.7,
                symbols=list(mutation_feedback.killed_mutant_signatures)[:12],
                file_paths=list(mutation_feedback.target_source_paths)[:3],
                stage_name="test_writer",
            )
            broadcast_count += 1
    except Exception as exc:  # pragma: no cover — defensive
        # Audit H12: don't silently lose memory-bus broadcast errors.
        # The rollout itself shouldn't fail because of a bus issue, but
        # the operator should see what went wrong.
        logger = __import__("logging").getLogger(__name__)
        logger.warning(
            "memory_bus broadcast failed for rollout (%s: %s); "
            "downstream candidates lose this insight",
            type(exc).__name__,
            exc,
        )
    return broadcast_count


@dataclass
class EdgePredictionFeedback:
    """Per-iteration accounting of predicted vs exercised bug edges.

    Phase G.3 added a structured ``predicted_edges`` field to the
    test_writer's submission. Each predicted edge declares the surface
    the agent thinks the gold patch likely changes (boundary, off_by_one,
    null_vs_empty, ...) and which of its test_artifact_paths exercise it.
    This helper counts predictions, counts predictions actually linked
    to a test artifact, and surfaces "unaddressed" edges so the next
    iteration's prompt knows to write tests for them.

    Generalizes outside benchmarks: predicted_edges is the agent's own
    structured reasoning over the issue text — no gold patch required.
    """

    predicted_count: int = 0
    exercised_count: int = 0  # predictions with at least one test_artifact_path
    unaddressed_edges: list[dict[str, Any]] = field(default_factory=list)
    iteration_index: int = 0

    def is_actionable(self) -> bool:
        """Skip when there's no signal: zero predictions or all
        addressed."""
        return bool(self.unaddressed_edges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_count": self.predicted_count,
            "exercised_count": self.exercised_count,
            "unaddressed_edges": list(self.unaddressed_edges),
            "iteration_index": self.iteration_index,
        }


def derive_edge_prediction_feedback(
    *,
    submission: dict[str, Any] | None,
    iteration_index: int = 0,
    max_unaddressed_reported: int = 4,
) -> EdgePredictionFeedback:
    """Walk the agent's submission and classify each predicted_edge as
    exercised (has a test_artifact_path) or unaddressed.

    Defensive about missing fields: a submission with no predicted_edges
    yields a non-actionable EdgePredictionFeedback (predicted_count=0)
    so the prompt block doesn't render — the agent isn't punished for
    older submissions that pre-date the schema field.
    """
    feedback = EdgePredictionFeedback(iteration_index=iteration_index)
    if not isinstance(submission, dict):
        return feedback
    predicted = submission.get("predicted_edges")
    if not isinstance(predicted, list) or not predicted:
        return feedback

    feedback.predicted_count = len(predicted)
    unaddressed: list[dict[str, Any]] = []
    for edge in predicted:
        if not isinstance(edge, dict):
            continue
        artifact_paths = edge.get("test_artifact_paths") or []
        # Filter to non-empty stripped strings — agents sometimes
        # populate the field with the empty list or [""] as a hedge.
        valid_paths = [str(p).strip() for p in artifact_paths if str(p or "").strip()]
        if valid_paths:
            feedback.exercised_count += 1
            continue
        unaddressed.append(
            {
                "edge_type": str(edge.get("edge_type") or "other"),
                "location": str(edge.get("location") or ""),
                "rationale": str(edge.get("rationale") or "")[:240],
            }
        )
    feedback.unaddressed_edges = unaddressed[:max_unaddressed_reported]
    return feedback


def render_edge_prediction_prompt_block(
    feedback: EdgePredictionFeedback,
) -> str:
    """Render edge-prediction feedback as a Markdown prompt section.

    Returns "" when not actionable so the prompt doesn't carry an empty
    section.
    """
    if not feedback.is_actionable():
        return ""
    lines = [
        "## Unaddressed bug-edge predictions from your previous iteration",
        "",
        (
            f"In iteration {feedback.iteration_index} you predicted "
            f"{feedback.predicted_count} bug edge(s). "
            f"{feedback.exercised_count} were exercised by tests in your "
            "portfolio. The following predictions are STILL UNADDRESSED — "
            "write at least one test per remaining edge in this iteration "
            "and fill its `test_artifact_paths` field:"
        ),
        "",
    ]
    for edge in feedback.unaddressed_edges:
        loc = f" at {edge['location']}" if edge.get("location") else ""
        lines.append(f"  * {edge['edge_type']}{loc}")
        if edge.get("rationale"):
            lines.append(f"      rationale: {edge['rationale']}")
    return "\n".join(lines) + "\n"


@dataclass
class MutationSensitivityFeedback:
    """Per-iteration mutation-sensitivity signal for the test_writer loop.

    Distinct from the post-rollout MutationReport because the in-loop
    setting has no gold patch — we mutate the broken worktree, and the
    "killed" count measures whether the agent's PASSING tests
    discriminate near-broken variants. Tests that still pass under
    mutation are too loose to plausibly catch real bugs and should be
    rewritten with stricter assertions.
    """

    sensitive_count: int = 0  # baseline-passing tests that flipped on mutation
    insensitive_count: int = 0  # baseline-passing tests still passing on mutation
    mutants_evaluated: int = 0
    sensitivity_score: float = 0.0  # killed / classified_mutants
    skip_reason: str = ""
    iteration_index: int = 0
    target_source_paths: list[str] = field(default_factory=list)
    # Phase I.2: signatures of mutants this rollout's tests killed
    # this iteration. Format: `operator@path:line`. Broadcast to the
    # cross-rollout bus so siblings can DIVERGE toward uncovered
    # weakness classes.
    killed_mutant_signatures: list[str] = field(default_factory=list)
    # Surviving mutants are direct obligations for the next iteration: write
    # or tighten tests until these mutant classes flip.
    survived_mutant_signatures: list[str] = field(default_factory=list)

    def is_actionable(self) -> bool:
        """Skip when there's no signal worth telling the agent.

        - Empty (0 mutants evaluated) → no information.
        - All mutants killed (perfect sensitivity) → nothing to nudge.
        - Skipped for cost reasons (skip_reason set) → don't pollute prompt.
        """
        if self.mutants_evaluated == 0 or self.skip_reason:
            return False
        # Only worth telling the agent when there's room to improve
        return self.insensitive_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensitive_count": self.sensitive_count,
            "insensitive_count": self.insensitive_count,
            "mutants_evaluated": self.mutants_evaluated,
            "sensitivity_score": round(self.sensitivity_score, 4),
            "skip_reason": self.skip_reason,
            "iteration_index": self.iteration_index,
            "target_source_paths": list(self.target_source_paths),
            "killed_mutant_signatures": list(self.killed_mutant_signatures),
            "survived_mutant_signatures": list(self.survived_mutant_signatures),
        }


def derive_mutation_sensitivity_feedback(
    *,
    mutation_report: Any,  # apex.evaluation.mutation_engine.MutationReport
    iteration_index: int = 0,
) -> MutationSensitivityFeedback:
    """Convert a MutationReport (from evaluate_mutation_sensitivity_in_loop)
    into iteration feedback the prompt builder can render.

    Defensive about the report shape — older callers may not have all
    fields populated. The is_actionable gate suppresses uninformative
    feedback (zero mutants evaluated, perfect sensitivity, skipped runs).
    """
    feedback = MutationSensitivityFeedback(iteration_index=iteration_index)
    if mutation_report is None:
        feedback.skip_reason = "no_report"
        return feedback
    feedback.target_source_paths = list(getattr(mutation_report, "source_paths", []) or [])
    feedback.mutants_evaluated = int(getattr(mutation_report, "killed", 0)) + int(
        getattr(mutation_report, "survived", 0)
    )
    feedback.sensitive_count = int(getattr(mutation_report, "killed", 0))
    feedback.insensitive_count = int(getattr(mutation_report, "survived", 0))
    feedback.sensitivity_score = float(getattr(mutation_report, "mutation_score", 0.0))
    # Phase I.2: extract killed mutant signatures so they can be
    # broadcast to sibling rollouts. Format `operator@path:line`
    # — globally unique across files, lets siblings see WHICH
    # weakness classes are already covered.
    killed_signatures: list[str] = []
    survived_signatures: list[str] = []
    for outcome in list(getattr(mutation_report, "per_mutant", []) or []):
        status = str(getattr(outcome, "status", "") or "").lower()
        if status not in {"killed", "survived"}:
            continue
        mutant = getattr(outcome, "mutant", None)
        if mutant is None:
            continue
        operator = str(getattr(mutant, "operator", "") or "").strip()
        source_path = str(getattr(mutant, "source_path", "") or "").strip()
        line = int(getattr(mutant, "line", 0) or 0)
        if operator and source_path and line:
            signature = f"{operator}@{source_path}:{line}"
            if status == "killed":
                killed_signatures.append(signature)
            else:
                survived_signatures.append(signature)
    feedback.killed_mutant_signatures = killed_signatures
    feedback.survived_mutant_signatures = survived_signatures
    baseline_status = str(getattr(mutation_report, "baseline_status", "") or "")
    if baseline_status in {
        "no_target_source_paths",
        "no_mutants_generated",
        "no_mutants",
        "no_baseline_passing_tests",
    }:
        feedback.skip_reason = baseline_status
    return feedback


def render_mutation_sensitivity_prompt_block(
    feedback: MutationSensitivityFeedback,
) -> str:
    """Render mutation-sensitivity feedback as a Markdown prompt section.

    Returns "" when the feedback is not actionable so the prompt doesn't
    carry an empty section.
    """
    if not feedback.is_actionable():
        return ""
    lines = [
        "## Mutation sensitivity from your previous iteration",
        "",
        (
            f"In iteration {feedback.iteration_index} we applied "
            f"{feedback.mutants_evaluated} small mutation(s) to "
            f"{', '.join(feedback.target_source_paths) or 'the source under test'} "
            "and re-ran your tests. Of the tests that PASSED on the "
            "unmutated worktree:"
        ),
        f"  * {feedback.sensitive_count} flipped (good — sensitive enough to discriminate)",
        f"  * {feedback.insensitive_count} still passed (weak — too loose to catch real bugs)",
        "",
        (
            "INSENSITIVE tests are usually shaped like 'function does not "
            "raise' or 'returns SOMETHING' rather than 'returns the EXACT "
            "expected value'. Tighten your assertions: prefer strict "
            "equality, exact return-shape checks, and explicit boundary "
            "values over presence-only checks. A test that passes on a "
            "near-broken variant cannot reliably catch the real bug either."
        ),
    ]
    if feedback.survived_mutant_signatures:
        lines.extend(
            [
                "",
                "Surviving mutant obligations to kill next:",
                *[f"  * `{signature}`" for signature in feedback.survived_mutant_signatures[:8]],
            ]
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Phase I.3 — Coverage-driven targeting
# ---------------------------------------------------------------------------


# Cap how many uncovered ranges we surface per file so the prompt stays
# readable. Files with > _MAX_RANGES_PER_FILE gaps are likely entirely
# uncovered and the agent gets the message from the first few entries.
_MAX_COVERAGE_RANGES_PER_FILE = 8
# And cap the number of files we surface — focus files are already
# capped to 3 in the engine, but defend against larger callers.
_MAX_COVERAGE_FILES = 8


@dataclass
class CoverageGapFeedback:
    """Per-iteration coverage-gap signal for the test_writer loop.

    Tells the agent WHICH lines of the focus file(s) the current test
    portfolio does not exercise. Unexercised lines are the strongest
    targeting signal we have: a bug behind an unexercised branch is
    invisible to any test in the portfolio, no matter how high the
    F2P or mutation-sensitivity scores climb. Generalizes outside
    benchmarks because it depends only on test paths + focus files,
    not on a benchmark task object.
    """

    target_source_paths: list[str] = field(default_factory=list)
    per_file_uncovered_ranges: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    per_file_missing_branches: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    missing_target_source_paths: list[str] = field(default_factory=list)
    per_file_total_lines: dict[str, int] = field(default_factory=dict)
    per_file_total_branches: dict[str, int] = field(default_factory=dict)
    overall_coverage_ratio: float = 0.0
    overall_branch_coverage_ratio: float = 0.0
    skip_reason: str = ""
    iteration_index: int = 0

    def is_actionable(self) -> bool:
        """Skip when there's no signal worth telling the agent.

        - Empty (no files reported) → no information.
        - 100% coverage → nothing to target.
        - skip_reason set (no_coverage_tool, no_test_paths, etc.) →
          don't pollute prompt with "we couldn't measure coverage."
        """
        if self.skip_reason:
            return False
        if (
            not self.per_file_uncovered_ranges
            and not self.per_file_missing_branches
            and not self.missing_target_source_paths
        ):
            return False
        return (
            any(ranges for ranges in self.per_file_uncovered_ranges.values())
            or any(branches for branches in self.per_file_missing_branches.values())
            or bool(self.missing_target_source_paths)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_source_paths": list(self.target_source_paths),
            "per_file_uncovered_ranges": {
                path: [list(r) for r in ranges]
                for path, ranges in self.per_file_uncovered_ranges.items()
            },
            "per_file_missing_branches": {
                path: [list(branch) for branch in branches]
                for path, branches in self.per_file_missing_branches.items()
            },
            "missing_target_source_paths": list(self.missing_target_source_paths),
            "per_file_total_lines": dict(self.per_file_total_lines),
            "per_file_total_branches": dict(self.per_file_total_branches),
            "overall_coverage_ratio": round(self.overall_coverage_ratio, 4),
            "overall_branch_coverage_ratio": round(
                self.overall_branch_coverage_ratio,
                4,
            ),
            "skip_reason": self.skip_reason,
            "iteration_index": self.iteration_index,
        }


def derive_coverage_gap_feedback(
    *,
    coverage_report: Any,  # apex.evaluation.coverage_engine.CoverageReport
    iteration_index: int = 0,
) -> CoverageGapFeedback:
    """Convert a CoverageReport (from evaluate_coverage_in_loop) into
    iteration feedback the prompt builder can render.

    Defensive about the report shape — older callers may not have all
    fields populated. Coverage-tool / no-tests skips are surfaced as
    skip_reason so the prompt block can opt out cleanly.
    """
    feedback = CoverageGapFeedback(iteration_index=iteration_index)
    if coverage_report is None:
        feedback.skip_reason = "no_report"
        return feedback
    feedback.target_source_paths = list(getattr(coverage_report, "target_source_paths", []) or [])
    status = str(getattr(coverage_report, "status", "") or "")
    if status in {
        "no_target_source_paths",
        "no_test_paths",
        "no_coverage_tool",
        "no_coverage_data",
        "timeout",
        "exception",
    }:
        feedback.skip_reason = status
        return feedback
    raw_ranges = getattr(coverage_report, "per_file_uncovered_ranges", {}) or {}
    raw_branches = getattr(coverage_report, "per_file_missing_branches", {}) or {}
    # Normalize: cap per-file ranges and total file count, ensure each
    # range is a (start, end) tuple of ints.
    capped_files: dict[str, list[tuple[int, int]]] = {}
    for path, ranges in list(raw_ranges.items())[:_MAX_COVERAGE_FILES]:
        normalized: list[tuple[int, int]] = []
        for r in list(ranges)[:_MAX_COVERAGE_RANGES_PER_FILE]:
            try:
                start, end = int(r[0]), int(r[1])
            except (TypeError, ValueError, IndexError):
                continue
            if start > 0 and end >= start:
                normalized.append((start, end))
        if normalized:
            capped_files[str(path)] = normalized
    feedback.per_file_uncovered_ranges = capped_files
    capped_branches: dict[str, list[tuple[int, int]]] = {}
    for path, branches in list(raw_branches.items())[:_MAX_COVERAGE_FILES]:
        normalized_branches: list[tuple[int, int]] = []
        for branch in list(branches)[:_MAX_COVERAGE_RANGES_PER_FILE]:
            try:
                source_line, target_line = int(branch[0]), int(branch[1])
            except (TypeError, ValueError, IndexError):
                continue
            if source_line > 0 and target_line > 0:
                normalized_branches.append((source_line, target_line))
        if normalized_branches:
            capped_branches[str(path)] = normalized_branches
    feedback.per_file_missing_branches = capped_branches
    feedback.missing_target_source_paths = list(
        getattr(coverage_report, "missing_target_source_paths", []) or []
    )[:_MAX_COVERAGE_FILES]
    feedback.per_file_total_lines = dict(getattr(coverage_report, "per_file_total_lines", {}) or {})
    feedback.per_file_total_branches = dict(
        getattr(coverage_report, "per_file_total_branches", {}) or {}
    )
    feedback.overall_coverage_ratio = float(
        getattr(coverage_report, "overall_coverage_ratio", 0.0) or 0.0
    )
    raw_branch_ratio = getattr(coverage_report, "overall_branch_coverage_ratio", None)
    feedback.overall_branch_coverage_ratio = (
        float(raw_branch_ratio) if isinstance(raw_branch_ratio, (int, float)) else 1.0
    )
    return feedback


def render_coverage_gap_prompt_block(feedback: CoverageGapFeedback) -> str:
    """Render coverage-gap feedback as a Markdown prompt section.

    Returns "" when the feedback is not actionable so the prompt
    doesn't carry an empty section. Lines are formatted as inclusive
    ranges (e.g., 42-58, 91, 117-122) so the agent can target them
    directly without parsing offset arithmetic.
    """
    if not feedback.is_actionable():
        return ""
    lines = [
        "## Coverage gaps NOT exercised by your current tests",
        "",
        (
            f"After iteration {feedback.iteration_index} your test "
            "portfolio still misses the following lines or branch arcs "
            f"of the focus source file(s) (line coverage: "
            f"{feedback.overall_coverage_ratio * 100:.1f}%; branch coverage: "
            f"{feedback.overall_branch_coverage_ratio * 100:.1f}%). Each "
            "unexercised branch is invisible to your tests — a bug there "
            "would pass quietly. Add at least one test that drives execution "
            "into each gap below:"
        ),
        "",
    ]
    for path, ranges in feedback.per_file_uncovered_ranges.items():
        formatted = ", ".join(
            f"{start}" if start == end else f"{start}-{end}" for start, end in ranges
        )
        lines.append(f"  * `{path}` lines: {formatted}")
    if feedback.missing_target_source_paths:
        formatted_targets = ", ".join(f"`{path}`" for path in feedback.missing_target_source_paths)
        lines.append(
            f"  * Target file(s) never imported or reported by coverage: {formatted_targets}"
        )
    for path, branches in feedback.per_file_missing_branches.items():
        formatted = ", ".join(f"{source}->{target}" for source, target in branches)
        lines.append(f"  * `{path}` missing branches: {formatted}")
    return "\n".join(lines) + "\n"


@dataclass
class GeneratedTestQualityFeedback:
    artifact_count: int = 0
    weak_artifact_count: int = 0
    issue_count: int = 0
    issue_counts: dict[str, int] = field(default_factory=dict)
    mean_assertion_effect_score: float = 0.0
    iteration_index: int = 0

    def is_actionable(self) -> bool:
        return self.weak_artifact_count > 0 or self.issue_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_count": self.artifact_count,
            "weak_artifact_count": self.weak_artifact_count,
            "issue_count": self.issue_count,
            "issue_counts": dict(self.issue_counts),
            "mean_assertion_effect_score": round(
                self.mean_assertion_effect_score,
                4,
            ),
            "iteration_index": self.iteration_index,
        }


def render_generated_test_quality_prompt_block(
    feedback: GeneratedTestQualityFeedback,
) -> str:
    if not feedback.is_actionable():
        return ""
    lines = [
        "## Weak generated-test oracle feedback",
        "",
        (
            f"In iteration {feedback.iteration_index}, "
            f"{feedback.weak_artifact_count}/{feedback.artifact_count} generated "
            "test artifact(s) had weak oracle signals. Passing weak tests do not "
            "count as useful coverage; rewrite them with exact behavioral assertions."
        ),
        "",
    ]
    if feedback.issue_counts:
        lines.append("Issue types observed:")
        for code, count in sorted(
            feedback.issue_counts.items(),
            key=lambda item: (-int(item[1]), item[0]),
        )[:6]:
            lines.append(f"  * {code}: {count}")
    lines.append(
        "Avoid `assert True`, self-comparisons, broad `Exception` oracles, "
        "and presence-only checks unless paired with exact value/shape assertions."
    )
    return "\n".join(lines) + "\n"


@dataclass
class TestStabilityFeedback:
    status: str = ""
    run_count: int = 0
    failed_run_count: int = 0
    flaky_nodeids: list[str] = field(default_factory=list)
    iteration_index: int = 0

    def is_actionable(self) -> bool:
        return self.failed_run_count > 0 or bool(self.flaky_nodeids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "run_count": self.run_count,
            "failed_run_count": self.failed_run_count,
            "flaky_nodeids": list(self.flaky_nodeids),
            "iteration_index": self.iteration_index,
        }


def render_test_stability_prompt_block(feedback: TestStabilityFeedback) -> str:
    if not feedback.is_actionable():
        return ""
    lines = [
        "## Generated-test stability feedback",
        "",
        (
            f"In iteration {feedback.iteration_index}, the generated suite was "
            f"rerun {feedback.run_count} time(s). "
            f"{feedback.failed_run_count} run(s) failed or timed out."
        ),
    ]
    if feedback.flaky_nodeids:
        lines.append("Flaky nodeids with inconsistent statuses:")
        for nodeid in feedback.flaky_nodeids[:8]:
            lines.append(f"  * {nodeid}")
    lines.append(
        "Remove randomness, wall-clock assumptions, network calls, shared global "
        "state, and order dependence before adding more test cases."
    )
    return "\n".join(lines) + "\n"


@dataclass
class AssertionMutationFeedback:
    status: str = ""
    mutated_assertion_count: int = 0
    survived: bool = False
    assertion_effective: bool = False
    test_paths: list[str] = field(default_factory=list)
    iteration_index: int = 0

    def is_actionable(self) -> bool:
        if self.survived:
            return True
        return self.status in {"no_assertions_mutated", "baseline_no_tests_collected"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mutated_assertion_count": self.mutated_assertion_count,
            "survived": self.survived,
            "assertion_effective": self.assertion_effective,
            "test_paths": list(self.test_paths),
            "iteration_index": self.iteration_index,
        }


def render_assertion_mutation_prompt_block(
    feedback: AssertionMutationFeedback,
) -> str:
    if not feedback.is_actionable():
        return ""
    lines = [
        "## Assertion-effect feedback",
        "",
    ]
    if feedback.survived:
        lines.append(
            f"In iteration {feedback.iteration_index}, Apex inverted "
            f"{feedback.mutated_assertion_count} assertion(s) and the generated "
            "suite still passed. Those assertions are not proving the behavior."
        )
    else:
        lines.append(
            f"In iteration {feedback.iteration_index}, Apex could not mutate any "
            "effective assertions in the generated suite."
        )
    if feedback.test_paths:
        lines.append("Affected generated tests:")
        for path in feedback.test_paths[:8]:
            lines.append(f"  * `{path}`")
    lines.append(
        "Rewrite these tests with strict observable assertions: exact return "
        "values, persisted state, emitted events, response payload shape, or a "
        "specific exception type and message pattern."
    )
    return "\n".join(lines) + "\n"


def derive_axis_coverage_feedback(
    *,
    test_artifacts: list[dict[str, Any]] | None,
    iteration_index: int = 0,
    required_axes: tuple[str, ...] = CANONICAL_REQUIRED_AXES,
) -> AxisCoverageFeedback:
    """Walk the agent's own test artifacts to compute per-iteration
    contract-axis coverage.

    The agent declares ``contract_axes`` (a list of axis names) per
    artifact. We aggregate across artifacts and compare against the
    canonical four required axes. Missing axes are surfaced to the
    next iteration's prompt as a "still missing X — write a test that
    exercises X" nudge.
    """
    feedback = AxisCoverageFeedback(iteration_index=iteration_index)
    artifacts = list(test_artifacts or [])
    feedback.artifact_count = len(artifacts)

    declared_axes: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        for raw_axis in artifact.get("contract_axes") or []:
            axis = str(raw_axis or "").strip().lower()
            if axis:
                declared_axes.add(axis)

    required_set = {axis.lower() for axis in required_axes}
    feedback.covered_axes = sorted(required_set & declared_axes)
    feedback.missing_axes = sorted(required_set - declared_axes)
    return feedback


def render_axis_coverage_prompt_block(feedback: AxisCoverageFeedback) -> str:
    """Render axis-coverage feedback as a Markdown prompt section.

    Returns "" when not actionable so the prompt doesn't carry an
    empty section.
    """
    if not feedback.is_actionable():
        return ""
    lines = [
        "## Contract-axis coverage from your previous iteration",
        "",
        (
            f"Across the {feedback.artifact_count} test artifact(s) you "
            f"declared in iteration {feedback.iteration_index}, the "
            "canonical contract axes covered were: "
            + (", ".join(feedback.covered_axes) or "(none)")
            + "."
        ),
        "",
        (
            "STILL MISSING: "
            + ", ".join(feedback.missing_axes)
            + ". Add at least one test per missing axis in this iteration. "
            "Recall the four required axes:"
        ),
        "  * positive_path: canonical happy-path acceptance test",
        "  * missing_boundary: empty / null / default / boundary input",
        "  * negative_malformed: error handling for invalid input",
        "  * multi_ordering: collection / ordering / multi-element behavior",
        (
            "Each artifact you submit MUST list the axes it covers in its "
            "`contract_axes` field — and only the axes it actually exercises."
        ),
    ]
    return "\n".join(lines) + "\n"


def classify_iteration_feedback(
    *,
    quick_verification_payload: dict[str, Any] | None,
    candidate_test_paths: list[str] | None,
    iteration_index: int = 0,
) -> IterationFeedback:
    """Convert a quick_verification dict into iteration feedback.

    Args:
        quick_verification_payload: The dict stashed at
            ``stage_submission["_apex_quick_verification"]`` by the
            test_writer's quick-verification step. May be None or empty
            (early iterations, disabled quick-verification).
        candidate_test_paths: Repo-relative paths of the tests THIS
            agent generated. Used to filter out passes/fails on tests
            that already lived in the repo.
        iteration_index: Which iteration this feedback is from.
            Surfaced on the result so the prompt can frame "in your
            previous attempt..."

    Returns an :class:`IterationFeedback`. Defensive about missing
    fields — older quick_verification payloads (pre-Phase B) just
    surface zero classifications.
    """
    feedback = IterationFeedback(iteration_index=iteration_index)
    if not quick_verification_payload:
        return feedback

    candidate_set = {str(p).strip() for p in (candidate_test_paths or []) if str(p or "").strip()}

    passed_tests = list(quick_verification_payload.get("passed_tests") or [])
    failed_tests = list(quick_verification_payload.get("failed_tests") or [])

    # Tests that PASSED on the broken (un-patched) worktree are P2P
    # candidates: by construction they cannot be F2P. Filter to the
    # agent's own tests if we have the candidate set, otherwise report
    # all (some legacy callers don't pass the path filter).
    for nodeid in passed_tests:
        if candidate_set and not _nodeid_matches_paths(str(nodeid), candidate_set):
            continue
        feedback.useless_p2p_tests.append(str(nodeid))
    feedback.p2p_count = len(feedback.useless_p2p_tests)
    feedback.useless_p2p_tests = feedback.useless_p2p_tests[:_MAX_REPORTED_NODEIDS]

    # Tests that FAILED on the broken worktree are F2P candidates —
    # they at least have the right "fail on broken" half of the
    # transition. The "pass on fixed" half can only be confirmed
    # post-patch by the F2P oracle, but the broken-side signal is
    # still useful encouragement for the next iteration.
    for nodeid in failed_tests:
        if candidate_set and not _nodeid_matches_paths(str(nodeid), candidate_set):
            continue
        feedback.likely_f2p_tests.append(str(nodeid))
    feedback.f2p_likely_count = len(feedback.likely_f2p_tests)
    feedback.likely_f2p_tests = feedback.likely_f2p_tests[:_MAX_REPORTED_NODEIDS]

    # Infrastructure failures (env / syntax / collection per the
    # Phase B.3 classifier) need to be fixed BEFORE the agent
    # iterates on test logic — surface them prominently.
    classification = dict(quick_verification_payload.get("failure_classification") or {})
    label = str(classification.get("label") or "").strip()
    feedback.failure_classification_label = label
    if label in {"env", "syntax", "collection"}:
        # The classifier's primary_signal often names a file/test;
        # we don't have a structured failing-test list here so we
        # surface the label + signal as a single "fix-this-first" entry.
        signal = str(classification.get("primary_signal") or "").strip()
        feedback.infrastructure_failures.append(f"{label}: {signal}" if signal else label)
        feedback.infrastructure_failure_count = 1

    return feedback


def classify_dual_state_f2p_feedback(
    *,
    f2p_payload: dict[str, Any],
    iteration_index: int = 0,
    max_excerpt_chars: int = 1200,
) -> IterationFeedback:
    """Convert real broken/fixed F2P output into next-prompt feedback.

    Quick-verification only knows whether tests pass or fail on the broken
    checkout. This path consumes the dual-state oracle payload, so P2P/F2F/P2F
    classifications, missing modules, and failure excerpts can be fed into the
    next generation prompt as concrete repair instructions.
    """
    summary = dict((f2p_payload or {}).get("summary") or {})
    repair = dict((f2p_payload or {}).get("repair_feedback") or {})
    transitions = dict((f2p_payload or {}).get("transitions") or {})
    feedback = IterationFeedback(iteration_index=iteration_index)

    for nodeid, info in transitions.items():
        if not isinstance(info, dict):
            continue
        kind = str(info.get("kind") or "").lower()
        if kind == "p2p":
            feedback.useless_p2p_tests.append(str(nodeid))
        elif kind == "f2p":
            feedback.likely_f2p_tests.append(str(nodeid))
    feedback.p2p_count = int(summary.get("p2p_count") or len(feedback.useless_p2p_tests))
    feedback.f2p_likely_count = int(summary.get("f2p_count") or len(feedback.likely_f2p_tests))
    feedback.useless_p2p_tests = feedback.useless_p2p_tests[:_MAX_REPORTED_NODEIDS]
    feedback.likely_f2p_tests = feedback.likely_f2p_tests[:_MAX_REPORTED_NODEIDS]

    failure_classes = [
        str(item).strip()
        for item in list(repair.get("failure_classes") or summary.get("failure_classes") or [])
        if str(item).strip()
    ]
    feedback.failure_classification_label = ",".join(failure_classes)
    infra_classes = {
        "no_tests_collected",
        "module_not_found",
        "f2f",
        "runner_error",
        "install_failed",
    }
    for code in failure_classes:
        if code in infra_classes:
            feedback.infrastructure_failures.append(code)

    missing_modules = [
        str(item).strip()
        for item in list(repair.get("missing_modules") or summary.get("missing_modules") or [])
        if str(item).strip()
    ]
    feedback.missing_modules = missing_modules[:_MAX_REPORTED_NODEIDS]
    if missing_modules and "module_not_found" not in feedback.infrastructure_failures:
        feedback.infrastructure_failures.append("module_not_found")
    feedback.infrastructure_failure_count = len(feedback.infrastructure_failures)

    hints: list[dict[str, str]] = []
    for item in list(repair.get("repair_hints") or summary.get("repair_hints") or []):
        if not isinstance(item, dict):
            continue
        hint = {
            "class": str(item.get("class") or "").strip(),
            "policy": str(item.get("policy") or "").strip(),
            "action": str(item.get("action") or "").strip(),
        }
        if any(hint.values()):
            hints.append(hint)
    feedback.repair_hints = hints[:_MAX_REPORTED_NODEIDS]

    excerpts = dict(repair.get("failure_excerpts") or summary.get("failure_excerpts") or {})
    for side in ("broken", "fixed"):
        excerpt = str(excerpts.get(side) or "").strip()
        if excerpt:
            feedback.failure_excerpts[side] = excerpt[-max_excerpt_chars:]

    return feedback


def render_iteration_feedback_prompt_block(
    feedback: IterationFeedback,
) -> str:
    """Render an :class:`IterationFeedback` as a Markdown prompt section.

    Returns "" when the feedback is empty (no tests classified) so
    the prompt doesn't carry a useless header.
    """
    if not feedback.is_actionable():
        return ""

    lines = [
        "## Feedback from your previous test-writer iteration",
        "",
        f"In iteration {feedback.iteration_index} you wrote tests against the "
        "broken (un-patched) code. Quick-verification observed:",
        "",
    ]
    if feedback.infrastructure_failure_count > 0:
        lines.append(
            f"- INFRASTRUCTURE FAILURE ({feedback.failure_classification_label}): "
            "fix this BEFORE iterating on test logic. The classifier flagged "
            f"{feedback.infrastructure_failures[0]}."
        )
        if feedback.missing_modules:
            lines.append(
                "- Missing modules/import aliases observed: " + ", ".join(feedback.missing_modules)
            )
    for hint in feedback.repair_hints[:4]:
        policy = str(hint.get("policy") or "").strip()
        action = str(hint.get("action") or "").strip()
        code = str(hint.get("class") or "").strip()
        if policy or action:
            lines.append(f"- Repair policy {code or 'measured'}: {policy} {action}".strip())
    if feedback.p2p_count > 0:
        lines.append(
            f"- {feedback.p2p_count} of your tests PASSED on broken code. These are P2P "
            "(useless) — they cannot be F2P by construction. Rewrite each to "
            "exercise the contract change so it FAILS on broken and would "
            "PASS only after the fix:"
        )
        for nodeid in feedback.useless_p2p_tests:
            lines.append(f"  * {nodeid}")
    if feedback.f2p_likely_count > 0:
        lines.append(
            f"- {feedback.f2p_likely_count} of your tests FAILED on broken code "
            "(good — they have the F2P shape). Keep these and add the missing "
            "contract axes (positive_path / missing_boundary / "
            "negative_malformed / multi_ordering)."
        )
        for nodeid in feedback.likely_f2p_tests[:3]:
            lines.append(f"  * {nodeid}")
    if feedback.failure_excerpts:
        lines.append("- Concrete runner excerpts to repair against:")
        for side in ("fixed", "broken"):
            excerpt = str(feedback.failure_excerpts.get(side) or "").strip()
            if excerpt:
                lines.append(f"  * {side}: {excerpt}")
    return "\n".join(lines) + "\n"


def _nodeid_matches_paths(nodeid: str, candidate_paths: set[str]) -> bool:
    """Match a pytest-style nodeid against the agent's test file paths.

    pytest nodeids look like ``tests/test_foo.py::test_bar[case-1]``.
    The agent's candidate paths are file paths only (``tests/test_foo.py``).
    We extract the file portion of the nodeid and check membership.
    """
    if not nodeid:
        return False
    file_part, _, _ = nodeid.partition("::")
    file_part = file_part.strip()
    if not file_part:
        return False
    return file_part in candidate_paths or any(
        file_part.endswith(path) or path.endswith(file_part) for path in candidate_paths
    )
