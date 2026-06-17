"""Mutation-targeted active learning loop (Phase 6 item 6.4).

After each test-suite candidate is selected, run mutation testing. For
mutants that SURVIVED, generate targeted tests in the next iteration
that specifically attack them (the LLM gets the mutant's diff and an
instruction "write a test that fails on the mutant but passes on the
original"). The loop terminates when the mutation kill rate stabilises,
when ``max_iterations`` is hit, or when every mutant is killed.

The active learner DOES NOT re-implement mutation testing — it
consumes :func:`apex.evaluation.mutation_engine.evaluate_mutation_score`
and the existing testgen pipeline (via a pluggable callable). Callers
that don't want a real LLM in the loop can pass a fake
``targeted_test_generator`` that returns canned tests.

Stopping criteria
-----------------

A naive "stop when all mutants killed" loop can run forever on a long
tail of unkillable equivalent mutants. We add two safety nets:

* ``max_iterations`` — hard cap (default 3).
* ``min_kill_rate_improvement`` — if the kill rate didn't improve by
  at least this many absolute percentage points in the last iteration,
  stop. Default 0.02 (2 percentage points). This catches the case
  where the new tests don't kill any new mutants and we're just
  burning LLM budget.

The loop always runs at least one iteration so the caller gets at
least one targeted-test attempt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# Hard cap so a misconfigured caller cannot loop forever even if it
# overrides max_iterations.
_ABSOLUTE_MAX_ITERATIONS = 10

# Default minimum kill-rate improvement to keep iterating, in absolute
# units (0.02 = 2 percentage points).
DEFAULT_MIN_KILL_RATE_IMPROVEMENT = 0.02

# Default number of surviving mutants to attack per iteration. Capping
# this keeps the LLM budget per iteration bounded; the next iteration
# attacks the next batch.
DEFAULT_TOP_K_MUTANTS_PER_ITERATION = 8

# Phase B.6 (Decisive-Edge): cap default max_iterations to 2. Prior
# default of 3 plus the absolute cap of 10 wasted LLM budget on long
# tails of equivalent mutants. Operators who want more iterations can
# pass an explicit ``max_iterations`` argument.
DEFAULT_MAX_ITERATIONS = 2

# Phase B.6: dedicated targeted-test prompt template lives next to this
# module. Loaded once at import time and cached so the active learner
# doesn't re-read the file on every iteration. The template uses
# ``{{placeholder}}`` markers so callers can do a simple ``str.replace``
# rather than risking format-string injection from mutated source.
_TARGETED_PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent / "prompts" / "mutation_attack.txt"


def _load_targeted_prompt_template() -> str:
    """Read the mutation-attack prompt template from disk, once.

    Wrapped so unit tests can monkey-patch the path or reset the cache.
    Falls back to an embedded minimal template if the file is missing
    so a packaging miss can't crash the loop in production.
    """
    try:
        text = _TARGETED_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
        if text.strip():
            return text
    except OSError as exc:
        logger.warning(
            "Could not read mutation-attack prompt template at %s (%s: %s); "
            "falling back to embedded default.",
            _TARGETED_PROMPT_TEMPLATE_PATH,
            type(exc).__name__,
            exc,
        )
    return _DEFAULT_EMBEDDED_TARGETED_PROMPT


_DEFAULT_EMBEDDED_TARGETED_PROMPT = (
    "Write ONE pytest-compatible test function targeting the surviving mutant.\n"
    "Original snippet:\n{{original_snippet}}\n"
    "Mutated snippet:\n{{mutated_snippet}}\n"
    "Mutation operator: {{operator}} at {{source_path}}:{{line}}.\n"
    "Existing tests (avoid duplicates):\n{{existing_test_summaries}}\n"
    "Return only the test function code."
)


# Cache for the loaded template — populated lazily so monkey-patching
# the path in tests works as long as the test happens before the first
# real call.
_CACHED_TARGETED_PROMPT_TEMPLATE: Optional[str] = None


def get_targeted_prompt_template() -> str:
    """Return the cached prompt template, loading on first access.

    Exposed publicly so callers wiring a real LLM-backed
    ``targeted_test_generator`` can render the template themselves
    rather than re-reading the file.
    """
    global _CACHED_TARGETED_PROMPT_TEMPLATE
    if _CACHED_TARGETED_PROMPT_TEMPLATE is None:
        _CACHED_TARGETED_PROMPT_TEMPLATE = _load_targeted_prompt_template()
    return _CACHED_TARGETED_PROMPT_TEMPLATE


def reset_targeted_prompt_template_cache() -> None:
    """Reset the cache so a subsequent call re-reads from disk.

    Used by unit tests that monkey-patch the template path.
    """
    global _CACHED_TARGETED_PROMPT_TEMPLATE
    _CACHED_TARGETED_PROMPT_TEMPLATE = None


def render_targeted_prompt(
    *,
    operator: str,
    source_path: str,
    line: int,
    original_snippet: str,
    mutated_snippet: str,
    existing_test_summaries: str = "",
) -> str:
    """Render the targeted-test prompt for one surviving mutant.

    Uses ``str.replace`` (not ``str.format``) so braces in the mutated
    source code are preserved verbatim.
    """
    template = get_targeted_prompt_template()
    rendered = template
    rendered = rendered.replace("{{operator}}", str(operator or ""))
    rendered = rendered.replace("{{source_path}}", str(source_path or ""))
    rendered = rendered.replace("{{line}}", str(line or 0))
    rendered = rendered.replace("{{original_snippet}}", str(original_snippet or ""))
    rendered = rendered.replace("{{mutated_snippet}}", str(mutated_snippet or ""))
    rendered = rendered.replace(
        "{{existing_test_summaries}}", str(existing_test_summaries or "(none)")
    )
    return rendered


class ActiveLearningStop(str, Enum):
    """Reason the loop terminated."""

    MAX_ITERATIONS = "max_iterations"
    KILL_RATE_STABLE = "kill_rate_stable"
    ALL_MUTANTS_KILLED = "all_mutants_killed"
    NO_SURVIVING_MUTANTS = "no_surviving_mutants"
    NO_GENERATOR_OUTPUT = "no_generator_output"


@dataclass
class _TestArtifact:
    """Minimal duck-typed test artifact.

    The codebase uses dicts shaped ``{"path": str, "content": str}`` as
    the canonical artifact format (see ``apex.modes`` and
    ``safe_materialize_test_artifacts``). We keep the same shape here.
    """

    path: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "content": self.content}


@dataclass
class IterationDiagnostic:
    """One per-iteration record for audit."""

    iteration: int
    surviving_before: int
    surviving_after: int
    killed_before: int
    killed_after: int
    kill_rate_before: float
    kill_rate_after: float
    new_tests_added: int
    targeted_mutants: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EnhancedTestSuite:
    """Result of running the active learner."""

    test_artifacts: list[dict[str, Any]]
    final_kill_rate: float
    initial_kill_rate: float
    final_surviving: int
    iterations_run: int
    stop_reason: ActiveLearningStop
    iteration_diagnostics: list[IterationDiagnostic] = field(default_factory=list)
    # Phase B.6 (Decisive-Edge): per-iteration kill-rate snapshot. The
    # first entry is the BASELINE (initial_kill_rate); each subsequent
    # entry is the post-iteration kill rate. Useful for plotting
    # convergence curves outside this module without having to reduce
    # iteration_diagnostics again.
    iteration_history: list[float] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_artifacts": list(self.test_artifacts),
            "final_kill_rate": float(self.final_kill_rate),
            "initial_kill_rate": float(self.initial_kill_rate),
            "final_surviving": int(self.final_surviving),
            "iterations_run": int(self.iterations_run),
            "stop_reason": str(self.stop_reason.value),
            "iteration_history": [float(x) for x in self.iteration_history],
            "iteration_diagnostics": [
                {
                    "iteration": d.iteration,
                    "surviving_before": d.surviving_before,
                    "surviving_after": d.surviving_after,
                    "killed_before": d.killed_before,
                    "killed_after": d.killed_after,
                    "kill_rate_before": d.kill_rate_before,
                    "kill_rate_after": d.kill_rate_after,
                    "new_tests_added": d.new_tests_added,
                    "targeted_mutants": list(d.targeted_mutants),
                }
                for d in self.iteration_diagnostics
            ],
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _surviving_mutants(report: Any) -> list[Any]:
    """Extract survivor outcomes from a MutationReport-shaped object.

    We duck-type so tests can pass a SimpleNamespace; production
    callers pass an :class:`apex.evaluation.mutation_engine.MutationReport`.
    A surviving mutant has ``status == "survived"`` per the
    documented MutantOutcome.status values.
    """
    per_mutant = getattr(report, "per_mutant", None) or []
    return [
        outcome
        for outcome in per_mutant
        if str(getattr(outcome, "status", "")).lower() == "survived"
    ]


def _kill_rate(report: Any) -> float:
    """Mutation score / kill rate from a MutationReport-shaped object.

    Falls back to ``killed / max(total_mutants, 1)`` if no
    ``mutation_score`` attribute is present (test-friendly).
    """
    score = getattr(report, "mutation_score", None)
    if score is not None:
        return float(score)
    killed = int(getattr(report, "killed", 0) or 0)
    total = int(getattr(report, "total_mutants", 0) or 0)
    return float(killed) / float(max(total, 1))


def _mutant_payload(outcome: Any) -> dict[str, Any]:
    """Serialize one MutantOutcome for prompts and diagnostics.

    Keeping this in one place means the prompt template above and the
    diagnostic record below share the same field names.
    """
    mutant = getattr(outcome, "mutant", None)
    return {
        "operator": str(getattr(mutant, "operator", "") or ""),
        "source_path": str(getattr(mutant, "source_path", "") or ""),
        "line": int(getattr(mutant, "line", 0) or 0),
        "col": int(getattr(mutant, "col", 0) or 0),
        "original_snippet": str(getattr(mutant, "original_snippet", "") or ""),
        "mutated_snippet": str(getattr(mutant, "mutated_snippet", "") or ""),
        "status": str(getattr(outcome, "status", "")),
    }


# ---------------------------------------------------------------------------
# Public type aliases for pluggable callables
# ---------------------------------------------------------------------------


# A targeted test generator takes a list of mutant payload dicts (one
# per surviving mutant we want to attack) and returns a list of new
# test artifacts (one per mutant, or empty if none could be produced).
# Production callers wire this into the existing testgen pipeline; tests
# pass a fake.
TargetedTestGenerator = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]

# A mutation runner takes the current test artifact list and returns
# something MutationReport-shaped. Production callers wire this to
# ``evaluate_mutation_score`` after re-materialising the suite into
# ``fixed_dir``; tests pass a fake.
MutationRunner = Callable[[list[dict[str, Any]]], Any]


# ---------------------------------------------------------------------------
# Active learner
# ---------------------------------------------------------------------------


@dataclass
class MutationActiveLearner:
    """Iteratively grow a test suite to kill more mutants.

    Each iteration:
      1. Identify surviving mutants in the current report.
      2. Pick the top-K (by mutant payload identity, not by score —
         every survivor is equally promising for a targeted test).
      3. Call ``targeted_test_generator`` to produce new tests aimed
         at those survivors.
      4. Append the new tests to the suite.
      5. Re-run mutation; check stopping criteria.

    The same instance can be re-used across calls to
    :meth:`attack_surviving_mutants`.
    """

    # Phase B.6: default lowered from 3 to 2 (DEFAULT_MAX_ITERATIONS).
    # Empirical Phase 6 evaluation showed iteration 3 rarely killed any
    # additional mutants on Commit0/SWE-Bench tasks. Operators who DO
    # want a longer loop can pass max_iterations=N explicitly; the
    # absolute cap of 10 still bounds the worst case.
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    top_k_mutants_per_iteration: int = DEFAULT_TOP_K_MUTANTS_PER_ITERATION
    min_kill_rate_improvement: float = DEFAULT_MIN_KILL_RATE_IMPROVEMENT

    def __post_init__(self) -> None:
        # Guard against pathological callers that override the cap to
        # something dangerous.
        self.max_iterations = max(1, min(int(self.max_iterations), _ABSOLUTE_MAX_ITERATIONS))
        self.top_k_mutants_per_iteration = max(1, int(self.top_k_mutants_per_iteration))
        self.min_kill_rate_improvement = max(0.0, float(self.min_kill_rate_improvement))

    # ------------------------------------------------------------------

    def attack_surviving_mutants(
        self,
        *,
        test_suite: list[dict[str, Any]],
        mutation_report: Any,
        targeted_test_generator: TargetedTestGenerator,
        mutation_runner: MutationRunner,
    ) -> EnhancedTestSuite:
        """Run the active-learning loop. See class docstring for semantics.

        ``test_suite`` is the starting test artifact list. ``mutation_report``
        is the BASELINE report on that suite — we don't re-run mutation
        before the first attack. The loop runs at most
        ``self.max_iterations`` more attacks; each attack updates
        ``test_suite`` and re-runs mutation via ``mutation_runner``.
        """
        if not callable(targeted_test_generator):
            return EnhancedTestSuite(
                test_artifacts=list(test_suite),
                final_kill_rate=_kill_rate(mutation_report),
                initial_kill_rate=_kill_rate(mutation_report),
                final_surviving=len(_surviving_mutants(mutation_report)),
                iterations_run=0,
                stop_reason=ActiveLearningStop.NO_GENERATOR_OUTPUT,
                error="targeted_test_generator is not callable",
            )
        if not callable(mutation_runner):
            return EnhancedTestSuite(
                test_artifacts=list(test_suite),
                final_kill_rate=_kill_rate(mutation_report),
                initial_kill_rate=_kill_rate(mutation_report),
                final_surviving=len(_surviving_mutants(mutation_report)),
                iterations_run=0,
                stop_reason=ActiveLearningStop.NO_GENERATOR_OUTPUT,
                error="mutation_runner is not callable",
            )

        current_suite = [dict(a) for a in (test_suite or [])]
        current_report = mutation_report
        initial_kill = _kill_rate(current_report)
        last_kill = initial_kill
        diagnostics: list[IterationDiagnostic] = []
        # Phase B.6: track per-iteration kill-rate history starting
        # with the baseline. Each successful mutation_runner call
        # appends the new kill rate; the final value equals
        # ``last_kill`` at loop exit.
        iteration_history: list[float] = [initial_kill]
        stop_reason: Optional[ActiveLearningStop] = None

        for iteration in range(1, self.max_iterations + 1):
            survivors = _surviving_mutants(current_report)
            if not survivors:
                stop_reason = (
                    ActiveLearningStop.NO_SURVIVING_MUTANTS
                    if iteration == 1
                    else ActiveLearningStop.ALL_MUTANTS_KILLED
                )
                break

            # Pick the top-K survivors. With no per-mutant ranking
            # signal in the report, we just take the first K — they
            # already arrive in a deterministic order from the mutation
            # engine (operator, file, line). Future work: prioritise by
            # operator class or coverage.
            targeted = survivors[: self.top_k_mutants_per_iteration]
            mutant_payloads = [_mutant_payload(o) for o in targeted]

            try:
                new_tests = targeted_test_generator(mutant_payloads)
            except Exception as exc:
                logger.warning(
                    "MutationActiveLearner: targeted_test_generator raised %s; stopping loop.",
                    exc,
                )
                stop_reason = ActiveLearningStop.NO_GENERATOR_OUTPUT
                diagnostics.append(
                    IterationDiagnostic(
                        iteration=iteration,
                        surviving_before=len(survivors),
                        surviving_after=len(survivors),
                        killed_before=int(getattr(current_report, "killed", 0) or 0),
                        killed_after=int(getattr(current_report, "killed", 0) or 0),
                        kill_rate_before=last_kill,
                        kill_rate_after=last_kill,
                        new_tests_added=0,
                        targeted_mutants=mutant_payloads,
                    )
                )
                break

            normalised_new_tests: list[dict[str, Any]] = []
            for entry in new_tests or []:
                if isinstance(entry, dict) and entry.get("path") and entry.get("content"):
                    normalised_new_tests.append(
                        {
                            "path": str(entry["path"]),
                            "content": str(entry["content"]),
                        }
                    )

            if not normalised_new_tests:
                # The generator declined to produce anything for this
                # iteration. We've still attempted, so record the
                # diagnostic and stop — another iteration would just
                # ask the same generator again with the same survivors.
                stop_reason = ActiveLearningStop.NO_GENERATOR_OUTPUT
                diagnostics.append(
                    IterationDiagnostic(
                        iteration=iteration,
                        surviving_before=len(survivors),
                        surviving_after=len(survivors),
                        killed_before=int(getattr(current_report, "killed", 0) or 0),
                        killed_after=int(getattr(current_report, "killed", 0) or 0),
                        kill_rate_before=last_kill,
                        kill_rate_after=last_kill,
                        new_tests_added=0,
                        targeted_mutants=mutant_payloads,
                    )
                )
                break

            current_suite = current_suite + normalised_new_tests

            try:
                next_report = mutation_runner(current_suite)
            except Exception as exc:
                logger.warning(
                    "MutationActiveLearner: mutation_runner raised %s; stopping loop.",
                    exc,
                )
                stop_reason = ActiveLearningStop.NO_GENERATOR_OUTPUT
                diagnostics.append(
                    IterationDiagnostic(
                        iteration=iteration,
                        surviving_before=len(survivors),
                        surviving_after=len(survivors),
                        killed_before=int(getattr(current_report, "killed", 0) or 0),
                        killed_after=int(getattr(current_report, "killed", 0) or 0),
                        kill_rate_before=last_kill,
                        kill_rate_after=last_kill,
                        new_tests_added=len(normalised_new_tests),
                        targeted_mutants=mutant_payloads,
                    )
                )
                break

            new_kill = _kill_rate(next_report)
            new_surviving = _surviving_mutants(next_report)
            diagnostics.append(
                IterationDiagnostic(
                    iteration=iteration,
                    surviving_before=len(survivors),
                    surviving_after=len(new_surviving),
                    killed_before=int(getattr(current_report, "killed", 0) or 0),
                    killed_after=int(getattr(next_report, "killed", 0) or 0),
                    kill_rate_before=last_kill,
                    kill_rate_after=new_kill,
                    new_tests_added=len(normalised_new_tests),
                    targeted_mutants=mutant_payloads,
                )
            )

            current_report = next_report
            improvement = new_kill - last_kill
            last_kill = new_kill
            iteration_history.append(new_kill)

            if not new_surviving:
                stop_reason = ActiveLearningStop.ALL_MUTANTS_KILLED
                break
            if improvement < self.min_kill_rate_improvement:
                stop_reason = ActiveLearningStop.KILL_RATE_STABLE
                break

        if stop_reason is None:
            stop_reason = ActiveLearningStop.MAX_ITERATIONS

        return EnhancedTestSuite(
            test_artifacts=current_suite,
            final_kill_rate=last_kill,
            initial_kill_rate=initial_kill,
            final_surviving=len(_surviving_mutants(current_report)),
            iterations_run=len(diagnostics),
            stop_reason=stop_reason,
            iteration_diagnostics=diagnostics,
            iteration_history=list(iteration_history),
        )


__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MIN_KILL_RATE_IMPROVEMENT",
    "DEFAULT_TOP_K_MUTANTS_PER_ITERATION",
    "ActiveLearningStop",
    "EnhancedTestSuite",
    "IterationDiagnostic",
    "MutationActiveLearner",
    "MutationRunner",
    "TargetedTestGenerator",
    "get_targeted_prompt_template",
    "render_targeted_prompt",
    "reset_targeted_prompt_template_cache",
]
