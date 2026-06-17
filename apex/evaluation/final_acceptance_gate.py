"""Final whole-file acceptance gate for generated test artifacts."""

from __future__ import annotations

import ast
import logging
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .test_minimizer import (
    drop_tests_from_artifact_with_report,
    test_names_in_artifact,
)

logger = logging.getLogger(__name__)


# Phase 4A item 4.3: estimator that returns a mutation_kill_contribution
# in [0, 1] for a single test function inside an artifact. Production
# callers can plug in a real per-test mutation runner; the default
# heuristic uses assertion-strength scoring (concrete == checks > raises >
# bool-only) so the gate works even without a mutation engine in the loop.
MutationKillEstimator = Callable[[str, str], float]


def strict_syntax_check(
    source: str,
    *,
    filename: str = "<generated>",
) -> tuple[bool, str | None]:
    """Run ``ast.parse`` AND ``compile`` on ``source``.

    Returns ``(True, None)`` when the source is syntactically valid Python
    that can be byte-compiled, ``(False, error_message)`` otherwise.

    ``ast.parse`` catches almost everything, but ``compile`` catches a few
    additional shapes (e.g. the W3 plan calls out async-await outside an
    async function, ``return`` at module scope, duplicate keyword args)
    that the parser accepts but byte-compilation rejects. Both checks are
    cheap so we run both — false negatives here translate directly to
    docker-runtime SyntaxError / IndentationError failures.
    """

    text = source if isinstance(source, str) else str(source or "")
    if not text.strip():
        # Empty/whitespace-only artifacts are a separate failure mode handled
        # upstream; treat them as valid here so the gate doesn't double-report.
        return True, None
    try:
        ast.parse(text, filename=filename)
    except SyntaxError as exc:
        return False, _format_syntax_error("ast.parse", exc)
    except (ValueError, MemoryError, RecursionError) as exc:  # pragma: no cover - defensive
        return False, f"{type(exc).__name__}: {exc}"
    # Audit M3: catch SyntaxWarning (e.g. `is`-with-literal) as a hard
    # failure. Eight v5_full_20260509 tests slipped past with these
    # warnings — they're genuine bugs (`x is 1` doesn't compare equality)
    # that we want to repair, not ship.
    import warnings as _warnings

    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", SyntaxWarning)
            compile(text, filename, "exec")
    except SyntaxError as exc:
        return False, _format_syntax_error("compile", exc)
    except SyntaxWarning as exc:
        return False, f"compile: SyntaxWarning: {exc}"
    except (ValueError, MemoryError, RecursionError) as exc:  # pragma: no cover - defensive
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def _format_syntax_error(stage: str, exc: SyntaxError) -> str:
    line = getattr(exc, "lineno", None)
    offset = getattr(exc, "offset", None)
    location = ""
    if line is not None:
        location = f" line {line}"
        if offset is not None:
            location += f", col {offset}"
    return f"{stage}: {type(exc).__name__}{location}: {exc.msg or str(exc)}".strip()


@dataclass(frozen=True)
class GeneratedArtifact:
    path: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, artifact: Any) -> "GeneratedArtifact":
        if isinstance(artifact, GeneratedArtifact):
            return artifact
        if isinstance(artifact, dict):
            return cls(
                path=str(artifact.get("path") or "tests/test_generated.py"),
                content=str(artifact.get("content") or ""),
                metadata={
                    key: value for key, value in artifact.items() if key not in {"path", "content"}
                },
            )
        return cls(path="tests/test_generated.py", content=str(artifact or ""))

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "content": self.content, **dict(self.metadata)}


@dataclass(frozen=True)
class FinalAcceptanceRun:
    status: str
    per_test_status: dict[str, str] = field(default_factory=dict)
    stdout_tail: str = ""
    stderr_tail: str = ""
    returncode: int | None = None
    diagnostic: str = ""
    failure_taxonomy: str = ""
    raw_parser_status: str = ""
    static_findings: list[dict[str, Any]] = field(default_factory=list)

    @property
    def failing_test_names(self) -> set[str]:
        return _names_for_statuses(self.per_test_status, {"fail", "failed"})

    @property
    def errored_test_names(self) -> set[str]:
        return _names_for_statuses(self.per_test_status, {"error", "errored"})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FinalAcceptanceResult:
    status: str
    artifact: GeneratedArtifact
    dropped_tests: list[str] = field(default_factory=list)
    iterations: int = 0
    telemetry: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""
    failure_taxonomy: str = ""
    static_findings: list[dict[str, Any]] = field(default_factory=list)
    raw_parser_status: str = ""
    # P1.2 fix: static-validator findings populated by ship_acceptance
    # before the adapter runs. The V4 audit found these were dead code
    # in production because enforce_final_acceptance never ran the
    # validators; ship_acceptance now does, and the fields here let
    # downstream telemetry consume the findings without depending on
    # which gate function was called.
    mock_path_findings: list[dict[str, Any]] = field(default_factory=list)
    attribute_chain_findings: list[dict[str, Any]] = field(default_factory=list)
    # Phase 4A item 4.3: tests that DETERMINISTICALLY fail under the
    # current oracle but have HIGH mutation-kill contribution (>= the
    # configured min). Dropping them would lose discriminating power;
    # surfacing them as oracle_disagreement signals to the caller that
    # either the test is wrong OR the implementation is wrong, and a
    # human should look at it. Each entry: {"test_name", "assertion",
    # "mutation_kill_contribution", "diagnostic"}.
    oracle_disagreements: list[dict[str, Any]] = field(default_factory=list)
    # Per-test deterministic-vs-flaky classification used by the gate's
    # selective-drop step. Flaky tests survive even when they fail in
    # the first run.
    flake_classification: dict[str, str] = field(default_factory=dict)
    # Aggregate suite-level mutation-kill estimate after the drop step.
    # Falls below ``final_acceptance_weak_artifact_threshold`` triggers
    # ``weak_minimized_artifact`` regardless of whether the suite is
    # non-empty (selective drop can leave a non-empty but signal-weak
    # suite, which is worse than failing loud).
    suite_mutation_kill_score: float = 0.0

    @property
    def shipped(self) -> bool:
        return self.status == "ship"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "artifact": self.artifact.to_dict(),
            "dropped_tests": list(self.dropped_tests),
            "iterations": self.iterations,
            "telemetry": list(self.telemetry),
            "note": self.note,
            "failure_taxonomy": self.failure_taxonomy,
            "static_findings": list(self.static_findings),
            "raw_parser_status": self.raw_parser_status,
            "mock_path_findings": list(self.mock_path_findings),
            "attribute_chain_findings": list(self.attribute_chain_findings),
            "oracle_disagreements": list(self.oracle_disagreements),
            "flake_classification": dict(self.flake_classification),
            "suite_mutation_kill_score": float(self.suite_mutation_kill_score),
        }


def enforce_final_acceptance(
    artifact: GeneratedArtifact | dict[str, Any] | str,
    *,
    benchmark_adapter: Any,
    workdir: Path,
    keep_minimum: int = 1,
    max_drop_iterations: int = 5,
    flake_retries: Optional[int] = None,
    min_mutation_contribution: Optional[float] = None,
    weak_artifact_threshold: Optional[float] = None,
    mutation_kill_estimator: Optional[MutationKillEstimator] = None,
) -> FinalAcceptanceResult:
    """Run the full artifact and selectively drop failing tests.

    Phase 4A item 4.3 — selective gating, NOT blanket drop.

    The adapter owns benchmark-specific execution. The gate runs the full
    suite, classifies each failing test as flaky vs deterministic via
    ``flake_retries`` re-runs, computes a mutation_kill_contribution for
    each deterministic failure via ``mutation_kill_estimator``, then:

      * drops only deterministic failures with low mutation_kill
        contribution (< ``min_mutation_contribution``);
      * preserves flaky failures (a test that passes on retry has
        signal — it's just noisy);
      * preserves high-mutation-kill deterministic failures and surfaces
        them as ``oracle_disagreements`` (the test or the implementation
        is wrong; either way, dropping it would lose information that
        the human should triage).

    After all drops, if the surviving suite has a mutation_kill_score
    below ``weak_artifact_threshold``, returns ``weak_minimized_artifact``
    even if the suite is non-empty — a passing-but-toothless suite is
    worse than failing loud.

    All thresholds default to ``OrchestrationConfig`` values when not
    supplied so operators can tune behavior centrally.
    """

    flake_retries_resolved = _resolve_int_threshold(
        flake_retries, "final_acceptance_flake_retries", default=3
    )
    min_mutation = _resolve_float_threshold(
        min_mutation_contribution,
        "final_acceptance_min_mutation_contribution",
        default=0.05,
    )
    weak_threshold = _resolve_float_threshold(
        weak_artifact_threshold,
        "final_acceptance_weak_artifact_threshold",
        default=0.20,
    )
    estimator = mutation_kill_estimator or _default_mutation_kill_estimator

    current = GeneratedArtifact.from_any(artifact)
    dropped_all: list[str] = []
    telemetry: list[dict[str, Any]] = []
    oracle_disagreements: list[dict[str, Any]] = []
    flake_classification: dict[str, str] = {}
    original_test_count = len(test_names_in_artifact(current.content))
    # Strict W3 syntax gate. A candidate that fails ast.parse OR compile()
    # cannot pass any benchmark adapter run — short-circuit before paying
    # the docker / pytest cost so the failure category is a clear
    # ``syntax_error`` rather than a noisy adapter ``harness_error``.
    language = _artifact_language(current)
    if language in {"python", "py", "python3"}:
        syntax_ok, syntax_error = strict_syntax_check(
            current.content,
            filename=current.path or "<generated>",
        )
        if not syntax_ok:
            return FinalAcceptanceResult(
                status="syntax_error",
                artifact=current,
                dropped_tests=dropped_all,
                iterations=0,
                telemetry=telemetry,
                note=f"strict_syntax_check failed: {syntax_error}",
                failure_taxonomy="artifact_failed",
                raw_parser_status="syntax_error",
            )
    else:
        telemetry.append(
            {
                "raw_parser_status": "unsupported",
                "language": language,
                "note": "syntax check deferred to runner adapter",
            }
        )
    namespace_result = _validate_artifact_namespace(current)
    if namespace_result is not None:
        telemetry.append({"static_namespace": namespace_result})
        if namespace_result.get("status") == "fail":
            return FinalAcceptanceResult(
                status="artifact_failed",
                artifact=current,
                dropped_tests=dropped_all,
                iterations=0,
                telemetry=telemetry,
                note="namespace/static validation failed",
                failure_taxonomy="artifact_failed",
                static_findings=list(namespace_result.get("findings") or []),
                raw_parser_status=str(namespace_result.get("status") or ""),
            )
    iterations = max(1, int(max_drop_iterations or 1))
    for index in range(iterations):
        raw_run = benchmark_adapter.run_unfiltered(current, Path(workdir))
        run = _coerce_run(raw_run)
        telemetry.append(run.to_dict())
        if run.failure_taxonomy in {"artifact_failed", "collection_failed", "setup_failed"}:
            return FinalAcceptanceResult(
                status=run.failure_taxonomy,
                artifact=current,
                dropped_tests=dropped_all,
                iterations=index + 1,
                telemetry=telemetry,
                note=run.diagnostic,
                failure_taxonomy=run.failure_taxonomy,
                static_findings=list(run.static_findings),
                raw_parser_status=run.raw_parser_status,
            )
        if run.status == "harness_error":
            return FinalAcceptanceResult(
                status=run.failure_taxonomy or "harness_infra_error",
                artifact=current,
                dropped_tests=dropped_all,
                iterations=index + 1,
                telemetry=telemetry,
                note=run.diagnostic,
                failure_taxonomy=run.failure_taxonomy or "harness_infra_error",
                static_findings=list(run.static_findings),
                raw_parser_status=run.raw_parser_status,
            )
        failing = run.failing_test_names | run.errored_test_names
        if not failing and run.status in {"pass", "ok", "passed"}:
            suite_score = _suite_mutation_kill_score(current.content, estimator)
            quality_failure = _minimized_artifact_quality_failure(
                current=current,
                dropped_tests=dropped_all,
                original_test_count=original_test_count,
            )
            # Phase 4A item 4.3: weak_minimized_artifact also fires on
            # low mutation-kill suites, regardless of suite size.
            if quality_failure is None and suite_score < weak_threshold:
                quality_failure = (
                    f"suite_mutation_kill_score={suite_score:.3f} "
                    f"< weak_artifact_threshold={weak_threshold:.3f}"
                )
            if quality_failure is not None:
                return FinalAcceptanceResult(
                    status="weak_minimized_artifact",
                    artifact=current,
                    dropped_tests=dropped_all,
                    iterations=index + 1,
                    telemetry=telemetry,
                    note=quality_failure,
                    failure_taxonomy="artifact_failed",
                    raw_parser_status=run.raw_parser_status,
                    oracle_disagreements=oracle_disagreements,
                    flake_classification=flake_classification,
                    suite_mutation_kill_score=suite_score,
                )
            return FinalAcceptanceResult(
                status="ship",
                artifact=current,
                dropped_tests=dropped_all,
                iterations=index + 1,
                telemetry=telemetry,
                oracle_disagreements=oracle_disagreements,
                flake_classification=flake_classification,
                suite_mutation_kill_score=suite_score,
            )
        if not failing:
            return FinalAcceptanceResult(
                status="harness_infra_error",
                artifact=current,
                dropped_tests=dropped_all,
                iterations=index + 1,
                telemetry=telemetry,
                note=run.diagnostic or "runner failed without per-test failures",
                failure_taxonomy="harness_infra_error",
            )
        if language not in {"python", "py", "python3"}:
            return FinalAcceptanceResult(
                status="artifact_failed",
                artifact=current,
                dropped_tests=dropped_all,
                iterations=index + 1,
                telemetry=telemetry,
                note="runner reported failing tests; adapter drop/minimize is unsupported",
                failure_taxonomy="artifact_failed",
                raw_parser_status="unsupported",
            )
        # Phase 4A item 4.3 — selective gate. Classify each failing test
        # as flaky vs deterministic via ``flake_retries`` re-runs of the
        # full suite, then compute mutation_kill_contribution per
        # deterministic failure to decide drop vs preserve-with-warning.
        flaky_set, deterministic_set = _classify_failures(
            failing=failing,
            run=run,
            adapter=benchmark_adapter,
            artifact=current,
            workdir=Path(workdir),
            retries=flake_retries_resolved,
            telemetry=telemetry,
        )
        for name in flaky_set:
            flake_classification[name] = "flaky"
        for name in deterministic_set:
            flake_classification[name] = "deterministic"

        # For each deterministic failure compute mutation_kill_contribution.
        drop_candidates: set[str] = set()
        for name in deterministic_set:
            contribution = _mutation_kill_contribution_for_test(current.content, name, estimator)
            if contribution >= min_mutation:
                # High-mutation-kill but failing → oracle_disagreement.
                oracle_disagreements.append(
                    {
                        "test_name": name,
                        "assertion": _first_assertion_text_for_test(current.content, name),
                        "mutation_kill_contribution": round(contribution, 4),
                        "diagnostic": (
                            "deterministic failure preserved as "
                            "oracle_disagreement: either the test or the "
                            "implementation is wrong"
                        ),
                    }
                )
            else:
                drop_candidates.add(name)

        if not drop_candidates:
            # Nothing to drop this iteration — every failure is either
            # flaky (will retry) or high-mutation-kill (preserve).
            # Bail out and ship if the suite still has signal; otherwise
            # surface as artifact_failed so the caller doesn't see a
            # silent "ship" with all failures hidden.
            suite_score = _suite_mutation_kill_score(current.content, estimator)
            if oracle_disagreements:
                return FinalAcceptanceResult(
                    status="oracle_disagreement",
                    artifact=current,
                    dropped_tests=dropped_all,
                    iterations=index + 1,
                    telemetry=telemetry,
                    note=(
                        f"{len(oracle_disagreements)} test(s) deterministically "
                        "fail under the current oracle but have high mutation "
                        "kill contribution; preserved as oracle_disagreement"
                    ),
                    failure_taxonomy="oracle_disagreement",
                    oracle_disagreements=oracle_disagreements,
                    flake_classification=flake_classification,
                    suite_mutation_kill_score=suite_score,
                )
            # All failures were flaky; ship as-is.
            return FinalAcceptanceResult(
                status="ship",
                artifact=current,
                dropped_tests=dropped_all,
                iterations=index + 1,
                telemetry=telemetry,
                oracle_disagreements=oracle_disagreements,
                flake_classification=flake_classification,
                suite_mutation_kill_score=suite_score,
            )

        remaining = set(test_names_in_artifact(current.content)) - drop_candidates
        if len(remaining) < max(1, int(keep_minimum or 1)):
            return FinalAcceptanceResult(
                status="dropped_to_empty",
                artifact=current,
                dropped_tests=sorted(set(dropped_all) | drop_candidates),
                iterations=index + 1,
                telemetry=telemetry,
                failure_taxonomy="artifact_failed",
                oracle_disagreements=oracle_disagreements,
                flake_classification=flake_classification,
            )
        next_text, dropped = drop_tests_from_artifact_with_report(
            current.content,
            drop_candidates,
            keep_minimum=keep_minimum,
        )
        if not dropped:
            return FinalAcceptanceResult(
                status="artifact_failed"
                if _has_artifact_sentinel(run.per_test_status)
                else "harness_infra_error",
                artifact=current,
                dropped_tests=dropped_all,
                iterations=index + 1,
                telemetry=telemetry,
                note="drop transformer could not remove reported failures",
                failure_taxonomy="artifact_failed"
                if _has_artifact_sentinel(run.per_test_status)
                else "harness_infra_error",
                oracle_disagreements=oracle_disagreements,
                flake_classification=flake_classification,
            )
        dropped_all.extend(dropped)
        current = GeneratedArtifact(
            path=current.path,
            content=next_text,
            metadata=dict(current.metadata),
        )
    suite_score = _suite_mutation_kill_score(current.content, estimator)
    return FinalAcceptanceResult(
        status="did_not_stabilize",
        artifact=current,
        dropped_tests=dropped_all,
        iterations=iterations,
        telemetry=telemetry,
        note="did_not_stabilize",
        failure_taxonomy="artifact_failed",
        oracle_disagreements=oracle_disagreements,
        flake_classification=flake_classification,
        suite_mutation_kill_score=suite_score,
    )


_FINAL_ACCEPTANCE_ENV_OVERRIDES: dict[str, str] = {
    "final_acceptance_flake_retries": "APEX_FINAL_ACCEPTANCE_FLAKE_RETRIES",
    "final_acceptance_min_mutation_contribution": (
        "APEX_FINAL_ACCEPTANCE_MIN_MUTATION_CONTRIBUTION"
    ),
    "final_acceptance_weak_artifact_threshold": ("APEX_FINAL_ACCEPTANCE_WEAK_ARTIFACT_THRESHOLD"),
}


def _resolve_int_threshold(explicit: Optional[int], attr: str, *, default: int) -> int:
    if explicit is not None:
        return max(0, int(explicit))
    env_name = _FINAL_ACCEPTANCE_ENV_OVERRIDES.get(attr, "")
    if env_name:
        raw = os.environ.get(env_name)
        if raw is not None:
            try:
                return max(0, int(raw))
            except (TypeError, ValueError):
                pass
    try:
        from ..core.config import OrchestrationConfig

        return int(getattr(OrchestrationConfig(), attr, default))
    except Exception:  # pragma: no cover - defensive
        return default


def _resolve_float_threshold(explicit: Optional[float], attr: str, *, default: float) -> float:
    if explicit is not None:
        return float(explicit)
    env_name = _FINAL_ACCEPTANCE_ENV_OVERRIDES.get(attr, "")
    if env_name:
        raw = os.environ.get(env_name)
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
    try:
        from ..core.config import OrchestrationConfig

        return float(getattr(OrchestrationConfig(), attr, default))
    except Exception:  # pragma: no cover - defensive
        return default


def _classify_failures(
    *,
    failing: set[str],
    run: FinalAcceptanceRun,
    adapter: Any,
    artifact: GeneratedArtifact,
    workdir: Path,
    retries: int,
    telemetry: list[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    """Re-run the suite up to ``retries`` times to split flaky from
    deterministic failures.

    A test is *flaky* when it passes in at least one retry; otherwise
    *deterministic*. Each retry is a full ``benchmark_adapter.run_unfiltered``
    invocation against the unmodified artifact — we don't try to run a
    single test in isolation because most adapters don't expose that
    contract. Telemetry records every retry so the caller can see the
    classification cost.

    When ``retries <= 0`` we treat every initial failure as deterministic
    (back-compat with the legacy gate). On adapter exceptions during a
    retry we conservatively keep the test in the deterministic bucket so
    a flaky retry harness can't silently rescue genuine failures.
    """

    if not failing or retries <= 0:
        return set(), set(failing)

    still_failing = set(failing)
    flaky: set[str] = set()
    for retry_index in range(int(retries)):
        if not still_failing:
            break
        try:
            raw_run = adapter.run_unfiltered(artifact, Path(workdir))
        except Exception as exc:  # pragma: no cover - defensive
            telemetry.append(
                {
                    "flake_retry_index": retry_index + 1,
                    "harness_error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        retry_run = _coerce_run(raw_run)
        retry_failing = retry_run.failing_test_names | retry_run.errored_test_names
        passed_this_retry = still_failing - retry_failing
        if passed_this_retry:
            flaky |= passed_this_retry
            still_failing -= passed_this_retry
        telemetry.append(
            {
                "flake_retry_index": retry_index + 1,
                "still_failing": sorted(still_failing),
                "passed_this_retry": sorted(passed_this_retry),
            }
        )
    return flaky, still_failing


def _default_mutation_kill_estimator(artifact_text: str, test_name: str) -> float:
    """Estimate per-test mutation_kill_contribution from assertion strength.

    Phase 4A item 4.3 — when no production mutation runner is plugged
    in, this heuristic ranks tests by the strength of their assertions:

      * concrete equality (``==``, ``!=``, ``is``) → 1.0 weight per
      * ordered comparisons (``<``, ``>=`` ...) → 0.7 weight per
      * exception expectations (``pytest.raises``, ``self.assertRaises``,
        plain ``raises``) → 0.6 weight per
      * bare ``assert <expr>`` (truthiness only) → 0.3 weight per
      * helper-call assertions (``assertEqual``, ``assert_array_equal``)
        → 0.8 weight per

    Returns the SUM of weights divided by the maximum possible (we
    treat the max as the test's own assertion count × 1.0). A test with
    one strong concrete equality scores 1.0; a test with three bare
    ``assert x`` scores 0.3; a test with no assertions at all scores 0.

    Inexact by construction — operators with a real per-test mutation
    runner should plug it in. The point of the heuristic is to avoid
    dropping tests with strong oracles when the cheap signal is
    available; the gate's selectivity threshold (default 0.05) is
    permissive enough that even bare-assert tests survive when the
    mutation runner is unavailable.
    """

    try:
        tree = ast.parse(artifact_text)
    except (SyntaxError, ValueError):
        return 0.0
    target_func: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == test_name:
                target_func = node
                break
    if target_func is None:
        return 0.0

    weights: list[float] = []
    for node in ast.walk(target_func):
        if isinstance(node, ast.Assert):
            test_node = node.test
            if isinstance(test_node, ast.Compare):
                ops = [type(o).__name__ for o in test_node.ops]
                if any(o in {"Eq", "NotEq", "Is", "IsNot"} for o in ops):
                    weights.append(1.0)
                elif any(o in {"Lt", "LtE", "Gt", "GtE", "In", "NotIn"} for o in ops):
                    weights.append(0.7)
                else:
                    weights.append(0.5)
            elif isinstance(test_node, ast.Call):
                func_name = _ast_attr_chain_name(test_node.func)
                if func_name in {
                    "isinstance",
                    "issubclass",
                    "all",
                    "any",
                }:
                    weights.append(0.5)
                elif "approx" in func_name:
                    weights.append(0.9)
                else:
                    weights.append(0.4)
            else:
                weights.append(0.3)
        elif isinstance(node, ast.With):
            for item in node.items:
                ctx = item.context_expr
                if isinstance(ctx, ast.Call):
                    name = _ast_attr_chain_name(ctx.func)
                    if "raises" in name or "assertRaises" in name:
                        weights.append(0.6)
        elif isinstance(node, ast.Call):
            func_name = _ast_attr_chain_name(node.func)
            if func_name.endswith("assertEqual") or func_name.endswith("assertNotEqual"):
                weights.append(1.0)
            elif func_name.endswith("assertTrue") or func_name.endswith("assertFalse"):
                weights.append(0.4)
            elif "assert_array_equal" in func_name or "assert_allclose" in func_name:
                weights.append(0.8)
            elif func_name.endswith("assertRaises"):
                weights.append(0.6)
    if not weights:
        return 0.0
    return min(1.0, sum(weights) / max(1.0, float(len(weights))))


def _ast_attr_chain_name(node: ast.AST) -> str:
    parts: list[str] = []
    current: ast.AST | None = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _mutation_kill_contribution_for_test(
    artifact_text: str, test_name: str, estimator: MutationKillEstimator
) -> float:
    try:
        return float(estimator(artifact_text, test_name))
    except Exception:  # pragma: no cover - defensive
        return 0.0


def _suite_mutation_kill_score(artifact_text: str, estimator: MutationKillEstimator) -> float:
    """Aggregate the per-test estimator across the suite.

    Phase 4A item 4.3: ``weak_minimized_artifact`` triggers when this
    score falls below ``final_acceptance_weak_artifact_threshold``,
    even on a non-empty suite. Average across the per-test estimates
    so a couple of strong tests can carry a suite of bare-assert
    fillers (selective drop should already have removed the worst
    offenders).
    """

    test_names = test_names_in_artifact(artifact_text)
    if not test_names:
        return 0.0
    contributions = [
        _mutation_kill_contribution_for_test(artifact_text, name, estimator) for name in test_names
    ]
    return float(sum(contributions) / max(1, len(contributions)))


def _first_assertion_text_for_test(artifact_text: str, test_name: str) -> str:
    """Return source text of the first assertion in ``test_name`` for
    the oracle_disagreement diagnostic. Empty when not parseable.
    """

    try:
        tree = ast.parse(artifact_text)
    except (SyntaxError, ValueError):
        return ""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == test_name:
            for inner in ast.walk(node):
                if isinstance(inner, ast.Assert):
                    try:
                        return ast.unparse(inner)
                    except Exception:  # pragma: no cover - defensive
                        return ""
                if isinstance(inner, ast.With):
                    for item in inner.items:
                        ctx = item.context_expr
                        if isinstance(ctx, ast.Call):
                            try:
                                return ast.unparse(inner)
                            except Exception:
                                return ""
            return ""
    return ""


def ship_acceptance(
    artifact: GeneratedArtifact | dict[str, Any] | str,
    *,
    benchmark_adapter: Any,
    workdir: Path,
    keep_minimum: int = 1,
    max_drop_iterations: int = 5,
    mock_path_allow_focal_import: bool = False,
    mock_path_extra_modules: tuple[str, ...] = (),
    attribute_chain_allow_import: bool = False,
    attribute_chain_extra_modules: tuple[str, ...] = (),
) -> FinalAcceptanceResult:
    """Canonical adapter-owned final acceptance entry point.

    P1.2 fix: this is the production gate that combines (a) the static
    pre-pytest validators (mock_path + attribute_chain) and (b) the
    adapter-driven iterative drop. Previously these were two separate
    gates and only the adapter side fired in production, leaving the
    mock_path / attribute_chain findings as dead code.

    The static validators run first against the artifact source;
    findings are surfaced via the new ``mock_path_findings`` and
    ``attribute_chain_findings`` fields on ``FinalAcceptanceResult``.
    High-confidence static findings block shipping before the adapter runs.
    This keeps benchmark acceptance from rewarding generated tests that only
    pass because they patch/mock the wrong target or dereference undefined
    chains the runner never exercises.
    """

    current = GeneratedArtifact.from_any(artifact)
    mock_findings_payload: list[dict[str, Any]] = []
    chain_findings_payload: list[dict[str, Any]] = []

    if _is_mock_path_validation_enabled():
        try:
            from .mock_path_validator import validate_mock_paths

            mock_result = validate_mock_paths(
                current.content,
                allow_import=mock_path_allow_focal_import,
                extra_modules=mock_path_extra_modules,
            )
            mock_findings_payload = [
                {
                    "test_name": f.test_name,
                    "target": f.target,
                    "call_kind": f.call_kind,
                    "line": f.line,
                    "reason": f.reason,
                }
                for f in mock_result.findings
            ]
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "ship_acceptance: validate_mock_paths failed (%s: %s); "
                "continuing without mock-path findings",
                type(exc).__name__,
                exc,
            )

    if _is_attribute_chain_check_enabled():
        try:
            from .import_preflight import detect_undefined_attribute_chains

            chain_result = detect_undefined_attribute_chains(
                current.content,
                allow_import=attribute_chain_allow_import,
                extra_modules=attribute_chain_extra_modules,
            )
            chain_findings_payload = [
                {
                    "chain": f.chain,
                    "root": f.root,
                    "missing_attr": f.missing_attr,
                    "resolved_module": f.resolved_module,
                    "line": f.line,
                    "enclosing_function": f.enclosing_function,
                    "reason": f.reason,
                }
                for f in chain_result.findings
            ]
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "ship_acceptance: detect_undefined_attribute_chains failed "
                "(%s: %s); continuing without chain findings",
                type(exc).__name__,
                exc,
            )

    if _block_static_acceptance_findings() and (mock_findings_payload or chain_findings_payload):
        findings = []
        findings.extend({"kind": "mock_path", **item} for item in mock_findings_payload)
        findings.extend(
            {"kind": "undefined_attribute_chain", **item} for item in chain_findings_payload
        )
        return FinalAcceptanceResult(
            status="artifact_failed",
            artifact=current,
            iterations=0,
            note="static acceptance findings blocked shipping",
            failure_taxonomy="artifact_failed",
            static_findings=findings,
            mock_path_findings=mock_findings_payload,
            attribute_chain_findings=chain_findings_payload,
        )

    base_result = enforce_final_acceptance(
        current,
        benchmark_adapter=benchmark_adapter,
        workdir=workdir,
        keep_minimum=keep_minimum,
        max_drop_iterations=max_drop_iterations,
    )
    # FinalAcceptanceResult is frozen; replace with the static-findings-bearing copy.
    return FinalAcceptanceResult(
        status=base_result.status,
        artifact=base_result.artifact,
        dropped_tests=list(base_result.dropped_tests),
        iterations=base_result.iterations,
        telemetry=list(base_result.telemetry),
        note=base_result.note,
        failure_taxonomy=base_result.failure_taxonomy,
        static_findings=list(base_result.static_findings),
        raw_parser_status=base_result.raw_parser_status,
        mock_path_findings=mock_findings_payload,
        attribute_chain_findings=chain_findings_payload,
        oracle_disagreements=list(base_result.oracle_disagreements),
        flake_classification=dict(base_result.flake_classification),
        suite_mutation_kill_score=base_result.suite_mutation_kill_score,
    )


def _is_mock_path_validation_enabled() -> bool:
    """Honor ``APEX_MOCK_PATH_VALIDATOR_ENABLED`` env (default ON)."""

    raw = os.environ.get("APEX_MOCK_PATH_VALIDATOR_ENABLED")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _is_attribute_chain_check_enabled() -> bool:
    """Honor ``APEX_ATTRIBUTE_CHAIN_CHECK_ENABLED`` env (default ON)."""

    raw = os.environ.get("APEX_ATTRIBUTE_CHAIN_CHECK_ENABLED")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _block_static_acceptance_findings() -> bool:
    raw = os.environ.get("APEX_FINAL_ACCEPTANCE_BLOCK_STATIC_FINDINGS")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _minimized_artifact_quality_failure(
    *,
    current: GeneratedArtifact,
    dropped_tests: list[str],
    original_test_count: int,
) -> str | None:
    if not dropped_tests:
        return None
    current_test_count = len(test_names_in_artifact(current.content))
    min_survivors = _env_int(
        "APEX_FINAL_ACCEPTANCE_MIN_SURVIVING_AFTER_DROP",
        default=2,
    )
    if current_test_count < max(1, min_survivors):
        return (
            "minimized artifact retained too few tests "
            f"({current_test_count} < {max(1, min_survivors)})"
        )
    max_drop_ratio = _env_float(
        "APEX_FINAL_ACCEPTANCE_MAX_DROP_RATIO",
        default=0.5,
    )
    denominator = max(1, int(original_test_count or current_test_count or 1))
    drop_ratio = len(set(dropped_tests)) / denominator
    if drop_ratio > max_drop_ratio:
        return (
            f"minimized artifact dropped too many tests ({drop_ratio:.3f} > {max_drop_ratio:.3f})"
        )
    return None


def _env_int(name: str, *, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, *, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _coerce_run(raw: Any) -> FinalAcceptanceRun:
    if isinstance(raw, FinalAcceptanceRun):
        if not raw.failure_taxonomy:
            taxonomy = _classify_failure_taxonomy(
                status=raw.status,
                per_test=raw.per_test_status,
                diagnostic=raw.diagnostic,
                stdout=raw.stdout_tail,
                stderr=raw.stderr_tail,
            )
            return FinalAcceptanceRun(
                status=raw.status,
                per_test_status=dict(raw.per_test_status),
                stdout_tail=raw.stdout_tail,
                stderr_tail=raw.stderr_tail,
                returncode=raw.returncode,
                diagnostic=raw.diagnostic,
                failure_taxonomy=taxonomy,
                raw_parser_status=raw.raw_parser_status,
                static_findings=list(raw.static_findings),
            )
        return raw
    if isinstance(raw, dict):
        per_test = {
            str(key): str(value or "").lower()
            for key, value in dict(raw.get("per_test_status") or {}).items()
        }
        status = str(raw.get("status") or "").lower()
        if not status:
            status = "pass" if per_test and set(per_test.values()) <= {"pass"} else "fail"
        taxonomy = _classify_failure_taxonomy(
            status=status,
            per_test=per_test,
            diagnostic=str(raw.get("diagnostic") or raw.get("error") or raw.get("note") or ""),
            stdout=str(raw.get("stdout_tail") or raw.get("stdout") or ""),
            stderr=str(raw.get("stderr_tail") or raw.get("stderr") or ""),
        )
        return FinalAcceptanceRun(
            status=status,
            per_test_status=per_test,
            stdout_tail=str(raw.get("stdout_tail") or raw.get("stdout") or "")[-4000:],
            stderr_tail=str(raw.get("stderr_tail") or raw.get("stderr") or "")[-4000:],
            returncode=raw.get("returncode"),
            diagnostic=str(raw.get("diagnostic") or raw.get("error") or ""),
            failure_taxonomy=taxonomy,
            raw_parser_status=str(raw.get("raw_parser_status") or raw.get("parser_status") or ""),
            static_findings=list(raw.get("static_findings") or []),
        )
    return FinalAcceptanceRun(
        status="harness_error", diagnostic=f"unsupported run payload: {type(raw).__name__}"
    )


def _names_for_statuses(per_test: dict[str, str], statuses: Iterable[str]) -> set[str]:
    wanted = {str(status).lower() for status in statuses}
    names: set[str] = set()
    for nodeid, raw_status in per_test.items():
        if str(raw_status or "").lower() not in wanted:
            continue
        parts = str(nodeid or "").split("::")
        for part in parts[1:] or parts:
            clean = part.split("[", 1)[0].strip()
            if clean.startswith("test_"):
                names.add(clean)
    return names


def _has_artifact_sentinel(per_test: dict[str, str]) -> bool:
    return any(str(nodeid).startswith("__suite__::") for nodeid in per_test)


def _classify_failure_taxonomy(
    *,
    status: str,
    per_test: dict[str, str],
    diagnostic: str,
    stdout: str,
    stderr: str,
) -> str:
    normalized = (status or "").lower()
    if normalized in {
        "artifact_failed",
        "collection_failed",
        "setup_failed",
        "harness_infra_error",
        "harness_log_missing",
    }:
        return normalized
    joined = "\n".join(part for part in (diagnostic, stdout, stderr) if part).lower()
    if any(
        "__suite__::collection" in key or "__suite__::collection_or_setup" in key
        for key in per_test
    ):
        return "collection_failed"
    if any("__suite__::setup" in key for key in per_test):
        return "setup_failed"
    if normalized == "harness_error":
        if any(
            marker in joined
            for marker in (
                "some tests failed",
                "unfiltered tests failed",
                "error collecting",
                "collection_or_setup",
            )
        ):
            return "artifact_failed"
        if "log missing" in joined or "missing log" in joined or "no eval log produced" in joined:
            return "harness_log_missing"
        return "harness_infra_error"
    return ""


def _validate_artifact_namespace(artifact: GeneratedArtifact) -> dict[str, Any] | None:
    context = artifact.metadata.get("repo_context")
    forbidden = artifact.metadata.get("forbidden_generated_names")
    required = artifact.metadata.get("required_focal_symbols")
    if not context and not forbidden and not required:
        return None
    try:
        from .repo_context import validate_generated_namespace

        return validate_generated_namespace(
            artifact.content,
            context,
            forbidden_generated_names=forbidden or (),
            required_focal_symbols=required or (),
        ).to_dict()
    except Exception as exc:
        return {
            "status": "skipped",
            "findings": [],
            "parse_error": f"{type(exc).__name__}: {exc}",
        }


def _artifact_language(artifact: GeneratedArtifact) -> str:
    explicit = str(artifact.metadata.get("language") or "").lower()
    if explicit:
        return explicit
    suffix = Path(artifact.path or "").suffix.lower()
    if suffix in {".py", ".pyi", ""}:
        return "python"
    if suffix in {".js", ".jsx", ".mjs", ".cjs"}:
        return "javascript"
    if suffix in {".ts", ".tsx"}:
        return "typescript"
    if suffix == ".go":
        return "go"
    if suffix == ".java":
        return "java"
    return suffix.lstrip(".") or "unknown"


# ---------------------------------------------------------------------------
# In-process per-test acceptance gate (general; benchmark-agnostic).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerTestAcceptanceResult:
    """Outcome of an in-process per-test acceptance pass.

    Distinct from ``FinalAcceptanceResult`` (the docker / benchmark-adapter
    gate). This gate runs pytest *locally* against the candidate test file
    in the supplied workdir, drops any tests that fail or error, and
    returns the cleaned artifact. It exists so APEX no longer relies on
    the downstream benchmark harness's per-test filter to rescue brittle
    candidates — that lift moves into APEX itself, which dramatically
    improves the strict (unfiltered) pass@1 and transfers to harnesses
    that don't have a forgiving filter.
    """

    status: str  # "shipped" | "shipped_no_drops" | "skipped" | "syntax_error" | "no_tests" | "all_dropped" | "harness_error"
    artifact: GeneratedArtifact
    dropped_tests: list[str] = field(default_factory=list)
    failing_test_names: list[str] = field(default_factory=list)
    erroring_test_names: list[str] = field(default_factory=list)
    iterations: int = 0
    pytest_returncode: int | None = None
    pytest_stdout_tail: str = ""
    pytest_stderr_tail: str = ""
    note: str = ""
    mock_path_findings: list[dict[str, Any]] = field(default_factory=list)
    attribute_chain_findings: list[dict[str, Any]] = field(default_factory=list)

    @property
    def shipped(self) -> bool:
        return self.status in {"shipped", "shipped_no_drops", "skipped"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "artifact": self.artifact.to_dict(),
            "dropped_tests": list(self.dropped_tests),
            "failing_test_names": list(self.failing_test_names),
            "erroring_test_names": list(self.erroring_test_names),
            "iterations": self.iterations,
            "pytest_returncode": self.pytest_returncode,
            "pytest_stdout_tail": self.pytest_stdout_tail,
            "pytest_stderr_tail": self.pytest_stderr_tail,
            "note": self.note,
            "mock_path_findings": list(self.mock_path_findings),
            "attribute_chain_findings": list(self.attribute_chain_findings),
        }


# pytest's status markers in -v / --tb=short output. Captures both the
# top-level test name and any parametrization suffix.
_PYTEST_STATUS_RE = re.compile(r"^(?P<status>PASSED|FAILED|ERROR|SKIPPED)\s+(?P<nodeid>\S+)")


def enforce_per_test_acceptance(
    artifact: GeneratedArtifact | dict[str, Any] | str,
    *,
    workdir: Path,
    benchmark_adapter: Any | None = None,
    pytest_timeout_seconds: float = 60.0,
    max_drop_iterations: int = 3,
    keep_minimum: int = 1,
    enabled: bool | None = None,
    extra_pytest_args: Iterable[str] | None = None,
    python_executable: str | None = None,
    trace_repair_llm_caller: Any | None = None,
    focal_source: str = "",
    focal_module_path: str = "",
    max_repairs_per_iteration: int = 5,
    mock_path_validation_enabled: bool | None = None,
    mock_path_allow_focal_import: bool = False,
    mock_path_extra_modules: tuple[str, ...] = (),
    attribute_chain_check_enabled: bool | None = None,
    attribute_chain_allow_import: bool = False,
    attribute_chain_extra_modules: tuple[str, ...] = (),
) -> PerTestAcceptanceResult:
    """Run pytest in-process against ``artifact``, drop failing tests,
    return the cleaned artifact.

    Args:
        artifact: candidate test file (raw text, dict, or
            ``GeneratedArtifact``).
        workdir: directory to run pytest in. The artifact is written to
            ``workdir / artifact.path`` (relative). The focal source must
            already be importable from ``workdir`` — this is the same
            workdir contract the existing benchmark adapters use.
        pytest_timeout_seconds: per-pytest-invocation wall cap. The full
            gate may run up to ``max_drop_iterations`` invocations.
        max_drop_iterations: fix-and-retry limit. After this many
            iterations the gate ships whatever survives; stops looping
            on syntactically-valid-but-still-failing tests so the next
            layer (downstream docker, repair loop) can take over.
        keep_minimum: refuse to drop below this many tests; if a
            candidate has fewer than this many surviving tests, ship the
            artifact unchanged (the downstream gate / harness will
            decide what to do).
        enabled: explicit override. ``None`` honors
            ``APEX_PER_TEST_ACCEPTANCE_ENABLED`` env (default ON).
            ``False`` returns ``status="skipped"`` immediately, useful
            for benchmarks that own the filter themselves.
        extra_pytest_args: passed through after the test file path.
            Useful for ``-p no:cacheprovider`` or coverage-related
            flags that the operator wants to set per-call.
        python_executable: optional override for the python interpreter.
            Defaults to ``sys.executable``.

    Returns:
        ``PerTestAcceptanceResult`` with the cleaned artifact and a
        diagnostic payload. ``shipped`` is True when an artifact (clean
        or untouched) is suitable to ship downstream.
    """

    if benchmark_adapter is None:
        try:
            from .docker_acceptance_adapter import get_docker_task_context

            ctx = get_docker_task_context()
            benchmark_adapter = getattr(ctx, "adapter", None) if ctx is not None else None
        except Exception:  # pragma: no cover - defensive
            benchmark_adapter = None

    if enabled is None:
        enabled = _is_per_test_acceptance_enabled()
    current = GeneratedArtifact.from_any(artifact)
    if not enabled:
        return PerTestAcceptanceResult(
            status="skipped",
            artifact=current,
            note="APEX_PER_TEST_ACCEPTANCE_ENABLED disabled",
        )

    syntax_ok, syntax_error = strict_syntax_check(
        current.content,
        filename=current.path or "<generated>",
    )
    if not syntax_ok:
        # Don't even attempt pytest on something that can't parse — the
        # caller's downstream W3 gate will reject it with a clearer message.
        return PerTestAcceptanceResult(
            status="syntax_error",
            artifact=current,
            note=f"strict_syntax_check failed: {syntax_error}",
        )

    test_names = test_names_in_artifact(current.content)
    if not test_names:
        return PerTestAcceptanceResult(
            status="no_tests",
            artifact=current,
            note="no test_* functions found",
        )

    # Static pre-pytest passes:
    #   * P1 step 5 — mock_path_validator (drops tests guaranteed to
    #     error inside ``patch.start()``)
    #   * P2 step 8 — attribute-chain check (drops tests that reference
    #     ``module.attr`` paths the resolved module doesn't expose)
    # Both run before pytest so we don't pay the spawn cost for
    # guaranteed errors. Both bias hard toward false-negatives.
    # If a benchmark adapter is active, these static checks must not import
    # focal modules on the host; the target environment owns dynamic checks.
    if benchmark_adapter is not None:
        mock_path_allow_focal_import = False
        attribute_chain_allow_import = False
    mock_findings_payload: list[dict[str, Any]] = []
    chain_findings_payload: list[dict[str, Any]] = []
    static_offending: set[str] = set()

    if mock_path_validation_enabled is None:
        from .mock_path_validator import is_mock_path_validation_enabled

        mock_path_validation_enabled = is_mock_path_validation_enabled()
    if mock_path_validation_enabled:
        from .mock_path_validator import validate_mock_paths

        mock_result = validate_mock_paths(
            current.content,
            allow_import=mock_path_allow_focal_import,
            extra_modules=mock_path_extra_modules,
        )
        mock_findings_payload = [
            {
                "test_name": f.test_name,
                "target": f.target,
                "call_kind": f.call_kind,
                "line": f.line,
                "reason": f.reason,
            }
            for f in mock_result.findings
        ]
        static_offending |= mock_result.offending_test_names()

    if attribute_chain_check_enabled is None:
        from .import_preflight import is_attribute_chain_check_enabled

        attribute_chain_check_enabled = is_attribute_chain_check_enabled()
    if attribute_chain_check_enabled:
        from .import_preflight import detect_undefined_attribute_chains

        chain_result = detect_undefined_attribute_chains(
            current.content,
            allow_import=attribute_chain_allow_import,
            extra_modules=attribute_chain_extra_modules,
        )
        chain_findings_payload = [
            {
                "chain": f.chain,
                "root": f.root,
                "missing_attr": f.missing_attr,
                "resolved_module": f.resolved_module,
                "line": f.line,
                "enclosing_function": f.enclosing_function,
                "reason": f.reason,
            }
            for f in chain_result.findings
        ]
        static_offending |= chain_result.offending_test_names()

    pre_pytest_drops: list[str] = []
    offending = static_offending & set(test_names)
    if offending:
        remaining = set(test_names) - offending
        if len(remaining) >= max(1, int(keep_minimum or 1)):
            next_text, dropped_static = drop_tests_from_artifact_with_report(
                current.content,
                offending,
                keep_minimum=keep_minimum,
            )
            if dropped_static:
                current = GeneratedArtifact(
                    path=current.path,
                    content=next_text,
                    metadata=dict(current.metadata),
                )
                test_names = test_names_in_artifact(current.content)
                if not test_names:
                    return PerTestAcceptanceResult(
                        status="all_dropped",
                        artifact=current,
                        dropped_tests=list(dropped_static),
                        mock_path_findings=mock_findings_payload,
                        attribute_chain_findings=chain_findings_payload,
                        note=(
                            "static pre-pytest validators dropped every test; "
                            "shipping artifact unchanged for downstream review"
                        ),
                    )
                pre_pytest_drops = list(dropped_static)

    workdir_path = Path(workdir).expanduser().resolve()
    interpreter = python_executable or sys.executable
    extra_args = tuple(extra_pytest_args or ())
    iterations_run = 0
    dropped_total: list[str] = list(pre_pytest_drops)
    last_returncode: int | None = None
    last_stdout = ""
    last_stderr = ""
    last_failing: set[str] = set()
    last_erroring: set[str] = set()
    # P2 step 9: per-test repair-attempt counts persist across iterations.
    repair_attempt_counts: dict[str, int] = {}

    iterations = max(1, int(max_drop_iterations or 1))
    for index in range(iterations):
        iterations_run = index + 1
        if benchmark_adapter is not None:
            run = _invoke_benchmark_adapter_for_per_test_acceptance(
                artifact=current,
                workdir=workdir_path,
                benchmark_adapter=benchmark_adapter,
                timeout_seconds=pytest_timeout_seconds,
            )
        else:
            run = _invoke_pytest_in_process(
                artifact=current,
                workdir=workdir_path,
                timeout_seconds=pytest_timeout_seconds,
                interpreter=interpreter,
                extra_args=extra_args,
            )
        last_returncode = run.returncode
        last_stdout = run.stdout_tail
        last_stderr = run.stderr_tail
        if run.harness_error:
            return PerTestAcceptanceResult(
                status="harness_error",
                artifact=current,
                dropped_tests=list(dropped_total),
                iterations=iterations_run,
                pytest_returncode=last_returncode,
                pytest_stdout_tail=last_stdout,
                pytest_stderr_tail=last_stderr,
                note=run.note or "pytest invocation harness error",
                mock_path_findings=mock_findings_payload,
                attribute_chain_findings=chain_findings_payload,
            )
        last_failing = set(run.failing)
        last_erroring = set(run.erroring)
        if not last_failing and not last_erroring:
            return PerTestAcceptanceResult(
                status="shipped" if dropped_total else "shipped_no_drops",
                artifact=current,
                dropped_tests=list(dropped_total),
                iterations=iterations_run,
                pytest_returncode=last_returncode,
                pytest_stdout_tail=last_stdout,
                pytest_stderr_tail=last_stderr,
                mock_path_findings=mock_findings_payload,
                attribute_chain_findings=chain_findings_payload,
            )
        # Reduce nodeids to bare function names for the AST drop.
        bad_names = _nodeid_set_to_function_names(last_failing | last_erroring)
        if not bad_names:
            # Pytest reported a failure but we couldn't pin it to a test
            # function we recognize. Bail without modifying the artifact.
            return PerTestAcceptanceResult(
                status="harness_error",
                artifact=current,
                dropped_tests=list(dropped_total),
                failing_test_names=sorted(last_failing),
                erroring_test_names=sorted(last_erroring),
                iterations=iterations_run,
                pytest_returncode=last_returncode,
                pytest_stdout_tail=last_stdout,
                pytest_stderr_tail=last_stderr,
                note="pytest reported failures but no test_ name could be matched",
                mock_path_findings=mock_findings_payload,
                attribute_chain_findings=chain_findings_payload,
            )
        # Optional: invoke LLM trace repair before dropping (P0 step 2).
        # When ``trace_repair_llm_caller`` is None, this is a no-op and
        # we proceed straight to the drop step (the original behavior).
        if trace_repair_llm_caller is not None:
            from .repair_strategies import repair_failing_tests_with_trace

            traces = extract_pytest_traces(last_stdout)
            # Only attempt repair on tests whose trace we actually have.
            repair_traces = {name: traces[name] for name in bad_names if name in traces}
            if repair_traces:
                repair_outcome = repair_failing_tests_with_trace(
                    artifact_text=current.content,
                    failing_test_traces=repair_traces,
                    focal_source=focal_source,
                    focal_module_path=focal_module_path,
                    llm_caller=trace_repair_llm_caller,
                    max_repairs_per_call=max(1, int(max_repairs_per_iteration)),
                    kind_attempt_counts=repair_attempt_counts,
                )
                if repair_outcome.changed:
                    current = GeneratedArtifact(
                        path=current.path,
                        content=repair_outcome.artifact_text,
                        metadata=dict(current.metadata),
                    )
                    # Re-run pytest to see whether the repair worked.
                    # Only count names that are STILL failing in the
                    # next iteration as drop candidates.
                    continue
        remaining = set(test_names_in_artifact(current.content)) - bad_names
        if len(remaining) < max(1, int(keep_minimum or 1)):
            return PerTestAcceptanceResult(
                status="all_dropped",
                artifact=current,
                dropped_tests=sorted(set(dropped_total) | bad_names),
                failing_test_names=sorted(last_failing),
                erroring_test_names=sorted(last_erroring),
                iterations=iterations_run,
                pytest_returncode=last_returncode,
                pytest_stdout_tail=last_stdout,
                pytest_stderr_tail=last_stderr,
                note=(
                    "dropping the failing tests would leave fewer than "
                    f"keep_minimum={keep_minimum}; shipping artifact unchanged"
                ),
                mock_path_findings=mock_findings_payload,
                attribute_chain_findings=chain_findings_payload,
            )
        next_text, dropped = drop_tests_from_artifact_with_report(
            current.content,
            bad_names,
            keep_minimum=keep_minimum,
        )
        if not dropped:
            # AST drop transformer couldn't actually remove the named tests
            # (rare — usually means the names didn't match function defs).
            return PerTestAcceptanceResult(
                status="harness_error",
                artifact=current,
                dropped_tests=list(dropped_total),
                failing_test_names=sorted(last_failing),
                erroring_test_names=sorted(last_erroring),
                iterations=iterations_run,
                pytest_returncode=last_returncode,
                pytest_stdout_tail=last_stdout,
                pytest_stderr_tail=last_stderr,
                note="AST drop transformer made no progress",
                mock_path_findings=mock_findings_payload,
                attribute_chain_findings=chain_findings_payload,
            )
        dropped_total.extend(dropped)
        current = GeneratedArtifact(
            path=current.path,
            content=next_text,
            metadata=dict(current.metadata),
        )
    # Loop exhausted. Ship whatever we have — downstream gates will
    # decide whether the residue is acceptable.
    return PerTestAcceptanceResult(
        status="shipped" if dropped_total else "shipped_no_drops",
        artifact=current,
        dropped_tests=list(dropped_total),
        failing_test_names=sorted(last_failing),
        erroring_test_names=sorted(last_erroring),
        iterations=iterations_run,
        pytest_returncode=last_returncode,
        pytest_stdout_tail=last_stdout,
        pytest_stderr_tail=last_stderr,
        note="iteration cap hit; shipping with residual failures",
        mock_path_findings=mock_findings_payload,
        attribute_chain_findings=chain_findings_payload,
    )


def _is_per_test_acceptance_enabled() -> bool:
    raw = os.environ.get("APEX_PER_TEST_ACCEPTANCE_ENABLED")
    if raw is None:
        return True  # default ON — operators opt out explicitly
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class _PytestRun:
    returncode: int
    stdout_tail: str
    stderr_tail: str
    failing: list[str] = field(default_factory=list)
    erroring: list[str] = field(default_factory=list)
    harness_error: bool = False
    note: str = ""


def _invoke_benchmark_adapter_for_per_test_acceptance(
    *,
    artifact: GeneratedArtifact,
    workdir: Path,
    benchmark_adapter: Any,
    timeout_seconds: float,
) -> _PytestRun:
    """Run per-test acceptance through the active benchmark adapter."""

    try:
        try:
            raw_run = benchmark_adapter.run_unfiltered(
                artifact,
                workdir,
                timeout_seconds=float(timeout_seconds),
            )
        except TypeError:
            raw_run = benchmark_adapter.run_unfiltered(artifact, workdir)
    except Exception as exc:  # pragma: no cover - adapter boundary
        return _PytestRun(
            returncode=1,
            stdout_tail="",
            stderr_tail="",
            harness_error=True,
            note=f"benchmark adapter raised: {type(exc).__name__}: {exc}",
        )
    run = _coerce_run(raw_run)
    failing = sorted(run.failing_test_names)
    erroring = sorted(run.errored_test_names)
    status = str(run.status or "").lower()
    setup_statuses = {
        "artifact_failed",
        "collection_failed",
        "harness_error",
        "harness_log_missing",
        "setup_failed",
        "syntax_error",
    }
    harness_error = False
    note = run.diagnostic
    if status in setup_statuses or str(run.failure_taxonomy or "").lower() in setup_statuses:
        harness_error = True
        note = note or run.failure_taxonomy or status
    elif status not in {"pass", "passed", "ok"} and not failing and not erroring:
        harness_error = True
        note = note or "benchmark adapter failed without per-test failures"
    return _PytestRun(
        returncode=run.returncode
        if run.returncode is not None
        else (0 if status in {"pass", "passed", "ok"} else 1),
        stdout_tail=run.stdout_tail,
        stderr_tail=run.stderr_tail,
        failing=failing,
        erroring=erroring,
        harness_error=harness_error,
        note=note,
    )


def _invoke_pytest_in_process(
    *,
    artifact: GeneratedArtifact,
    workdir: Path,
    timeout_seconds: float,
    interpreter: str,
    extra_args: tuple[str, ...],
) -> _PytestRun:
    """Write the artifact into a tmp dir under ``workdir`` and run pytest
    against it. The tmp dir gets the existing workdir on ``sys.path`` so
    the focal module imports resolve naturally."""

    if not workdir.exists():
        return _PytestRun(
            returncode=-1,
            stdout_tail="",
            stderr_tail="",
            harness_error=True,
            note=f"workdir does not exist: {workdir}",
        )
    test_rel_path = artifact.path or "tests/test_apex_per_test_gate.py"
    if Path(test_rel_path).is_absolute():
        # Defensive: collapse absolute paths so the tmp-dir write stays
        # inside ``workdir``. The artifact path is meant to be a relative
        # hint for the harness; treat the basename as authoritative if
        # someone passed an absolute path through.
        test_rel_path = Path(test_rel_path).name
    target_path = workdir / test_rel_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the original on-disk file so we can restore it after the
    # gate runs (the workdir may already host the operator's version).
    original_text: str | None = None
    if target_path.exists():
        try:
            original_text = target_path.read_text(encoding="utf-8")
        except OSError:
            original_text = None
    target_path.write_text(artifact.content, encoding="utf-8")

    # Force the workdir to be importable so `from focal_module import X`
    # in the test resolves. Honor existing PYTHONPATH if set.
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    parts = [str(workdir)]
    if existing_pythonpath:
        parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(parts)

    cmd = [
        interpreter,
        "-m",
        "pytest",
        str(target_path),
        "-v",
        "--tb=short",
        "-p",
        "no:cacheprovider",
        # Don't write per-test JSON reports; we parse stdout.
        # Surface short failures so we can pin per-test status.
    ]
    cmd.extend(extra_args)
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(workdir),
            env=env,
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout_seconds)),
            check=False,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        returncode = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = (
            (exc.stdout or "").decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else str(exc.stdout or "")
        )
        stderr = (
            (exc.stderr or "").decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else str(exc.stderr or "")
        )
        returncode = -9
        timed_out = True
    except (OSError, ValueError) as exc:  # pragma: no cover - defensive
        # Restore original file before bailing.
        if original_text is not None:
            try:
                target_path.write_text(original_text, encoding="utf-8")
            except OSError:
                pass
        return _PytestRun(
            returncode=-1,
            stdout_tail="",
            stderr_tail="",
            harness_error=True,
            note=f"pytest spawn failed: {type(exc).__name__}: {exc}",
        )

    # Restore the original on-disk file (if any). The cleaned artifact
    # lives in the result; we don't need to leave our test file behind.
    if original_text is not None:
        try:
            target_path.write_text(original_text, encoding="utf-8")
        except OSError:
            pass
    else:
        try:
            target_path.unlink()
        except OSError:
            pass

    if timed_out:
        return _PytestRun(
            returncode=returncode,
            stdout_tail=stdout[-4000:],
            stderr_tail=stderr[-4000:],
            harness_error=True,
            note=f"pytest timed out after {timeout_seconds}s",
        )

    failing, erroring = _parse_pytest_status_lines(stdout)
    # Returncode 5 = no tests collected. Treat as harness_error so the
    # caller can bail; the candidate would also fail downstream.
    if returncode == 5:
        return _PytestRun(
            returncode=returncode,
            stdout_tail=stdout[-4000:],
            stderr_tail=stderr[-4000:],
            harness_error=True,
            note="pytest collected no tests",
        )
    # Returncodes 2 (collection error), 3 (interrupt), 4 (internal):
    # treat as harness_error so we don't keep retrying.
    if returncode in {2, 3, 4} and not failing and not erroring:
        return _PytestRun(
            returncode=returncode,
            stdout_tail=stdout[-4000:],
            stderr_tail=stderr[-4000:],
            harness_error=True,
            note=f"pytest non-test exit ({returncode})",
        )
    return _PytestRun(
        returncode=returncode,
        stdout_tail=stdout[-4000:],
        stderr_tail=stderr[-4000:],
        failing=failing,
        erroring=erroring,
        harness_error=False,
    )


def _parse_pytest_status_lines(stdout: str) -> tuple[list[str], list[str]]:
    failing: list[str] = []
    erroring: list[str] = []
    seen: set[str] = set()
    for line in (stdout or "").splitlines():
        match = _PYTEST_STATUS_RE.match(line.strip())
        if match is None:
            continue
        status = match.group("status").upper()
        nodeid = match.group("nodeid").strip()
        if not nodeid or nodeid in seen:
            continue
        seen.add(nodeid)
        if status == "FAILED":
            failing.append(nodeid)
        elif status == "ERROR":
            erroring.append(nodeid)
    return failing, erroring


def _nodeid_set_to_function_names(nodeids: Iterable[str]) -> set[str]:
    """Reduce pytest nodeids (path::TestClass::test_x[param]) to bare
    test function names so the AST drop transformer can find them."""

    out: set[str] = set()
    for nodeid in nodeids:
        text = str(nodeid or "")
        if not text:
            continue
        # Strip the file portion if present.
        _, sep, rest = text.partition("::")
        candidate = rest if sep else text
        # Walk segments; keep the rightmost test_ segment.
        chosen = ""
        for segment in candidate.split("::"):
            cleaned = segment.split("[", 1)[0].strip()
            if cleaned.startswith("test_"):
                chosen = cleaned
        if chosen:
            out.add(chosen)
    return out


# Pytest's ``--tb=short`` output uses lines of underscores around each
# failed-test header: ``__________________ test_foo __________________``.
_PYTEST_TRACE_HEADER_RE = re.compile(r"^_{3,}\s+(\S+?)\s+_{3,}\s*$")
# The trailing summary line ``=== short test summary info ===`` marks the
# end of all traces; we stop reading once we see it.
_PYTEST_SUMMARY_LINE_RE = re.compile(r"^=+\s*(short test summary info|FAILURES|ERRORS)\s*=+")


def extract_pytest_traces(stdout: str) -> dict[str, str]:
    """Parse pytest's ``--tb=short`` stdout into a {test_name: trace} dict.

    Used by the execution-trace repair loop (P0 step 2): each failing
    test's trace becomes the context the LLM sees when asked to repair
    the test. Reduces nodeid → bare test_ function name to match what
    ``_nodeid_set_to_function_names`` produces, so callers can pair the
    two by key.

    Robust to mixed FAILURES + ERRORS sections; both produce the same
    underscored header pattern. Returns ``{}`` when stdout has no
    parseable trace blocks.
    """

    traces: dict[str, list[str]] = {}
    current: str | None = None
    in_traces = False
    for raw in (stdout or "").splitlines():
        line = raw.rstrip()
        # Section markers: stop at the trailing summary, start at a header.
        header = _PYTEST_TRACE_HEADER_RE.match(line)
        if header is not None:
            in_traces = True
            nodeid = header.group(1)
            # Reduce nodeid to bare test_ function name.
            chosen = ""
            for segment in str(nodeid).split("::"):
                cleaned = segment.split("[", 1)[0].strip()
                if cleaned.startswith("test_"):
                    chosen = cleaned
            current = chosen if chosen else nodeid
            traces.setdefault(current, [])
            continue
        # Hit the final summary block: stop accumulating per-test trace.
        if in_traces and _PYTEST_SUMMARY_LINE_RE.match(line):
            in_traces = False
            current = None
            continue
        if current is None:
            continue
        traces[current].append(raw)
    return {name: "\n".join(lines).strip() for name, lines in traces.items() if lines}
