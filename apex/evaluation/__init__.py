"""Benchmark and evaluation utilities."""

# Type-checking imports below mirror the lazy public export map.
# ruff: noqa: F401

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_EXPORTS = {
    "BenchmarkReport": ("benchmark", "BenchmarkReport"),
    "BenchmarkRunner": ("benchmark", "BenchmarkRunner"),
    "BenchmarkTask": ("benchmark", "BenchmarkTask"),
    "BenchmarkTaskResult": ("benchmark", "BenchmarkTaskResult"),
    "compare_benchmark_reports": ("compare", "compare_benchmark_reports"),
    "render_benchmark_comparison_markdown": ("compare", "render_benchmark_comparison_markdown"),
    "COMMIT0_LITE_REPOS": ("commit0_benchmark", "COMMIT0_LITE_REPOS"),
    "Commit0BenchmarkReport": ("commit0_benchmark", "Commit0BenchmarkReport"),
    "Commit0BenchmarkRunner": ("commit0_benchmark", "Commit0BenchmarkRunner"),
    "Commit0Evaluation": ("commit0_benchmark", "Commit0Evaluation"),
    "Commit0Task": ("commit0_benchmark", "Commit0Task"),
    "Commit0TaskResult": ("commit0_benchmark", "Commit0TaskResult"),
    # Real-world TDD entry points (decoupled from benchmark task shape).
    # These are the canonical public APIs for IDE plugins, CI gates, and
    # agentic-coding orchestrators that want APEX's F2P / mutation /
    # minimization signals without the SWE-Bench Pro task object.
    "evaluate_f2p": ("f2p_oracle", "evaluate_f2p"),
    "DualStateTask": ("f2p_oracle", "DualStateTask"),
    "evaluate_dual_state_task": ("f2p_oracle", "evaluate_dual_state_task"),
    "evaluate_f2p_on_sandboxes": ("f2p_oracle", "evaluate_f2p_on_sandboxes"),
    "evaluate_tdd_iteration": ("f2p_oracle", "evaluate_tdd_iteration"),
    "evaluate_mutation_score": ("mutation_engine", "evaluate_mutation_score"),
    "generate_mutants": ("mutation_engine", "generate_mutants"),
    "minimize_suite": ("test_minimizer", "minimize_suite"),
    "MinimizationReport": ("test_minimizer", "MinimizationReport"),
    "analyze_test_artifact_quality": (
        "test_quality",
        "analyze_test_artifact_quality",
    ),
    "analyze_test_artifacts_quality": (
        "test_quality",
        "analyze_test_artifacts_quality",
    ),
    "TestArtifactQuality": ("test_quality", "TestArtifactQuality"),
    "TestQualityIssue": ("test_quality", "TestQualityIssue"),
    "TestQualityReport": ("test_quality", "TestQualityReport"),
    "OracleGroundingReport": ("testgen_oracle_grounding", "OracleGroundingReport"),
    "ground_oracles_for_testgen": ("testgen_oracle_grounding", "ground_oracles_for_testgen"),
    "render_oracle_grounding_block": ("testgen_oracle_grounding", "render_oracle_grounding_block"),
    "TestgenSmokeMatrix": ("testgen_smoke_matrix", "TestgenSmokeMatrix"),
    "TestgenSmokeRecord": ("testgen_smoke_matrix", "TestgenSmokeRecord"),
    "build_smoke_matrix": ("testgen_smoke_matrix", "build_smoke_matrix"),
    "AssertionMutationReport": ("assertion_mutation", "AssertionMutationReport"),
    "evaluate_assertion_effect_in_loop": (
        "assertion_mutation",
        "evaluate_assertion_effect_in_loop",
    ),
    "TestStabilityReport": ("test_stability", "TestStabilityReport"),
    "evaluate_test_stability": ("test_stability", "evaluate_test_stability"),
    "classify_iteration_feedback": (
        "iteration_feedback",
        "classify_iteration_feedback",
    ),
    "render_iteration_feedback_prompt_block": (
        "iteration_feedback",
        "render_iteration_feedback_prompt_block",
    ),
    "IterationFeedback": ("iteration_feedback", "IterationFeedback"),
    # Phase I.4: surrogate-patch test selection (e-Otter++ pattern)
    "evaluate_via_surrogate_oracle": (
        "surrogate_oracle",
        "evaluate_via_surrogate_oracle",
    ),
    "SurrogateOracleReport": ("surrogate_oracle", "SurrogateOracleReport"),
    "SurrogateFixCandidate": ("surrogate_oracle", "SurrogateFixCandidate"),
    # Phase I.9: TestGenEval benchmark adapter
    "TestGenEvalTask": ("testgeneval_benchmark", "TestGenEvalTask"),
    "TestGenEvalTaskResult": (
        "testgeneval_benchmark",
        "TestGenEvalTaskResult",
    ),
    "TestGenEvalReport": ("testgeneval_benchmark", "TestGenEvalReport"),
    "evaluate_testgeneval_task": (
        "testgeneval_benchmark",
        "evaluate_testgeneval_task",
    ),
    "run_testgeneval": ("testgeneval_benchmark", "run_testgeneval"),
    "load_testgeneval_tasks_from_json": (
        "testgeneval_benchmark",
        "load_tasks_from_json",
    ),
}

__all__ = list(_EXPORTS)

if TYPE_CHECKING:
    from .assertion_mutation import (
        AssertionMutationReport,
        evaluate_assertion_effect_in_loop,
    )
    from .benchmark import BenchmarkReport, BenchmarkRunner, BenchmarkTask, BenchmarkTaskResult
    from .commit0_benchmark import (
        COMMIT0_LITE_REPOS,
        Commit0BenchmarkReport,
        Commit0BenchmarkRunner,
        Commit0Evaluation,
        Commit0Task,
        Commit0TaskResult,
    )
    from .compare import compare_benchmark_reports, render_benchmark_comparison_markdown
    from .f2p_oracle import (
        DualStateTask,
        evaluate_dual_state_task,
        evaluate_f2p,
        evaluate_f2p_on_sandboxes,
        evaluate_tdd_iteration,
    )
    from .iteration_feedback import (
        IterationFeedback,
        classify_iteration_feedback,
        render_iteration_feedback_prompt_block,
    )
    from .mutation_engine import evaluate_mutation_score, generate_mutants
    from .surrogate_oracle import (
        SurrogateFixCandidate,
        SurrogateOracleReport,
        evaluate_via_surrogate_oracle,
    )
    from .test_minimizer import (
        MinimizationReport,
        minimize_suite,
    )
    from .test_quality import (
        TestArtifactQuality,
        TestQualityIssue,
        TestQualityReport,
        analyze_test_artifact_quality,
        analyze_test_artifacts_quality,
    )
    from .test_stability import TestStabilityReport, evaluate_test_stability
    from .testgen_oracle_grounding import (
        OracleGroundingReport,
        ground_oracles_for_testgen,
        render_oracle_grounding_block,
    )
    from .testgen_smoke_matrix import (
        TestgenSmokeMatrix,
        TestgenSmokeRecord,
        build_smoke_matrix,
    )
    from .testgeneval_benchmark import (
        TestGenEvalReport,
        TestGenEvalTask,
        TestGenEvalTaskResult,
        evaluate_testgeneval_task,
        run_testgeneval,
    )
    from .testgeneval_benchmark import (
        load_tasks_from_json as load_testgeneval_tasks_from_json,
    )


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
