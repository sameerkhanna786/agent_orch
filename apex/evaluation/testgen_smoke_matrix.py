"""Shared smoke-matrix reporting for test-generation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TestgenSmokeRecord:
    task_id: str
    benchmark: str = "unknown"
    pass_at_1: float = 0.0
    all_pass_at_1: float = 0.0
    generated_test_count: int = 0
    meaningful_test_count: int = 0
    oracle_grounded_count: int = 0
    coverage_ratio: float = 0.0
    mutation_score: float = 0.0
    assertion_effective: bool = False
    failure_class: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "benchmark": self.benchmark,
            "pass_at_1": self.pass_at_1,
            "all_pass_at_1": self.all_pass_at_1,
            "generated_test_count": self.generated_test_count,
            "meaningful_test_count": self.meaningful_test_count,
            "oracle_grounded_count": self.oracle_grounded_count,
            "coverage_ratio": self.coverage_ratio,
            "mutation_score": self.mutation_score,
            "assertion_effective": self.assertion_effective,
            "failure_class": self.failure_class,
        }


@dataclass(frozen=True)
class TestgenSmokeMatrix:
    records: list[TestgenSmokeRecord] = field(default_factory=list)

    @property
    def task_count(self) -> int:
        return len(self.records)

    @property
    def mean_pass_at_1(self) -> float:
        return _mean([record.pass_at_1 for record in self.records])

    @property
    def mean_all_pass_at_1(self) -> float:
        return _mean([record.all_pass_at_1 for record in self.records])

    @property
    def mean_coverage_ratio(self) -> float:
        return _mean([record.coverage_ratio for record in self.records])

    @property
    def mean_mutation_score(self) -> float:
        return _mean([record.mutation_score for record in self.records])

    @property
    def total_meaningful_tests(self) -> int:
        return sum(record.meaningful_test_count for record in self.records)

    @property
    def total_oracle_grounded(self) -> int:
        return sum(record.oracle_grounded_count for record in self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_count": self.task_count,
            "mean_pass_at_1": round(self.mean_pass_at_1, 4),
            "mean_all_pass_at_1": round(self.mean_all_pass_at_1, 4),
            "mean_coverage_ratio": round(self.mean_coverage_ratio, 4),
            "mean_mutation_score": round(self.mean_mutation_score, 4),
            "total_meaningful_tests": self.total_meaningful_tests,
            "total_oracle_grounded": self.total_oracle_grounded,
            "records": [record.to_dict() for record in self.records],
        }


def smoke_record_from_result(
    result: Any,
    *,
    benchmark: str = "unknown",
) -> TestgenSmokeRecord:
    diagnostics = dict(getattr(result, "diagnostics", {}) or {})
    quality = dict(diagnostics.get("test_quality_summary") or {})
    generation = dict(diagnostics.get("generation") or {})
    oracle = dict(generation.get("oracle_grounding") or diagnostics.get("oracle_grounding") or {})
    assertion = dict(diagnostics.get("assertion_mutation_summary") or {})
    apex_validation = dict(diagnostics.get("apex_validation") or {})
    failure_payload = diagnostics.get("failure_classification")
    failure_classification = dict(failure_payload) if isinstance(failure_payload, dict) else {}
    failure = str(
        apex_validation.get("failure_class") or failure_classification.get("failure_class") or ""
    )
    return TestgenSmokeRecord(
        task_id=str(getattr(result, "instance_id", "") or ""),
        benchmark=benchmark,
        pass_at_1=float(getattr(result, "pass_at_1", 0.0) or 0.0),
        all_pass_at_1=float(getattr(result, "all_pass_at_1", 0.0) or 0.0),
        generated_test_count=int(getattr(result, "generated_test_count", 0) or 0),
        meaningful_test_count=int(quality.get("meaningful_test_count") or 0),
        oracle_grounded_count=int(oracle.get("grounded_count") or 0),
        coverage_ratio=float(getattr(result, "coverage_ratio", 0.0) or 0.0),
        mutation_score=float(getattr(result, "mutation_score", 0.0) or 0.0),
        assertion_effective=bool(assertion.get("assertion_effective")),
        failure_class=failure,
    )


def build_smoke_matrix(results: list[Any], *, benchmark: str = "unknown") -> TestgenSmokeMatrix:
    """Build a common metric matrix from benchmark-specific task results."""

    return TestgenSmokeMatrix(
        records=[smoke_record_from_result(result, benchmark=benchmark) for result in results]
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
