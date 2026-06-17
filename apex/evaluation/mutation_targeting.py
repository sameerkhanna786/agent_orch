"""Mutation-aware assertion-shape selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AssertionShape:
    name: str
    strength: int
    rationale: str


def choose_assertion_shape(value: Any, *, result_type: str = "") -> AssertionShape:
    """Choose the strongest safe assertion shape for an observed value."""

    normalized_type = str(result_type or type(value).__name__).lower()
    if "ndarray" in normalized_type or "array" in normalized_type:
        return AssertionShape(
            name="numpy_assert_allclose",
            strength=5,
            rationale="array contents kill more mutants than shape-only checks",
        )
    if isinstance(value, float):
        return AssertionShape(
            name="pytest_approx",
            strength=4,
            rationale="numeric tolerance preserves validity while checking value",
        )
    if isinstance(value, (list, tuple, dict, str, int, bool, type(None))):
        return AssertionShape(
            name="exact_equality",
            strength=5,
            rationale="exact observed value is mutation-discriminating",
        )
    return AssertionShape(
        name="repr_equality",
        strength=3,
        rationale="repr equality is safer for non-JSON objects than identity checks",
    )
