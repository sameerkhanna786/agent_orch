"""Benchmark runner entry points used by APEX evaluation scripts."""

from __future__ import annotations

__all__ = [
    "TestGenEvalLiteRunConfig",
    "build_testgenevallite_command",
    "pre_flight_for_testgen",
    "run_testgenevallite",
]


def __getattr__(name: str):
    if name in {"pre_flight_for_testgen"}:
        from ._preflight import pre_flight_for_testgen

        return pre_flight_for_testgen
    if name in {
        "TestGenEvalLiteRunConfig",
        "build_testgenevallite_command",
        "run_testgenevallite",
    }:
        from .testgenevallite import (
            TestGenEvalLiteRunConfig,
            build_testgenevallite_command,
            run_testgenevallite,
        )

        return {
            "TestGenEvalLiteRunConfig": TestGenEvalLiteRunConfig,
            "build_testgenevallite_command": build_testgenevallite_command,
            "run_testgenevallite": run_testgenevallite,
        }[name]
    raise AttributeError(name)
