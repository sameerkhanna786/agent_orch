"""Reference orchestration-as-code programs the engine runs."""

from .best_of_n import (
    SolveResult,
    WorkerSpec,
    best_of_n_solve,
    make_pytest_score_fn,
)

__all__ = ["best_of_n_solve", "WorkerSpec", "SolveResult", "make_pytest_score_fn"]
