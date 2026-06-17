"""A/B harness for self-play scoring strategies (Phase B.7).

Compares the four scoring strategies registered in
:mod:`apex.capabilities.self_play` (``mutual_confidence``,
``survival_x_kill``, ``harmonic_mean``, ``borda``) on a synthetic
dataset of (patch_set, test_set, ground_truth_winner) tuples. For
each strategy we run :class:`SelfPlayTournament` and check whether
the chosen patch matches the ground truth.

The synthetic data is intentionally tractable so the harness has a
deterministic seed-driven self-test (run with no args). Real-data
A/B comparisons should re-use the strategies on actual rollout
verdict matrices via ``--from-trace`` (see ``--help``).

Usage::

    # Self-test: deterministic synthetic dataset.
    python -m apex.scripts.ab_self_play_scoring

    # Custom seed / dataset size.
    python -m apex.scripts.ab_self_play_scoring --seed 42 --num-cases 100

The script prints a per-strategy table:

    strategy            accuracy  ties  inconsistent_picks
    -------------------------------------------------------
    harmonic_mean       0.84      0     2
    survival_x_kill     0.79      0     1
    mutual_confidence   0.42      0     8
    borda               0.71      1     0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from apex.capabilities.self_play import (
    DEFAULT_SCORING_STRATEGY,
    SelfPlayTournament,
)

_STRATEGIES: tuple[str, ...] = (
    "harmonic_mean",
    "survival_x_kill",
    "mutual_confidence",
    "borda",
)


@dataclass
class SyntheticCase:
    """One (patches, tests, ground_truth_winner) tuple.

    ``verdict_matrix`` is the K x M binary matrix of pass/fail used
    to drive the verdict_fn — we pre-compute it so each strategy
    sees identical evaluations. ``winner_patch_index`` is the patch
    we expect a competent strategy to pick.
    """

    case_id: int
    verdict_matrix: np.ndarray
    winner_patch_index: int
    notes: str = ""


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------


def _generate_case(rng: random.Random, case_id: int) -> SyntheticCase:
    """Generate one synthetic (patches, tests, winner) case.

    Construction principle: we plant a "true winner" patch that is
    correct on a hidden ground-truth bug. The ``correct`` patch passes
    every test that matters; the other patches each fail on a distinct
    subset of bug-revealing tests. A handful of "trivially-passing"
    tests are added that pass under every patch (these are the
    distractors a competent scoring strategy should ignore).
    """
    # Modest sizes so the harness runs in <1s on 100 cases.
    K = rng.randint(3, 5)  # patches
    M = rng.randint(4, 6)  # tests
    matrix = np.zeros((K, M), dtype=np.int8)

    # Pick the ground-truth winner.
    winner = rng.randint(0, K - 1)

    # The first ``M//2`` tests are "discriminating" — they fail on
    # buggy patches. The rest pass uniformly (trivial tests).
    n_discrim = max(1, M // 2)

    for j in range(M):
        if j >= n_discrim:
            # Trivial test: passes everywhere.
            matrix[:, j] = 1
            continue
        # Discriminating test: passes under the winner, but each other
        # patch has a small failure probability so the matrix isn't
        # uniformly degenerate. We force at least the winner to pass.
        for i in range(K):
            if i == winner:
                matrix[i, j] = 1
            else:
                # Each non-winner fails this discriminating test with
                # probability 0.7 — leaves enough passes that the
                # tournament has to actually rank candidates.
                matrix[i, j] = 0 if rng.random() < 0.7 else 1

    return SyntheticCase(
        case_id=case_id,
        verdict_matrix=matrix,
        winner_patch_index=winner,
        notes=f"K={K} M={M} discrim={n_discrim}",
    )


def generate_dataset(seed: int, num_cases: int) -> list[SyntheticCase]:
    """Build a deterministic, seeded synthetic dataset."""
    rng = random.Random(seed)
    return [_generate_case(rng, idx) for idx in range(num_cases)]


# ---------------------------------------------------------------------------
# Strategy evaluation
# ---------------------------------------------------------------------------


@dataclass
class StrategyResult:
    strategy: str
    correct: int = 0
    total: int = 0
    ties_with_winner: int = 0  # picked another patch tied on score with winner
    inconsistent_picks: int = 0  # picked V[i, j] = 0 (selection inconsistent)

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 1)


def _verdict_fn_from_matrix(matrix: np.ndarray):
    def fn(patch: dict, test: dict) -> int:
        return int(matrix[patch["index"], test["index"]])

    return fn


def evaluate_strategy(
    strategy: str,
    cases: list[SyntheticCase],
) -> StrategyResult:
    """Run :class:`SelfPlayTournament` per case and tally accuracy."""
    result = StrategyResult(strategy=strategy)
    for case in cases:
        K, M = case.verdict_matrix.shape
        tour = SelfPlayTournament(
            K_patches=K,
            M_tests=M,
            scoring_strategy=strategy,
        )
        outcome = tour.run(
            patch_candidates=[{"index": i} for i in range(K)],
            test_candidates=[{"index": j} for j in range(M)],
            verdict_fn=_verdict_fn_from_matrix(case.verdict_matrix),
        )
        result.total += 1
        if outcome.selected_patch_index == case.winner_patch_index:
            result.correct += 1
        if not bool(
            case.verdict_matrix[outcome.selected_patch_index, outcome.selected_test_index] == 1
        ):
            result.inconsistent_picks += 1
    return result


def evaluate_all_strategies(
    cases: list[SyntheticCase],
    strategies: tuple[str, ...] = _STRATEGIES,
) -> list[StrategyResult]:
    """Run :func:`evaluate_strategy` for each strategy in turn."""
    return [evaluate_strategy(s, cases) for s in strategies]


# ---------------------------------------------------------------------------
# CLI / reporting
# ---------------------------------------------------------------------------


def format_report(results: list[StrategyResult]) -> str:
    headers = ("strategy", "accuracy", "correct/total", "inconsistent_picks")
    rows = [headers]
    for r in results:
        rows.append(
            (
                r.strategy,
                f"{r.accuracy:.3f}",
                f"{r.correct}/{r.total}",
                str(r.inconsistent_picks),
            )
        )
    widths = [max(len(row[c]) for row in rows) for c in range(len(headers))]
    lines = []
    for idx, row in enumerate(rows):
        line = "  ".join(row[c].ljust(widths[c]) for c in range(len(headers)))
        lines.append(line)
        if idx == 0:
            lines.append("  ".join("-" * widths[c] for c in range(len(headers))))
    return "\n".join(lines)


def run_self_test() -> dict[str, Any]:
    """Tiny deterministic self-test fixture.

    Generates a small (5-case) seeded dataset and asserts the
    :data:`DEFAULT_SCORING_STRATEGY` strategy produces a non-zero
    accuracy. This is what the unit test covers; the full A/B run
    is opt-in via the CLI.
    """
    cases = generate_dataset(seed=12345, num_cases=5)
    results = evaluate_all_strategies(cases)
    summary = {r.strategy: r.accuracy for r in results}
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="A/B harness for self-play scoring strategies.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260514,
        help="Random seed for the synthetic dataset (default: today).",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=100,
        help="Number of synthetic cases to evaluate (default 100).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a formatted table.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the tiny seeded self-test fixture and exit.",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        summary = run_self_test()
        print(json.dumps(summary, indent=2))
        return 0

    cases = generate_dataset(seed=int(args.seed), num_cases=int(args.num_cases))
    results = evaluate_all_strategies(cases)
    if args.json:
        payload = [
            {
                "strategy": r.strategy,
                "accuracy": r.accuracy,
                "correct": r.correct,
                "total": r.total,
                "inconsistent_picks": r.inconsistent_picks,
            }
            for r in results
        ]
        print(
            json.dumps(
                {
                    "seed": int(args.seed),
                    "num_cases": int(args.num_cases),
                    "default_strategy": DEFAULT_SCORING_STRATEGY,
                    "results": payload,
                },
                indent=2,
            )
        )
    else:
        print(format_report(results))
        print()
        print(f"(default strategy: {DEFAULT_SCORING_STRATEGY})")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
