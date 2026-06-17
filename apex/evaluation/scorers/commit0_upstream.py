"""Commit0 upstream-canonical scorer.

Implements :class:`apex.core.fairness_audit.BenchmarkScorerProtocol` for
the upstream-canonical Commit0 scoring path. The upstream path is the
official ``commit0`` Docker harness (``commit0.harness.run_pytest_ids``)
invoked unmodified — the only published number in the original Commit0
paper. APEX wraps it in
:meth:`apex.evaluation.commit0_benchmark.Commit0BenchmarkRunner._evaluate_repo_official`
and tags the resulting :class:`Commit0Evaluation` with
``score_source="upstream_audit"``.

Inputs to :meth:`Commit0UpstreamScorer.score_task`:

* ``task``: a :class:`apex.evaluation.commit0_benchmark.Commit0Task`.
* ``apex_artifacts``: a dict carrying the pre-computed evaluations
  under the key ``"upstream"`` (see the matching docstring on
  :class:`apex.evaluation.scorers.commit0_private.Commit0PrivateScorer`).

Outputs (flat metric mapping, all numeric so they participate in the
fairness delta):

* ``pass_rate`` (float in [0, 1]): the published per-task pass rate.
* ``returncode`` (int): the canonical shell returncode from the docker
  harness invocation.
* ``passed`` (int): pytest-counted passing test count.
* ``failed`` (int): pytest-counted failing test count.
* ``errors`` (int): pytest-counted error count.
* ``score_source`` (str, non-numeric): provenance string;
  ``"upstream_audit"`` for canonical audit numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .commit0_private import _DictEvaluationView, _no_evaluation_payload, _observed_returncode

logger = logging.getLogger("apex.scorers.commit0_upstream")


@dataclass
class Commit0UpstreamScorer:
    """Score a Commit0 task with upstream-canonical (audit) rules.

    Just like :class:`Commit0PrivateScorer`, this is a pure function of
    the pre-computed evaluation in ``apex_artifacts["upstream"]``. It
    does not invoke the docker harness itself.
    """

    name: str = "commit0_upstream"

    def score_task(self, task: Any, apex_artifacts: Any) -> dict[str, Any]:
        evaluation = self._extract(apex_artifacts)
        if evaluation is None:
            return _no_evaluation_payload("commit0_upstream_eval_missing")
        observed_returncode = _observed_returncode(evaluation)
        return {
            "pass_rate": float(getattr(evaluation, "pass_rate", 0.0) or 0.0),
            "returncode": observed_returncode,
            "raw_returncode": observed_returncode,
            "scored_returncode": int(
                getattr(evaluation, "scored_returncode", observed_returncode)
                if getattr(evaluation, "scored_returncode", None) is not None
                else observed_returncode
            ),
            "passed": int(getattr(evaluation, "passed", 0) or 0),
            "failed": int(getattr(evaluation, "failed", 0) or 0),
            "errors": int(getattr(evaluation, "errors", 0) or 0),
            "score_source": str(
                getattr(evaluation, "score_source", "upstream_audit") or "upstream_audit"
            ),
        }

    @staticmethod
    def _extract(apex_artifacts: Any) -> Any:
        if apex_artifacts is None:
            return None
        if isinstance(apex_artifacts, dict):
            target = apex_artifacts.get("upstream")
            if target is None:
                return None
            if isinstance(target, dict):
                return _DictEvaluationView(target)
            return target
        return None


def make_commit0_upstream_scorer() -> Commit0UpstreamScorer:
    """Factory mirroring the other benchmark scorer factories."""
    return Commit0UpstreamScorer()


__all__ = [
    "Commit0UpstreamScorer",
    "make_commit0_upstream_scorer",
]
