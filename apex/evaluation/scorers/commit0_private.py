"""Commit0 APEX-private scorer.

Implements :class:`apex.core.fairness_audit.BenchmarkScorerProtocol` for
the APEX-private Commit0 scoring path. The APEX-private path is the
local pytest-json-report-driven evaluation produced by
:meth:`apex.evaluation.commit0_benchmark.Commit0BenchmarkRunner.evaluate_repo`,
optionally rewritten by the
``BenchmarkConfig.commit0_use_pytest_json_exitcode`` flag.

Inputs to :meth:`Commit0PrivateScorer.score_task`:

* ``task``: a :class:`apex.evaluation.commit0_benchmark.Commit0Task`.
* ``apex_artifacts``: a dict shape produced by
  :meth:`Commit0BenchmarkRunner._record_fairness_audit_delta`. The dict
  carries the already-computed APEX-private and upstream-canonical
  :class:`apex.evaluation.commit0_benchmark.Commit0Evaluation` instances
  so the scorers don't have to re-run pytest. Required keys:

    * ``"apex_private"``: a ``Commit0Evaluation`` instance.
    * ``"upstream"`` (optional): the upstream-canonical evaluation. The
      private scorer doesn't read this; it's there for the upstream
      scorer to find.

Outputs (flat metric mapping, all numeric so they participate in the
fairness delta):

* ``pass_rate`` (float in [0, 1]): the published per-task pass rate.
* ``returncode`` (int): the headline returncode after any private
  rewrite.
* ``passed`` (int): pytest-counted passing test count.
* ``failed`` (int): pytest-counted failing test count.
* ``errors`` (int): pytest-counted error count.
* ``score_source`` (str, non-numeric): provenance string. Recorded but
  does not participate in the numeric delta.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("apex.scorers.commit0_private")


@dataclass
class Commit0PrivateScorer:
    """Score a Commit0 task with APEX-private rules.

    The scorer is intentionally a pure function of the
    pre-computed :class:`Commit0Evaluation` objects on
    ``apex_artifacts["apex_private"]``. It does not re-run pytest or
    docker. This keeps the fairness audit O(N) in tasks rather than
    O(2N).
    """

    name: str = "commit0_private"

    def score_task(self, task: Any, apex_artifacts: Any) -> dict[str, Any]:
        evaluation = _coerce_evaluation(apex_artifacts, side="apex_private")
        if evaluation is None:
            return _no_evaluation_payload("commit0_private_eval_missing")
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
            "score_source": str(getattr(evaluation, "score_source", "shell_rc") or "shell_rc"),
        }


def _coerce_evaluation(apex_artifacts: Any, *, side: str) -> Any:
    """Pull ``apex_artifacts[side]`` as a Commit0Evaluation-like object.

    We accept either a plain dict (from ``Commit0Evaluation.to_dict()``)
    or a live ``Commit0Evaluation`` instance. The dict path keeps the
    scorer test-friendly; the live-object path is what the runner
    actually passes in.
    """
    if apex_artifacts is None:
        return None
    if isinstance(apex_artifacts, dict):
        target = apex_artifacts.get(side)
        if target is None:
            return None
        if isinstance(target, dict):
            return _DictEvaluationView(target)
        return target
    return None


def _no_evaluation_payload(reason: str) -> dict[str, Any]:
    return {
        "pass_rate": 0.0,
        "returncode": 1,
        "raw_returncode": 1,
        "scored_returncode": 1,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "score_source": reason,
    }


def _observed_returncode(evaluation: Any) -> int:
    value = getattr(evaluation, "returncode", None)
    if value is None:
        return 1
    return int(value)


class _DictEvaluationView:
    """Tiny attribute-shim around a dict so we don't import the runner."""

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name == "_data":
            raise AttributeError(name)
        return self._data.get(name)


def make_commit0_private_scorer() -> Commit0PrivateScorer:
    """Factory mirroring the other benchmark scorer factories."""
    return Commit0PrivateScorer()


__all__ = [
    "Commit0PrivateScorer",
    "make_commit0_private_scorer",
]
