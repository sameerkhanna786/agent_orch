"""TestGenEval APEX-private scorer.

Implements :class:`apex.core.fairness_audit.BenchmarkScorerProtocol` for
the APEX-private TestGenEval scoring path. The APEX-private path is the
parallel log-marker re-aggregator in
:mod:`apex.evaluation.runners.testgenevallite_aggregate`. This bypasses
the upstream ``generate_report.py`` (which crashes on
``KeyError("baseline_covs")`` for certain
``kjain14/testgenevallite`` rows) and re-derives pass@1 by regex.

Inputs to :meth:`TestGenEvalPrivateScorer.score_task`:

* ``task``: anything with an ``instance_id`` / ``task_id`` attribute.
* ``apex_artifacts``: ignored by this scorer; the per-task pass/fail
  signal is read directly from ``self.log_dir``. ``apex_artifacts`` is
  accepted for protocol conformance and so the runner can pass a
  shared ``output_dir`` for diagnostics.

Outputs (flat metric mapping; numeric fields participate in the
fairness delta, non-numeric fields are recorded for provenance only):

* ``pass_at_1`` (float in {0.0, 1.0}): 1.0 when the task's eval log
  shows ``filtered_pass`` or ``unfiltered_pass``, 0.0 otherwise.
* ``pass_at_1_unfiltered`` (float): 1.0 only on ``unfiltered_pass``.
* ``mutation_score`` (float): the percent-form mutation score parsed
  from ``MutationLOG:`` lines, or ``-1.0`` if the log lacked a
  measurement.
* ``coverage`` (float): the percent-form coverage parsed from
  ``CoverageLOG:`` lines, or ``-1.0`` if the log lacked a
  measurement.
* ``status`` (str, non-numeric): one of ``unfiltered_pass`` /
  ``filtered_pass`` / ``failed`` / ``no_signal`` / ``no_log``.
* ``score_source`` (str, non-numeric): always
  ``"apex_private_log_marker"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apex.evaluation.runners.testgenevallite_aggregate import (
    parse_official_eval_log,
)


@dataclass
class TestGenEvalPrivateScorer:
    """Score a TestGenEval task with the APEX-private log-marker rules.

    The scorer is a pure function of the per-task ``.eval.log`` file the
    upstream harness already wrote. It does not re-run the harness or
    re-execute any tests, so the audit cost is O(N) rather than O(2N).
    """

    # Tell pytest this is not a test class despite the ``Test`` prefix.
    __test__ = False

    log_dir: Path
    model_name: str
    dataset_name: str = "kjain14/testgenevallite"
    split: str = "test"
    name: str = "testgeneval_private"

    def score_task(self, task: Any, apex_artifacts: Any) -> dict[str, Any]:
        instance_id = _resolve_instance_id(task)
        log_path = self._log_path_for(instance_id)
        if log_path is None:
            return _no_log_payload("private_log_missing")
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            return _no_log_payload("private_log_unreadable")
        parsed = parse_official_eval_log(text)
        passed = parsed.status in {"filtered_pass", "unfiltered_pass"}
        unfiltered = parsed.status == "unfiltered_pass"
        return {
            "pass_at_1": 1.0 if passed else 0.0,
            "pass_at_1_unfiltered": 1.0 if unfiltered else 0.0,
            "mutation_score": (float(parsed.mutation) if parsed.mutation is not None else -1.0),
            "coverage": (float(parsed.coverage) if parsed.coverage is not None else -1.0),
            "status": parsed.status,
            "score_source": "apex_private_log_marker",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_path_for(self, instance_id: str) -> Path | None:
        suffix = f".{self.model_name}.full.eval.log"
        candidate = self.log_dir / f"{instance_id}{suffix}"
        if candidate.exists():
            return candidate
        # Fall back to a glob — the harness sometimes appends a setting
        # token between the instance_id and the model_name that the
        # caller may not know.
        for entry in sorted(self.log_dir.iterdir()):
            if entry.name.startswith(f"{instance_id}.") and entry.name.endswith(suffix):
                return entry
        return None


def _resolve_instance_id(task: Any) -> str:
    if hasattr(task, "instance_id"):
        return str(getattr(task, "instance_id"))
    if hasattr(task, "task_id"):
        return str(getattr(task, "task_id"))
    if isinstance(task, dict):
        return str(task.get("instance_id") or task.get("task_id") or task.get("id") or "")
    return str(task)


def _no_log_payload(reason: str) -> dict[str, Any]:
    return {
        "pass_at_1": 0.0,
        "pass_at_1_unfiltered": 0.0,
        "mutation_score": -1.0,
        "coverage": -1.0,
        "status": "no_log",
        "score_source": reason,
    }


def make_testgeneval_private_scorer(
    *,
    log_dir: Path,
    model_name: str,
    dataset_name: str = "kjain14/testgenevallite",
    split: str = "test",
) -> TestGenEvalPrivateScorer:
    """Factory mirroring the other benchmark scorer factories."""
    return TestGenEvalPrivateScorer(
        log_dir=Path(log_dir),
        model_name=model_name,
        dataset_name=dataset_name,
        split=split,
    )


__all__ = [
    "TestGenEvalPrivateScorer",
    "make_testgeneval_private_scorer",
]
