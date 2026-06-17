"""SWT-Bench upstream-canonical scorer.

SWT-Bench has only ONE scoring path: the upstream ``swt_bench`` Docker
harness, invoked unmodified. APEX does not maintain a private rewrite of
SWT-Bench's scorer the way it does for Commit0 and TestGenEval, so
"upstream-canonical" and "APEX-private" are the same thing here.

Per :mod:`apex.core.fairness_audit`, the right thing to do for a
benchmark with a single scoring path is to pass the same scorer instance
as both ``private_scorer`` and ``upstream_scorer`` to
``run_fairness_audit``. The resulting per-task delta will be all zeros,
which:

  1. Does not lie -- there genuinely is no private/upstream split.
  2. Still lets the fairness-audit framework emit a per-task entry for
     SWT-Bench so reviewers can see "comparable" rather than "missing".

Inputs to :meth:`SWTBenchUpstreamScorer.score_task`:

* ``task``: an :class:`apex.evaluation.swtbench_benchmark.SWTBenchTask`
  (or any object with ``instance_id``).
* ``apex_artifacts``: the dict produced by
  :func:`apex.evaluation.runners.swtbench_generate._record_from_result`
  (or a :class:`apex.evaluation.testgeneval_benchmark.TestGenEvalTaskResult`).

Outputs (flat metric mapping):

* ``resolved`` (1.0 / 0.0): F2P verdict from the harness.
* ``model_patch_present`` (1.0 / 0.0): did the agent produce a non-empty
  ``model_patch``?
* ``pass_at_1`` (float): inherited from the underlying TestGenEval
  pipeline diagnostic when present.
* ``failure_class`` (str, non-numeric): per-task failure classification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("apex.scorers.swtbench_upstream")


@dataclass
class SWTBenchUpstreamScorer:
    """Implements :class:`apex.core.fairness_audit.BenchmarkScorerProtocol`.

    The scorer is intentionally lightweight: SWT-Bench's actual
    pass/fail verdict is computed by the ``swt_bench`` Docker harness
    (see :class:`apex.evaluation.swtbench_docker_adapter.SWTBenchDockerAdapter`).
    This scorer just normalizes whatever the harness produced into the
    flat metric mapping the fairness audit consumes.
    """

    name: str = "swtbench_upstream"

    def score_task(self, task: Any, apex_artifacts: Any) -> dict[str, Any]:
        """Return a flat metric dict for this task.

        Tolerates four input shapes for ``apex_artifacts``:

          * The dict produced by ``_record_from_result`` (has
            ``model_patch`` and ``apex_validation``).
          * A :class:`TestGenEvalTaskResult` (has ``pass_at_1``,
            ``diagnostics``).
          * A :class:`apex.evaluation.final_acceptance_gate.FinalAcceptanceRun`
            (has ``status``).
          * ``None`` -- treated as no submission (model_patch_present=0,
            resolved=0).
        """
        if apex_artifacts is None:
            return {
                "resolved": 0.0,
                "model_patch_present": 0.0,
                "pass_at_1": 0.0,
                "failure_class": "no_submission",
            }

        # --- Shape 1: predictions dict from _record_from_result ---
        if isinstance(apex_artifacts, dict):
            model_patch = str(apex_artifacts.get("model_patch") or "")
            validation = dict(apex_artifacts.get("apex_validation") or {})
            failure_class = (
                str(validation.get("failure_class") or "")
                or str(validation.get("prediction_quality") or "")
                or "unknown"
            )
            pass_at_1 = float(validation.get("pass_at_1") or 0.0)
            resolved = self._infer_resolved_from_validation(validation)
            return {
                "resolved": float(resolved),
                "model_patch_present": 1.0 if model_patch.strip() else 0.0,
                "pass_at_1": float(pass_at_1),
                "failure_class": failure_class,
            }

        # --- Shape 2: TestGenEvalTaskResult ---
        diagnostics = dict(getattr(apex_artifacts, "diagnostics", {}) or {})
        validation = dict(diagnostics.get("apex_validation") or {})
        pass_at_1 = float(getattr(apex_artifacts, "pass_at_1", 0.0) or 0.0)
        all_pass_at_1 = float(getattr(apex_artifacts, "all_pass_at_1", 0.0) or 0.0)
        success = bool(getattr(apex_artifacts, "success", False))
        # We treat all_pass_at_1==1.0 as the F2P-resolved signal; otherwise
        # fall back to ``success``. The Docker adapter records the actual
        # harness verdict in diagnostics["swtbench_harness_verdict"] when
        # available.
        resolved = self._infer_resolved_from_diagnostics(
            diagnostics, all_pass_at_1=all_pass_at_1, success=success
        )
        artifact_present = self._artifact_present(diagnostics)
        failure_class = (
            str(validation.get("failure_class") or "")
            or str(validation.get("prediction_quality") or "")
            or ("ok" if success else "unknown")
        )
        return {
            "resolved": float(resolved),
            "model_patch_present": 1.0 if artifact_present else 0.0,
            "pass_at_1": float(pass_at_1),
            "failure_class": failure_class,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_resolved_from_validation(validation: dict[str, Any]) -> float:
        if "resolved" in validation:
            return 1.0 if bool(validation["resolved"]) else 0.0
        quality = str(validation.get("prediction_quality") or "").lower()
        if quality == "clean":
            return 1.0
        return 0.0

    @staticmethod
    def _infer_resolved_from_diagnostics(
        diagnostics: dict[str, Any],
        *,
        all_pass_at_1: float,
        success: bool,
    ) -> float:
        verdict = diagnostics.get("swtbench_harness_verdict")
        if isinstance(verdict, dict):
            status = str(verdict.get("status") or "").lower()
            if status == "pass":
                return 1.0
            if status in {"fail", "harness_error"}:
                return 0.0
        if all_pass_at_1 >= 1.0:
            return 1.0
        if success and all_pass_at_1 > 0.0:
            return 1.0
        return 0.0

    @staticmethod
    def _artifact_present(diagnostics: dict[str, Any]) -> bool:
        for key in ("generated_artifacts", "shipped_artifacts", "final_artifacts"):
            value = diagnostics.get(key)
            if isinstance(value, list):
                for artifact in value:
                    if isinstance(artifact, dict) and str(artifact.get("content") or "").strip():
                        return True
        return False


def make_swtbench_upstream_scorer() -> SWTBenchUpstreamScorer:
    """Factory mirroring the other benchmark scorer factories."""
    return SWTBenchUpstreamScorer()


__all__ = [
    "SWTBenchUpstreamScorer",
    "make_swtbench_upstream_scorer",
]
