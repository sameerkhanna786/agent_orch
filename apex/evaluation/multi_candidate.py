"""Candidate ranking utilities for execution-guided test generation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

# ---------------------------------------------------------------------------
# Phase 4.2: calibrated weighted-composite ranking weights.
#
# The legacy lexicographic 10-tuple in :func:`TestgenCandidateEvaluation.ranking_key`
# put `unfiltered_pass_at_1` and the binary `meaningful_test_count > 0` ahead
# of `mutation_score` (mutation was the 7th tiebreaker). On real benchmark
# runs this caused candidates with shallow but passing tests to win over
# candidates with strong mutation discrimination, dragging the cohort
# mutation kill rate to ~16.7% on Phase G smoke.
#
# These DEFAULT weights are a literature-informed prior (PIT/Major mutation
# scores correlate with bug-detection capability; oracle grounding & assertion
# effect size correlate with non-trivial assertions; pass_at_1 stays the
# strongest signal because it dominates the overall benchmark metric). See
# ``apex/scripts/calibrate_testgen_ranking.py`` for the future-data refit
# pipeline.
# ---------------------------------------------------------------------------

DEFAULT_TESTGEN_RANKING_WEIGHTS: dict[str, float] = {
    "pass_at_1": 0.30,
    "mutation_score": 0.25,
    "coverage_delta": 0.15,
    "oracle_grounding": 0.10,
    "assertion_effect": 0.10,
    "dual_state_score": 0.05,
    "meaningful_test_count_log": 0.05,
}


def _resolve_ranking_weights(
    weights: Optional[Mapping[str, float]] = None,
) -> dict[str, float]:
    """Coerce a possibly-None / partial overrides dict into the full weight set.

    Missing keys fall back to ``DEFAULT_TESTGEN_RANKING_WEIGHTS`` rather than
    silently scoring 0 — partial overrides are common in benchmark-specific
    tuning and we want callers to be able to bump just the mutation weight
    without restating every other component.
    """
    resolved = dict(DEFAULT_TESTGEN_RANKING_WEIGHTS)
    if weights:
        for key, value in weights.items():
            try:
                resolved[key] = float(value)
            except (TypeError, ValueError):
                continue
    return resolved


@dataclass(frozen=True)
class TestgenCandidateEvaluation:
    # Tell pytest this is not a test class despite the ``Testgen``
    # prefix; pytest's default collection rule matches ``Test*``.
    __test__ = False

    candidate_id: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    unfiltered_pass_at_1: float = 0.0
    coverage_delta: float = 0.0
    mutation_score: float = 0.0
    num_methods: int = 0
    oracle_grounding_score: float = 0.0
    assertion_effect_score: float = 0.0
    quality_score: float = 0.0
    dual_state_score: float = 0.0
    meaningful_test_count: int = 0
    static_validity_status: str = "unknown"
    import_validity_status: str = "unknown"
    artifact_failure_taxonomy: str = "unknown"
    prediction_quality: str = "unknown"
    validation_uncertainty: str = "unknown"
    # Phase 4.8 (early): per-candidate failure_class so the selector
    # can distinguish ENV-class failures (GENERATOR_CRASH /
    # GENERATOR_TIMEOUT / EMPTY_OUTPUT) from APEX-class failures.
    # The selector treats ``GENERATOR_*`` and ``EMPTY_OUTPUT`` as
    # environment failures so they don't poison candidate selection.
    failure_class: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def validation_floor_rank(self) -> float:
        return _candidate_validation_floor(
            {
                "static_validity_status": self.static_validity_status,
                "import_validity_status": self.import_validity_status,
                "artifact_failure_taxonomy": self.artifact_failure_taxonomy,
                "prediction_quality": self.prediction_quality,
                "validation_uncertainty": self.validation_uncertainty,
                **dict(self.diagnostics or {}),
            }
        )

    def ranking_key(
        self,
    ) -> tuple[float, float, float, float, float, float, float, float, int, int]:
        """Legacy lexicographic ranking key.

        Phase 4.2 replaced the primary selection path with a calibrated
        weighted composite (:func:`composite_score`). This tuple is kept
        as the FALLBACK ordering for ``select_best_testgen_candidate``
        when the composite is undefined for every candidate (e.g. all
        candidates are env-flagged with ``-inf`` composite). It also
        remains useful for diagnostics and stable secondary sorting.
        """
        return (
            self.validation_floor_rank(),
            float(self.unfiltered_pass_at_1 or 0.0),
            1.0 if int(self.meaningful_test_count or 0) > 0 else 0.0,
            float(self.oracle_grounding_score or 0.0),
            float(self.coverage_delta or 0.0),
            float(self.assertion_effect_score or 0.0),
            float(self.mutation_score or 0.0),
            float(self.dual_state_score or 0.0),
            float(self.quality_score or 0.0),
            int(self.meaningful_test_count or 0),
            -int(self.num_methods or 0),
        )

    def composite_score(
        self,
        weights: Optional[Mapping[str, float]] = None,
    ) -> float:
        """Phase 4.2 weighted-composite ranking.

        Returns ``-inf`` for hard-veto candidates: those whose validation
        floor failed (the legacy "validation_floor_rank == 'fail'" gate)
        and those flagged as environment failures. Otherwise returns a
        weighted linear combination of the candidate's signal components.

        The default weight prior is in
        :data:`DEFAULT_TESTGEN_RANKING_WEIGHTS`. Callers may pass a
        partial override (missing keys fall back to the default).
        """
        # Hard veto 1: environment failure (generator crashed / timed out /
        # produced empty output). We do not want a transient generator
        # failure to win a comparison purely because its raw signal
        # components happen to be zero.
        if self.is_environment_failure():
            return float("-inf")
        # Hard veto 2: validation floor failed. ``validation_floor_rank()``
        # returns 0.0 when the candidate failed static / import / taxonomy
        # / prediction-quality checks (legacy "fail" tier).
        if self.validation_floor_rank() <= 0.0:
            return float("-inf")
        w = _resolve_ranking_weights(weights)
        return (
            w["pass_at_1"] * float(self.unfiltered_pass_at_1 or 0.0)
            + w["mutation_score"] * float(self.mutation_score or 0.0)
            + w["coverage_delta"] * float(self.coverage_delta or 0.0)
            + w["oracle_grounding"] * float(self.oracle_grounding_score or 0.0)
            + w["assertion_effect"] * float(self.assertion_effect_score or 0.0)
            + w["dual_state_score"] * float(self.dual_state_score or 0.0)
            + w["meaningful_test_count_log"]
            * math.log1p(max(0, int(self.meaningful_test_count or 0)))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "artifact_count": len(self.artifacts),
            "unfiltered_pass_at_1": self.unfiltered_pass_at_1,
            "coverage_delta": self.coverage_delta,
            "mutation_score": self.mutation_score,
            "num_methods": self.num_methods,
            "oracle_grounding_score": self.oracle_grounding_score,
            "assertion_effect_score": self.assertion_effect_score,
            "quality_score": self.quality_score,
            "dual_state_score": self.dual_state_score,
            "meaningful_test_count": self.meaningful_test_count,
            "validation_floor_rank": self.validation_floor_rank(),
            "static_validity_status": self.static_validity_status,
            "import_validity_status": self.import_validity_status,
            "artifact_failure_taxonomy": self.artifact_failure_taxonomy,
            "prediction_quality": self.prediction_quality,
            "validation_uncertainty": self.validation_uncertainty,
            "failure_class": self.failure_class,
            "diagnostics": dict(self.diagnostics),
        }

    def is_environment_failure(self) -> bool:
        """True iff the candidate's ``failure_class`` is one of the
        ENV-class testgen modes (GENERATOR_CRASH / GENERATOR_TIMEOUT /
        EMPTY_OUTPUT) or any ``env_*`` / ``harness_bug`` value from
        the core failure taxonomy. The selector treats these as
        non-charging — they should be retried on a clean container,
        not blamed on the candidate.
        """
        text = str(self.failure_class or "").strip()
        if not text:
            return False
        upper = text.upper()
        if upper in {"GENERATOR_CRASH", "GENERATOR_TIMEOUT", "EMPTY_OUTPUT"}:
            return True
        lowered = text.lower()
        return lowered.startswith("env_") or lowered == "harness_bug"

    # ------------------------------------------------------------------
    # Phase 4.8: from_generator_outcome — populate ``failure_class`` from
    # a generator process's outcome (stderr / returncode / exception
    # type) using the core failure classifier. Callers that build
    # candidates from a default-generator subprocess invocation should
    # use this helper so a CLI crash or empty-output is recorded as an
    # ENV failure (and therefore deprioritized by the selector) rather
    # than as ``unfiltered_pass_at_1=0.0`` (which would be indistinguishable
    # from a real "tests don't pass" candidate).
    # ------------------------------------------------------------------
    @classmethod
    def from_generator_outcome(
        cls,
        *,
        candidate_id: str,
        outcome: Mapping[str, Any],
        artifacts: Optional[list[dict[str, Any]]] = None,
        diagnostics: Optional[dict[str, Any]] = None,
    ) -> "TestgenCandidateEvaluation":
        """Build a candidate evaluation from a generator process's outcome.

        ``outcome`` is a dict-like with one or more of:
          * ``stderr`` (str): captured stderr from the generator process.
          * ``stdout`` (str): captured stdout (used as a secondary signal).
          * ``returncode`` (int): process exit code (default 1).
          * ``exception_type`` (str): name of the Python exception class
            raised by the LLM client, if any (e.g. ``"TimeoutError"``).
          * ``output_tokens`` / ``output`` / ``artifact_count``: hints
            for empty-output detection. If ``output`` (str) is empty/blank
            *and* there are no ``artifacts``, we classify as
            ``EMPTY_OUTPUT``.
          * ``phase`` (str): optional context phase forwarded to the core
            classifier (default ``"test_execution"`` since generator runs
            after the install/baseline phases).

        Mapping rules (Phase 4.8):
          * ``exception_type`` matches ``Timeout`` (case-insensitive) →
            ``GENERATOR_TIMEOUT``.
          * Empty/blank stdout AND no artifacts AND returncode == 0 →
            ``EMPTY_OUTPUT``.
          * Any other non-zero returncode → defer to
            :func:`apex.core.failure_classifier.classify_failure`. If
            that returns an env-class verdict (env_*, harness_bug) we
            label as ``GENERATOR_CRASH``. Otherwise we propagate
            ``APEX_MISS`` so the candidate is still scored as a real
            (zero-pass) candidate.
        """

        from apex.core.failure_classifier import FailureClass, classify_failure

        artifacts_list: list[dict[str, Any]] = list(artifacts or [])
        diag: dict[str, Any] = dict(diagnostics or {})

        stderr = str(outcome.get("stderr") or "")
        stdout = str(outcome.get("stdout") or outcome.get("output") or "")
        returncode = int(outcome.get("returncode") or 0)
        exception_type = str(outcome.get("exception_type") or "").strip()
        phase = str(outcome.get("phase") or "test_execution") or "test_execution"

        failure_class = ""

        # 1) Explicit timeout signal — covers both subprocess.TimeoutExpired
        #    and the LLM client's own TimeoutError.
        if exception_type and "timeout" in exception_type.lower():
            failure_class = "GENERATOR_TIMEOUT"
        else:
            # 2) Empty / garbage output — the CLI exited cleanly but
            #    produced no usable artifacts. Defer to the empty-output
            #    bucket so the selector doesn't mistake it for a real
            #    zero-pass candidate.
            empty_output = (not stdout.strip()) and (not artifacts_list)
            if empty_output and returncode == 0 and not exception_type:
                failure_class = "EMPTY_OUTPUT"
            elif returncode != 0 or stderr.strip() or exception_type:
                # 3) Hand off to the core classifier and translate.
                verdict = classify_failure(
                    stderr=stderr,
                    stdout=stdout,
                    returncode=returncode if returncode else 1,
                    context={"phase": phase},
                )
                if verdict.failure_class.is_environment:
                    failure_class = "GENERATOR_CRASH"
                elif verdict.failure_class is FailureClass.HARNESS_BUG:
                    failure_class = "GENERATOR_CRASH"
                else:
                    # APEX_MISS / UNCLASSIFIED — keep the candidate in
                    # the scored pool with whatever signal it produced.
                    failure_class = ""

        diag.setdefault("generator_outcome", dict(outcome))
        return cls(
            candidate_id=candidate_id,
            artifacts=artifacts_list,
            failure_class=failure_class,
            diagnostics=diag,
        )


def select_best_testgen_candidate(
    candidates: list[TestgenCandidateEvaluation],
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> TestgenCandidateEvaluation | None:
    """Pick the best candidate via the Phase 4.2 weighted composite.

    Selection algorithm:
      1. Drop env-flagged candidates (``is_environment_failure``) from
         the active pool. If every candidate is env-flagged we fall back
         to the full set so we still surface *something* — better to
         report a broken winner than to swallow them all silently.
      2. Compute :meth:`TestgenCandidateEvaluation.composite_score` on
         each pool member. The composite returns ``-inf`` for hard-veto
         candidates (validation floor "fail" or env failure).
      3. If at least one candidate has a finite composite, return the
         argmax. Ties are broken by the legacy lexicographic
         :meth:`ranking_key` (preserves the secondary "smaller artifact
         wins" intent from earlier phases).
      4. If every composite is ``-inf`` the cohort is structurally
         degenerate; fall back to the legacy lexicographic key so the
         caller still gets a deterministic pick.

    Phase 4.8: candidates flagged as ``is_environment_failure`` (i.e.
    GENERATOR_CRASH / GENERATOR_TIMEOUT / EMPTY_OUTPUT or any env_*
    failure) are deprioritized — they're returned only when *every*
    candidate is an env failure.
    """

    if not candidates:
        return None
    real_candidates = [c for c in candidates if not c.is_environment_failure()]
    pool = real_candidates if real_candidates else candidates

    scored: list[tuple[float, tuple[float, ...], TestgenCandidateEvaluation]] = []
    for c in pool:
        score = c.composite_score(weights=weights)
        scored.append((score, c.ranking_key(), c))

    finite_scored = [item for item in scored if math.isfinite(item[0])]
    if finite_scored:
        finite_scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return finite_scored[0][2]

    # Fallback: every candidate composite is -inf (all hard-vetoed).
    # Use the legacy lexicographic key so callers still get a stable
    # pick — and so existing diagnostics that depend on a non-None
    # selection continue to work.
    return max(pool, key=lambda candidate: candidate.ranking_key())


def summarize_candidate_selection(
    candidates: list[TestgenCandidateEvaluation],
    selected: TestgenCandidateEvaluation | None,
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> dict[str, Any]:
    resolved = _resolve_ranking_weights(weights)
    return {
        "candidate_count": len(candidates),
        "selected_candidate": selected.candidate_id if selected else None,
        "ranking_weights": resolved,
        "candidates": [
            {
                **candidate.to_dict(),
                "composite_score": candidate.composite_score(weights=resolved),
            }
            for candidate in candidates
        ],
    }


_PASS_STATUSES = {
    "pass",
    "passed",
    "ok",
    "clean",
    "valid",
    "deferred",
    "deferred_to_adapter_environment",
}
_UNKNOWN_STATUSES = {
    "",
    "unknown",
    "unsupported",
    "unavailable",
    "skipped",
    "uncertain",
    "adapter_deferred",
}
_FAIL_STATUSES = {
    "fail",
    "failed",
    "error",
    "errored",
    "invalid",
    "artifact_failed",
    "collection_failed",
    "setup_failed",
    "missing_symbol",
    "missing_import",
    "namespace_collision",
}
_BAD_PREDICTION_QUALITY = {"failed", "fallback_last_valid", "forged_pass", "known_bad"}


def _status_rank(value: Any) -> float:
    text = str(value or "").strip().lower()
    if text in _FAIL_STATUSES:
        return 0.0
    if text in _PASS_STATUSES:
        return 2.0
    if text in _UNKNOWN_STATUSES:
        return 1.0
    if "fail" in text or "error" in text or "missing" in text or "collision" in text:
        return 0.0
    return 1.0


def _candidate_validation_floor(evidence: dict[str, Any]) -> float:
    nested = dict(evidence.get("apex_validation") or {})
    tier_1 = nested.get("tier_1_static") or evidence.get("tier_1_static")
    if isinstance(tier_1, dict):
        evidence.setdefault("static_validity_status", tier_1.get("status"))
    tier_2 = nested.get("tier_2_import") or evidence.get("tier_2_import")
    if isinstance(tier_2, dict):
        evidence.setdefault("import_validity_status", tier_2.get("status"))

    ranks = [
        _status_rank(evidence.get("static_validity_status")),
        _status_rank(evidence.get("import_validity_status")),
    ]
    taxonomy = str(evidence.get("artifact_failure_taxonomy") or "").strip().lower()
    if taxonomy and taxonomy not in _UNKNOWN_STATUSES:
        ranks.append(_status_rank(taxonomy))
    prediction_quality = (
        str(evidence.get("prediction_quality") or nested.get("prediction_quality") or "")
        .strip()
        .lower()
    )
    if prediction_quality in _BAD_PREDICTION_QUALITY:
        ranks.append(0.0)
    elif prediction_quality in {"clean", "minimized", "validated"}:
        ranks.append(2.0)
    else:
        ranks.append(1.0)
    uncertainty = str(evidence.get("validation_uncertainty") or "").strip().lower()
    if uncertainty in {"timeout", "flaky", "uncertain"}:
        ranks.append(1.0)
    return min(ranks)
