"""Clarification and abstention assessment for underspecified tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# WS3I: actions that imply an APEX-side abstain (we cannot confidently accept yet).
# ``escalate_harness_configuration`` is deliberately EXCLUDED — it is an
# env/harness escalation, not an APEX abstain.
_ABSTAIN_ACTIONS = frozenset(
    {
        "abstain_until_scorers_reconcile",
        "collect_expected_inventory_before_acceptance",
        "collect_more_evidence",
    }
)


@dataclass(frozen=True)
class ClarificationAssessment:
    needed: bool
    kind: str = ""
    reasons: list[str] = field(default_factory=list)
    action: str = "continue"

    @property
    def should_abstain(self) -> bool:
        """WS3I: True when this assessment implies an APEX-side abstain. Consumed
        only when the clarification-abstain arm is enabled (default off)."""
        return bool(self.needed) and self.action in _ABSTAIN_ACTIONS

    def to_dict(self) -> dict[str, Any]:
        return {
            "needed": bool(self.needed),
            "kind": self.kind,
            "reasons": list(self.reasons),
            "action": self.action,
        }


def assess_clarification_need(
    *,
    issue_plan: Any = None,
    verification_taxonomy: dict[str, Any] | None = None,
    evidence_ledger: dict[str, Any] | None = None,
) -> ClarificationAssessment:
    taxonomy_kind = str((verification_taxonomy or {}).get("kind") or "")
    reasons: list[str] = []
    if taxonomy_kind == "scorer_disagreement":
        reasons.append("private and canonical scorers disagree")
        return ClarificationAssessment(True, "scorer_ambiguity", reasons, "abstain_until_scorers_reconcile")
    if taxonomy_kind == "harness_config_failure":
        reasons.append("verification failed in harness/config setup")
        return ClarificationAssessment(True, "harness_ambiguity", reasons, "escalate_harness_configuration")
    test_context = getattr(issue_plan, "test_context", None)
    expected_count = int(getattr(test_context, "expected_test_count", 0) or 0)
    expected_ids = list(getattr(test_context, "expected_test_ids", []) or [])
    evidence_score = float((evidence_ledger or {}).get("score") or 0.0)
    if expected_count <= 0 and not expected_ids and evidence_score < 0.35:
        reasons.append("missing expected-test inventory and weak verification evidence")
        return ClarificationAssessment(True, "missing_expected_id_ambiguity", reasons, "collect_expected_inventory_before_acceptance")
    if taxonomy_kind in {"timeout_inconclusive", "unclassified"} and evidence_score < 0.4:
        reasons.append("verification outcome is inconclusive and evidence is weak")
        return ClarificationAssessment(True, "verification_ambiguity", reasons, "collect_more_evidence")
    return ClarificationAssessment(False)
