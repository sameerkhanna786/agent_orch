"""Typed task blackboard for mode-aware rollout coordination."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

from ..controller_policy import (
    EVIDENCE_MODE_EVAL_ONLY_SUITE,
    EVIDENCE_MODE_GOLD_SUITE_VISIBLE,
    EVIDENCE_MODE_NO_SUITE_VISIBLE,
    EVIDENCE_MODE_PARTIAL_SUITE_VISIBLE,
    EvidencePolicy,
)

if TYPE_CHECKING:
    from ..planning.manager import IssuePlan


PROVENANCE_AGENT_VISIBLE = "agent_visible"
PROVENANCE_AGENT_GENERATED = "agent_generated"
PROVENANCE_VERIFIED = "verified"
PROVENANCE_SPECULATIVE = "speculative"
PROVENANCE_EVALUATION_ONLY = "evaluation_only"

RECORD_TEST_INVENTORY = "test_inventory"
RECORD_CONTRACT_OBLIGATION = "contract_obligation"
RECORD_FAILURE_FRONTIER = "failure_frontier"
RECORD_LOCALIZATION_HYPOTHESIS = "localization_hypothesis"
RECORD_PATCH_DELTA = "patch_delta"
RECORD_VERIFICATION = "verification"
RECORD_REFLECTION = "reflection"
RECORD_NEGATIVE_EVIDENCE = "negative_evidence"
RECORD_HIDDEN_RISK_REVIEW = "hidden_risk_review"

_AGENT_VISIBLE_PROVENANCE = frozenset(
    {
        PROVENANCE_AGENT_VISIBLE,
        PROVENANCE_AGENT_GENERATED,
        PROVENANCE_VERIFIED,
        PROVENANCE_SPECULATIVE,
    }
)
_PROMPT_VISIBLE_TEST_ID_LIMIT = 32


def _dedupe(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        result.append(text)
        seen.add(text)
    return result


def _stable_record_id(record_type: str, description: str, source: str) -> str:
    digest = hashlib.sha1(f"{record_type}\n{source}\n{description}".encode("utf-8")).hexdigest()[
        :12
    ]
    return f"bb_{digest}"


def _bounded_prompt_test_ids(values: Iterable[Any]) -> list[str]:
    return _dedupe(values)[:_PROMPT_VISIBLE_TEST_ID_LIMIT]


@dataclass
class BlackboardRecord:
    """One append-only, provenance-tagged fact or hypothesis."""

    record_type: str
    description: str
    provenance: str
    confidence: float = 0.5
    source: str = ""
    file_paths: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    record_id: str = ""

    def normalized(self) -> "BlackboardRecord":
        record_type = str(self.record_type or "").strip().lower()
        description = str(self.description or "").strip()
        source = str(self.source or "").strip()
        provenance = str(self.provenance or PROVENANCE_SPECULATIVE).strip().lower()
        if provenance not in {
            PROVENANCE_AGENT_VISIBLE,
            PROVENANCE_AGENT_GENERATED,
            PROVENANCE_VERIFIED,
            PROVENANCE_SPECULATIVE,
            PROVENANCE_EVALUATION_ONLY,
        }:
            provenance = PROVENANCE_SPECULATIVE
        record_id = str(self.record_id or "").strip() or _stable_record_id(
            record_type,
            description,
            source,
        )
        return BlackboardRecord(
            record_type=record_type,
            description=description,
            provenance=provenance,
            confidence=max(0.0, min(float(self.confidence or 0.0), 1.0)),
            source=source,
            file_paths=_dedupe(self.file_paths),
            symbols=_dedupe(self.symbols),
            test_ids=_dedupe(self.test_ids),
            payload=dict(self.payload or {}),
            record_id=record_id,
        )

    @property
    def agent_visible(self) -> bool:
        return self.normalized().provenance in _AGENT_VISIBLE_PROVENANCE

    def to_dict(self) -> dict[str, Any]:
        record = self.normalized()
        return {
            "record_id": record.record_id,
            "record_type": record.record_type,
            "description": record.description,
            "provenance": record.provenance,
            "confidence": round(record.confidence, 4),
            "source": record.source,
            "file_paths": list(record.file_paths),
            "symbols": list(record.symbols),
            "test_ids": list(record.test_ids),
            "payload": dict(record.payload),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BlackboardRecord":
        return cls(
            record_id=str(data.get("record_id") or ""),
            record_type=str(data.get("record_type") or ""),
            description=str(data.get("description") or ""),
            provenance=str(data.get("provenance") or ""),
            confidence=float(data.get("confidence") or 0.0),
            source=str(data.get("source") or ""),
            file_paths=[str(item) for item in list(data.get("file_paths") or [])],
            symbols=[str(item) for item in list(data.get("symbols") or [])],
            test_ids=[str(item) for item in list(data.get("test_ids") or [])],
            payload=dict(data.get("payload") or {}),
        ).normalized()


BlackboardRecord.__test__ = False


@dataclass
class TaskBlackboard:
    """Append-only typed coordination state for a single task."""

    evidence_policy: EvidencePolicy = field(default_factory=EvidencePolicy)
    records: list[BlackboardRecord] = field(default_factory=list)

    def append(self, record: BlackboardRecord) -> None:
        normalized = record.normalized()
        if normalized.description:
            self.records.append(normalized)

    def agent_visible_records(self) -> list[BlackboardRecord]:
        return [record.normalized() for record in self.records if record.agent_visible]

    def to_dict(self, *, agent_visible_only: bool = False) -> dict[str, Any]:
        records = self.agent_visible_records() if agent_visible_only else list(self.records)
        return {
            "evidence_policy": self.evidence_policy.normalized().to_dict(),
            "records": [record.normalized().to_dict() for record in records],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskBlackboard":
        return cls(
            evidence_policy=EvidencePolicy.from_dict(data.get("evidence_policy") or {}),
            records=[
                BlackboardRecord.from_dict(item)
                for item in list(data.get("records") or [])
                if isinstance(item, dict)
            ],
        )


TaskBlackboard.__test__ = False


def _policy_from_issue_plan(issue_plan: "IssuePlan") -> EvidencePolicy:
    constraints = getattr(issue_plan, "evaluation_constraints", None)
    resolver = getattr(constraints, "resolved_evidence_policy", None)
    if callable(resolver):
        return resolver()
    test_context = getattr(issue_plan, "test_context", None)
    if test_context is not None:
        return EvidencePolicy.from_dict(getattr(test_context, "evidence_policy", {}) or {})
    return EvidencePolicy().normalized()


def build_initial_task_blackboard(issue_plan: "IssuePlan") -> TaskBlackboard:
    """Build the initial prompt-visible blackboard from issue planning state."""

    policy = _policy_from_issue_plan(issue_plan)
    blackboard = TaskBlackboard(evidence_policy=policy)
    test_context = issue_plan.test_context

    if policy.mode != EVIDENCE_MODE_EVAL_ONLY_SUITE:
        if test_context.command or test_context.test_inventory_framework:
            expected_test_ids = _dedupe(test_context.expected_test_ids)
            visible_expected_test_ids = _bounded_prompt_test_ids(expected_test_ids)
            visible_failing_test_ids = _bounded_prompt_test_ids(test_context.failing_test_ids)
            expected_test_count = (
                int(test_context.expected_test_count or 0)
                if policy.mode == EVIDENCE_MODE_GOLD_SUITE_VISIBLE
                else 0
            )
            expected_test_count = max(expected_test_count, len(expected_test_ids))
            inventory_payload = {
                "command": test_context.command or "",
                "framework": test_context.test_inventory_framework,
                "language": test_context.test_inventory_language,
                "expected_test_count": expected_test_count,
                "collection_command": test_context.test_collection_command,
            }
            if policy.mode == EVIDENCE_MODE_GOLD_SUITE_VISIBLE and expected_test_ids:
                # Prompt blackboard records must stay bounded; the full suite remains
                # available through TestContext/EvaluationConstraints and workspace files.
                inventory_payload.update(
                    {
                        "test_id_count": len(expected_test_ids),
                        "test_ids_truncated": len(visible_expected_test_ids)
                        < len(expected_test_ids),
                        "omitted_test_id_count": max(
                            0,
                            len(expected_test_ids) - len(visible_expected_test_ids),
                        ),
                    }
                )
            description = "Known visible test surface"
            if policy.mode == EVIDENCE_MODE_GOLD_SUITE_VISIBLE:
                description = "Gold visible test suite is the declared evaluation target"
            elif policy.mode == EVIDENCE_MODE_PARTIAL_SUITE_VISIBLE:
                description = "Visible tests are partial or incomplete contract evidence"
            elif policy.mode == EVIDENCE_MODE_NO_SUITE_VISIBLE:
                description = "No complete visible test suite is declared"
            blackboard.append(
                BlackboardRecord(
                    record_type=RECORD_TEST_INVENTORY,
                    description=description,
                    provenance=PROVENANCE_AGENT_VISIBLE,
                    confidence=0.95,
                    source=test_context.test_inventory_source or policy.provenance_label,
                    test_ids=(
                        visible_expected_test_ids
                        if policy.mode == EVIDENCE_MODE_GOLD_SUITE_VISIBLE
                        else visible_failing_test_ids
                    ),
                    payload=inventory_payload,
                )
            )

    for criterion in list(issue_plan.success_criteria or [])[:8]:
        blackboard.append(
            BlackboardRecord(
                record_type=RECORD_CONTRACT_OBLIGATION,
                description=str(criterion),
                provenance=PROVENANCE_AGENT_VISIBLE,
                confidence=0.82,
                source="success_criteria",
                file_paths=list(issue_plan.relevant_files[:4]),
            )
        )

    for expectation in list(test_context.expectations or [])[:8]:
        blackboard.append(
            BlackboardRecord(
                record_type=RECORD_CONTRACT_OBLIGATION,
                description=f"Preserve visible-test expectation: {expectation}",
                provenance=PROVENANCE_AGENT_VISIBLE,
                confidence=0.9 if policy.mode == EVIDENCE_MODE_GOLD_SUITE_VISIBLE else 0.72,
                source="visible_test_expectation",
                file_paths=list(test_context.focus_test_files[:4]),
                test_ids=[expectation],
            )
        )

    failure_items = _dedupe(
        list(test_context.failing_test_ids or [])
        + list(test_context.exception_summaries or [])
        + list(test_context.terminal_source_files or [])
    )
    if failure_items:
        blackboard.append(
            BlackboardRecord(
                record_type=RECORD_FAILURE_FRONTIER,
                description="Current rollout failure frontier: " + "; ".join(failure_items[:4]),
                provenance=PROVENANCE_VERIFIED,
                confidence=0.86,
                source="baseline_test_context",
                file_paths=list(test_context.terminal_source_files[:4])
                + list(test_context.source_focus_files[:4]),
                test_ids=list(test_context.failing_test_ids[:6]),
                payload={"failure_items": failure_items[:12]},
            )
        )

    focus_files = _dedupe(
        list(test_context.terminal_source_files or [])
        + list(test_context.source_focus_files or [])
        + list(test_context.incomplete_source_files or [])
        + list(issue_plan.risk_files or [])
        + list(issue_plan.relevant_files or [])
    )
    if focus_files:
        blackboard.append(
            BlackboardRecord(
                record_type=RECORD_LOCALIZATION_HYPOTHESIS,
                description="Initial implementation search focus: " + ", ".join(focus_files[:5]),
                provenance=PROVENANCE_SPECULATIVE,
                confidence=0.62,
                source="planner_localization",
                file_paths=focus_files[:8],
            )
        )

    if policy.mode == EVIDENCE_MODE_PARTIAL_SUITE_VISIBLE:
        blackboard.append(
            BlackboardRecord(
                record_type=RECORD_HIDDEN_RISK_REVIEW,
                description=(
                    "Do not stop at visible-test pass rate; check issue requirements, "
                    "interface compatibility, nearby regressions, and generated edge cases."
                ),
                provenance=PROVENANCE_AGENT_VISIBLE,
                confidence=0.88,
                source="evidence_policy",
            )
        )
    elif policy.mode == EVIDENCE_MODE_GOLD_SUITE_VISIBLE:
        blackboard.append(
            BlackboardRecord(
                record_type=RECORD_VERIFICATION,
                description=(
                    "Primary progress signal is the declared gold visible suite: preserve "
                    "collection and reduce failing/missing expected tests."
                ),
                provenance=PROVENANCE_AGENT_VISIBLE,
                confidence=0.93,
                source="evidence_policy",
            )
        )
    elif policy.mode == EVIDENCE_MODE_NO_SUITE_VISIBLE:
        blackboard.append(
            BlackboardRecord(
                record_type=RECORD_FAILURE_FRONTIER,
                description="Create a focused repro or validation command before broad edits.",
                provenance=PROVENANCE_AGENT_VISIBLE,
                confidence=0.8,
                source="evidence_policy",
            )
        )

    return blackboard


def blackboard_context_from_issue_plan(issue_plan: "IssuePlan") -> dict[str, Any]:
    """Return an agent-visible task_state_context fragment."""

    blackboard = build_initial_task_blackboard(issue_plan)
    payload = blackboard.to_dict(agent_visible_only=True)
    records = list(payload.get("records") or [])
    return {
        "blackboard": payload,
        "blackboard_record_count": len(records),
        "blackboard_record_type_counts": {
            record_type: sum(
                1 for record in records if str(record.get("record_type") or "") == record_type
            )
            for record_type in sorted(
                {
                    str(record.get("record_type") or "")
                    for record in records
                    if str(record.get("record_type") or "")
                }
            )
        },
    }


def failure_frontier_from_quick_verification(payload: dict[str, Any]) -> dict[str, Any]:
    """Summarize quick-verification failures for follow-up scheduling."""

    if not isinstance(payload, dict) or not payload:
        return {}
    failed_tests = _dedupe(list(payload.get("failed_tests") or []))[:12]
    missing_expected = _dedupe(list(payload.get("missing_expected_test_ids") or []))[:12]
    failure_clusters = []
    for item in list(payload.get("failure_clusters") or [])[:4]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("summary") or item.get("message") or item.get("nodeid") or "").strip()
        if text:
            failure_clusters.append(text)
    classification = (
        dict(payload.get("failure_classification") or {})
        if isinstance(payload.get("failure_classification"), dict)
        else {}
    )
    label = str(classification.get("label") or "").strip()
    output_excerpt = str(payload.get("output_excerpt") or "").strip()
    description_parts = []
    if label:
        description_parts.append(f"class={label}")
    if missing_expected:
        description_parts.append(f"missing_expected={len(missing_expected)}")
    if failed_tests:
        description_parts.append(f"failed_tests={len(failed_tests)}")
    if failure_clusters:
        description_parts.append(f"clusters={len(failure_clusters)}")
    if not description_parts and output_excerpt:
        first_line = re.sub(r"\s+", " ", output_excerpt.splitlines()[0]).strip()
        description_parts.append(first_line[:160])
    if not description_parts:
        return {}
    return {
        "record_type": RECORD_FAILURE_FRONTIER,
        "description": "Quick verification frontier: " + "; ".join(description_parts),
        "provenance": PROVENANCE_VERIFIED,
        "confidence": 0.86,
        "source": "quick_verification",
        "test_ids": failed_tests + missing_expected,
        "payload": {
            "failed_tests": failed_tests,
            "missing_expected_test_ids": missing_expected,
            "failure_clusters": failure_clusters,
            "failure_classification": classification,
        },
    }
