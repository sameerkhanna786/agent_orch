"""Evidence-bound adversarial review for selector candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _changed_files(candidate: Any) -> list[str]:
    values = getattr(candidate, "changed_files", None)
    if isinstance(values, (list, tuple, set)):
        return [str(value).strip() for value in values if str(value).strip()]
    return []


def _is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    return (
        "/tests/" in f"/{normalized}"
        or "/test/" in f"/{normalized}"
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _patch_stats(patch: Any) -> dict[str, Any]:
    text = str(patch or "")
    added = 0
    removed = 0
    skipped = 0
    deleted_test_files = 0
    changed_inventory = False
    for line in text.splitlines():
        if line.startswith("diff --git "):
            changed_inventory = changed_inventory or any(
                marker in line
                for marker in (
                    ".apex_expected_test_ids",
                    "_apex_expected_test_ids",
                    "expected_test_ids",
                    "pytest_report_utils",
                )
            )
        if line.startswith("deleted file mode") and "test" in text.lower():
            deleted_test_files += 1
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
            lowered = line.lower()
            if "pytest.skip" in lowered or "mark.skip" in lowered or "xfail" in lowered:
                skipped += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return {
        "added_lines": added,
        "removed_lines": removed,
        "skip_or_xfail_added_lines": skipped,
        "deleted_test_files": deleted_test_files,
        "changed_test_inventory": changed_inventory,
    }


@dataclass(frozen=True)
class AdversarialReview:
    veto: bool = False
    risk_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "veto": bool(self.veto),
            "risk_score": round(max(0.0, min(float(self.risk_score), 1.0)), 4),
            "reasons": list(self.reasons),
            "features": dict(self.features),
        }


def review_candidate(candidate: Any, verification: Any = None) -> AdversarialReview:
    """Try to disprove a candidate using concrete evidence only."""

    reasons: list[str] = []
    risk = 0.0
    validity = getattr(candidate, "validity", None)
    validity_payload = validity.as_dict() if hasattr(validity, "as_dict") else _as_dict(validity)
    test_result = getattr(verification, "test_result", None) if verification is not None else None
    patch_features = _patch_stats(getattr(candidate, "patch", None))
    changed_files = _changed_files(candidate)
    test_file_edits = [path for path in changed_files if _is_test_path(path)]

    if validity_payload.get("protected_tests_unchanged") is False:
        reasons.append("protected tests changed")
        risk += 1.0
    if validity_payload.get("coverage_collapse_terminal") or validity_payload.get(
        "expected_coverage_preserved"
    ) is False:
        reasons.append("expected test coverage collapsed")
        risk += 1.0
    if getattr(test_result, "expected_coverage_preserved", None) is False:
        reasons.append("selector verifier observed expected coverage collapse")
        risk += 1.0
    if patch_features["changed_test_inventory"]:
        reasons.append("patch changes test inventory or harness helper")
        risk += 1.0
    if patch_features["deleted_test_files"] > 0:
        reasons.append("patch deletes tests")
        risk += 0.7
    if patch_features["skip_or_xfail_added_lines"] > 0:
        reasons.append("patch adds skip or xfail markers")
        risk += 0.35
    if test_file_edits:
        reasons.append("patch edits test files")
        risk += min(0.25, 0.05 * len(test_file_edits))
    removed = int(patch_features["removed_lines"])
    added = int(patch_features["added_lines"])
    if removed >= 40 and removed > max(added * 2, added + 25):
        reasons.append("large deletion-heavy patch")
        risk += 0.35
    if verification is not None and getattr(verification, "quality_gate_passed", None) is False:
        reasons.append("quality gate rejected candidate")
        risk += 1.0

    veto = any(
        reason
        in {
            "protected tests changed",
            "expected test coverage collapsed",
            "selector verifier observed expected coverage collapse",
            "patch changes test inventory or harness helper",
            "quality gate rejected candidate",
        }
        for reason in reasons
    )
    return AdversarialReview(
        veto=veto,
        risk_score=max(0.0, min(risk, 1.0)),
        reasons=list(dict.fromkeys(reasons)),
        features={
            **patch_features,
            "changed_file_count": len(changed_files),
            "test_file_edit_count": len(test_file_edits),
        },
    )
