"""Repo/task scoped memory for failed rollout patterns."""

from __future__ import annotations

from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def summarize_failed_rollout(rollout: Any) -> dict[str, Any]:
    quick = _as_dict(getattr(rollout, "quick_verification", None))
    verification = _as_dict(getattr(rollout, "verification", None))
    taxonomy = _as_dict(verification.get("verification_taxonomy"))
    patch_artifact = _as_dict(getattr(rollout, "patch_artifact", None))
    changed_files = _string_list(getattr(rollout, "changed_files", []))
    if not changed_files:
        changed_files = _string_list(patch_artifact.get("changed_files"))
    summary = str(
        getattr(rollout, "explanation", None)
        or patch_artifact.get("summary")
        or getattr(rollout, "failure_reason", None)
        or "failed rollout"
    ).strip()
    failed_tests = _string_list(quick.get("failed_tests"))[:8]
    passed_tests = _string_list(quick.get("passed_tests"))[:8]
    root_failure = str(
        taxonomy.get("kind")
        or quick.get("failure_class")
        or quick.get("failure_classification")
        or getattr(rollout, "failure_class", None)
        or "unknown"
    )
    return {
        "rollout_id": getattr(rollout, "rollout_id", None),
        "hypothesis": summary[:360],
        "files_edited": changed_files[:12],
        "tests_improved": passed_tests,
        "tests_regressed": failed_tests,
        "root_failure": root_failure,
        "quick_counts": {
            "passed": quick.get("passed"),
            "failed": quick.get("failed"),
            "errors": quick.get("errors"),
            "returncode": quick.get("returncode"),
        },
    }


def summarize_failed_rollouts(rollouts: list[Any], *, limit: int = 8) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for rollout in rollouts:
        verification = _as_dict(getattr(rollout, "verification", None))
        accepted = bool(verification.get("accepted") or getattr(rollout, "internally_accepted", False))
        success = bool(getattr(rollout, "success", False))
        if accepted:
            continue
        if success and not _as_dict(getattr(rollout, "quick_verification", None)).get("returncode"):
            continue
        summaries.append(summarize_failed_rollout(rollout))
    return summaries[: max(0, limit)]


def material_difference_from_failed_attempts(
    candidate: Any,
    summaries: list[dict[str, Any]],
    *,
    min_overlap: float = 0.75,
) -> dict[str, Any]:
    candidate_files = set(_string_list(getattr(candidate, "changed_files", [])))
    quick = _as_dict(getattr(candidate, "quick_verification", None))
    repeated: list[dict[str, Any]] = []
    for summary in summaries:
        failed_files = set(_string_list(summary.get("files_edited")))
        if not failed_files or not candidate_files:
            continue
        overlap = len(candidate_files & failed_files) / max(len(candidate_files | failed_files), 1)
        counts = _as_dict(summary.get("quick_counts"))
        same_counts = all(
            quick.get(key) == counts.get(key)
            for key in ("failed", "errors", "returncode")
            if quick.get(key) is not None or counts.get(key) is not None
        )
        if overlap >= min_overlap and same_counts:
            repeated.append(
                {
                    "rollout_id": summary.get("rollout_id"),
                    "file_overlap": round(overlap, 4),
                    "root_failure": summary.get("root_failure"),
                }
            )
    return {
        "materially_different": not repeated,
        "repeated_failed_attempts": repeated[:4],
    }
