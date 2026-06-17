"""Process-quality scoring for rollout candidates."""

from __future__ import annotations

from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _patch_line_stats(patch: Any) -> tuple[int, int]:
    added = 0
    removed = 0
    for line in str(patch or "").splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def _changed_files(candidate: Any) -> list[str]:
    values = getattr(candidate, "changed_files", None)
    if isinstance(values, (list, tuple, set)):
        return [str(value).strip() for value in values if str(value).strip()]
    return []


def score_process_quality(candidate: Any, verification: Any = None) -> dict[str, Any]:
    """Return a [0, 1] process score plus concrete penalty reasons."""

    score = 0.72
    penalties: list[str] = []
    bonuses: list[str] = []
    quick = _as_dict(getattr(candidate, "quick_verification", None))
    trajectory = list(getattr(candidate, "trajectory", []) or [])
    patch_artifact = _as_dict(getattr(candidate, "patch_artifact", None))
    changed_files = _changed_files(candidate)
    added, removed = _patch_line_stats(getattr(candidate, "patch", None))

    if not quick and verification is None:
        score -= 0.20
        penalties.append("missing verification evidence")
    if quick:
        score += 0.08
        bonuses.append("has rollout quick verification")
        if quick.get("scope") == "full_test_command":
            score += 0.08
            bonuses.append("ran full test command")
        if bool(quick.get("timed_out") or quick.get("full_scope_timed_out")):
            score -= 0.12
            penalties.append("verification timeout")
    if verification is not None and getattr(verification, "test_result", None) is not None:
        score += 0.08
        bonuses.append("has selector verification result")
    if changed_files and not patch_artifact.get("tests_run") and not quick:
        score -= 0.10
        penalties.append("edited files without recorded tests")
    if len(changed_files) >= 20:
        score -= 0.12
        penalties.append("broad rewrite footprint")
    if removed >= 120 or (removed >= 40 and removed > max(added * 2, added + 25)):
        score -= 0.12
        penalties.append("deletion-heavy patch")
    if any(
        path.replace("\\", "/").startswith(("tests/", "test/"))
        or "/tests/" in path.replace("\\", "/")
        for path in changed_files
    ):
        score -= 0.08
        penalties.append("test file edits")
    round_count = 0
    for entry in trajectory:
        if isinstance(entry, dict):
            round_count += int(entry.get("rounds_used") or 0)
            iterations = entry.get("iterations")
            if isinstance(iterations, list):
                round_count += len(iterations)
    if round_count >= 5 and float(getattr(candidate, "progress_score", 0.0) or 0.0) < 0.4:
        score -= 0.10
        penalties.append("late trajectory thrash")
    if bool(getattr(candidate, "selected_for_submission", False)):
        score += 0.03
    return {
        "score": round(max(0.0, min(score, 1.0)), 4),
        "penalties": list(dict.fromkeys(penalties)),
        "bonuses": list(dict.fromkeys(bonuses)),
        "features": {
            "changed_file_count": len(changed_files),
            "added_lines": added,
            "removed_lines": removed,
            "trajectory_round_count": round_count,
        },
    }
