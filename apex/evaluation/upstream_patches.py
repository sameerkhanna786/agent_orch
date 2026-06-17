"""Helpers for applying checked-in benchmark harness patches."""

from __future__ import annotations

import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UpstreamPatchResult:
    patch_path: str
    status: str
    detail: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_upstream_patch_directory(
    repo_dir: str | Path,
    patch_dir: str | Path,
    *,
    strip: int = 1,
) -> dict[str, Any]:
    """Apply every ``*.patch`` in ``patch_dir`` to ``repo_dir`` if possible.

    Patches are idempotent: already-applied patches are reported as
    ``already_applied``. Non-applicable patches are reported but do not raise;
    benchmark runners can decide whether those diagnostics are fatal.
    """

    repo = Path(repo_dir).expanduser()
    patches = sorted(Path(patch_dir).expanduser().glob("*.patch"))
    started = time.time()
    results = [
        apply_upstream_patch(repo, patch_path, strip=strip).to_dict() for patch_path in patches
    ]
    status_counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "status": "ok"
        if all(r["status"] in {"applied", "already_applied"} for r in results)
        else "partial",
        "repo_dir": str(repo),
        "patch_dir": str(Path(patch_dir).expanduser()),
        "patch_count": len(results),
        "status_counts": status_counts,
        "results": results,
        "duration_seconds": round(time.time() - started, 3),
    }


def apply_upstream_patch(
    repo_dir: str | Path,
    patch_path: str | Path,
    *,
    strip: int = 1,
) -> UpstreamPatchResult:
    repo = Path(repo_dir).expanduser()
    patch = Path(patch_path).expanduser()
    started = time.time()
    if not repo.exists():
        return UpstreamPatchResult(
            patch_path=str(patch),
            status="repo_missing",
            detail=str(repo),
            duration_seconds=round(time.time() - started, 3),
        )
    if not patch.exists():
        return UpstreamPatchResult(
            patch_path=str(patch),
            status="patch_missing",
            duration_seconds=round(time.time() - started, 3),
        )
    check = _git_apply(repo, patch, "--check", strip=strip)
    if check.returncode == 0:
        applied = _git_apply(repo, patch, strip=strip)
        return UpstreamPatchResult(
            patch_path=str(patch),
            status="applied" if applied.returncode == 0 else "failed",
            detail=_completed_tail(applied),
            duration_seconds=round(time.time() - started, 3),
        )
    reverse = _git_apply(repo, patch, "--reverse", "--check", strip=strip)
    if reverse.returncode == 0:
        return UpstreamPatchResult(
            patch_path=str(patch),
            status="already_applied",
            detail=_completed_tail(reverse),
            duration_seconds=round(time.time() - started, 3),
        )
    return UpstreamPatchResult(
        patch_path=str(patch),
        status="not_applicable",
        detail=_completed_tail(check),
        duration_seconds=round(time.time() - started, 3),
    )


def testgeneval_patch_dir() -> Path:
    return Path(__file__).resolve().parent / "upstream_patches" / "testgeneval"


# Patch files that are *always* safe to apply alongside running the
# upstream-canonical scoring path (they fix harness crashes, not
# scoring policy). The list is consulted by
# ``apply_testgeneval_upstream_patches`` when ``baseline_covs_only`` is
# True so the ``TestGenEvalUpstreamScorer`` can keep ``generate_report``
# from KeyError-aborting on lite rows without otherwise diverging from
# the published baseline.
_TESTGENEVAL_BASELINE_COVS_ONLY_PATCHES: tuple[str, ...] = ("baseline_covs_keyerror.patch",)


def apply_testgeneval_upstream_patches(
    repo_dir: str | Path,
    *,
    baseline_covs_only: bool = False,
) -> dict[str, Any]:
    """Apply APEX-maintained patches against ``kjain14/testgeneval``.

    Args:
        repo_dir: Path to the (vendored) testgeneval checkout.
        baseline_covs_only: If True, apply only the defensive
            ``baseline_covs_keyerror.patch`` so the upstream
            ``generate_report.py`` does not crash on lite-subset rows.
            Used by the upstream-canonical scoring path of the
            fairness audit so the harness behaviour otherwise matches
            the published baseline.
    """
    patch_dir = testgeneval_patch_dir()
    if not baseline_covs_only:
        return apply_upstream_patch_directory(repo_dir, patch_dir, strip=1)
    repo = Path(repo_dir).expanduser()
    started = time.time()
    results: list[dict[str, Any]] = []
    for filename in _TESTGENEVAL_BASELINE_COVS_ONLY_PATCHES:
        patch_path = patch_dir / filename
        results.append(apply_upstream_patch(repo, patch_path, strip=1).to_dict())
    status_counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "status": "ok"
        if all(r["status"] in {"applied", "already_applied"} for r in results)
        else "partial",
        "repo_dir": str(repo),
        "patch_dir": str(patch_dir),
        "patch_count": len(results),
        "status_counts": status_counts,
        "results": results,
        "duration_seconds": round(time.time() - started, 3),
        "baseline_covs_only": True,
    }


def _git_apply(
    repo_dir: Path,
    patch_path: Path,
    *args: str,
    strip: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "apply",
            f"-p{int(strip)}",
            *args,
            str(patch_path),
        ],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _completed_tail(completed: subprocess.CompletedProcess[str]) -> str:
    text = (completed.stderr or completed.stdout or "").strip()
    return text[-4000:]
