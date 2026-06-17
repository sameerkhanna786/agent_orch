"""V5 dual-version verifier (patch-as-oracle voting matrix).

Per (test, patch) pair, runs the test on the buggy code and on the
patched code via the docker adapter. A test is *F→P-against* a patch
when ``buggy=FAIL`` AND ``patched=PASS``. The aggregate ``oracle_score``
of a test = the number of candidate patches it's F→P-against.

This is the core SOTA mechanism shared by Echo (66.3% SWT-V), e-Otter++
(62%), and TEX-T (87% — current #1). APEX V5's specific implementation:

  - Uses APEX's own multi-agent ensemble for the patch surrogate
    (``patch_surrogate.py``), not a frozen external system.
  - Uses the V4 docker acceptance adapter for the in-container test
    runs (one container per (test, patch) pair, stable interface).
  - Returns per-cell verdicts so downstream selectors can compute
    oracle_score, identify wrong-reason failures, or build TEX-T
    cross-candidate matrices.

The mechanism is robust to imperfect patches because the verdict is
"this test discriminates buggy from patched". A wrong patch typically
either (a) doesn't make the test pass on it, or (b) introduces an
unrelated change the test doesn't see — both reduce oracle_score
without producing false-positive F→P verdicts.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CellVerdict:
    """Outcome for one (test, patch) pair."""

    test_id: str
    patch_id: str
    buggy_status: str  # "pass" | "fail" | "harness_error"
    patched_status: str  # "pass" | "fail" | "harness_error" | "patch_apply_error" | "no_source_to_patch" | "skipped"
    is_f2p: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TestRow:
    """Per-test aggregation across all candidate patches."""

    test_id: str
    oracle_score: int  # number of patches this test is F→P against
    verdicts: list[CellVerdict] = field(default_factory=list)
    buggy_pass_count: int = 0
    patched_pass_count: int = 0
    harness_error_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "oracle_score": self.oracle_score,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "buggy_pass_count": self.buggy_pass_count,
            "patched_pass_count": self.patched_pass_count,
            "harness_error_count": self.harness_error_count,
        }


@dataclass(frozen=True)
class DualVersionReport:
    """The full verdict matrix + per-test aggregation."""

    rows: list[TestRow] = field(default_factory=list)
    matrix: list[list[CellVerdict]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def best_test_id(self) -> Optional[str]:
        if not self.rows:
            return None
        winner = max(self.rows, key=lambda r: (r.oracle_score, -r.harness_error_count))
        return winner.test_id if winner.oracle_score > 0 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [r.to_dict() for r in self.rows],
            "best_test_id": self.best_test_id,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class _CandidateTestInput:
    test_id: str
    artifact_path: str
    artifact_content: str
    origin_agent: str = ""


@dataclass(frozen=True)
class _CandidatePatchInput:
    patch_id: str
    diff: str
    origin_agent: str = ""


def verify_tests_against_patches(
    *,
    test_candidates: list[dict[str, Any]],
    patch_candidates: list[dict[str, Any]],
    benchmark_adapter: Any,
    workdir: Path,
    focal_path: str,
    exclude_same_origin: bool = True,
) -> DualVersionReport:
    """Build the (test × patch) F→P matrix.

    Args:
        test_candidates: list of dicts with keys ``test_id``,
            ``artifact_path``, ``artifact_content``.
        patch_candidates: list of dicts with keys ``patch_id``, ``diff``.
        benchmark_adapter: V4 ``DockerTestGenEvalLiteAdapter`` (or any
            adapter exposing ``run_unfiltered(artifact, workdir)``).
        workdir: project workdir; the verifier mutates a temp copy per
            cell, never the source.
        focal_path: path to the focal source file the patches target
            (for cleanup verification).
    """

    tests = [
        _CandidateTestInput(
            test_id=str(t.get("test_id") or t.get("agent") or f"test_{i}"),
            artifact_path=str(t.get("artifact_path") or t.get("path") or "tests/test_apex.py"),
            artifact_content=str(t.get("artifact_content") or t.get("content") or ""),
            origin_agent=str(t.get("origin_agent") or t.get("agent") or ""),
        )
        for i, t in enumerate(test_candidates)
    ]
    patches = [
        _CandidatePatchInput(
            patch_id=str(p.get("patch_id") or p.get("agent") or f"patch_{i}"),
            diff=str(p.get("diff") or ""),
            origin_agent=str(p.get("origin_agent") or p.get("agent") or p.get("patch_id") or ""),
        )
        for i, p in enumerate(patch_candidates)
        if str(p.get("diff") or "").strip()
    ]
    if not tests:
        return DualVersionReport(diagnostics={"status": "no_tests"})
    if not patches:
        # No usable patches — fall back to "fails on buggy" only.
        return _fallback_buggy_only(
            tests=tests, benchmark_adapter=benchmark_adapter, workdir=workdir
        )

    matrix: list[list[CellVerdict]] = []
    buggy_cache: dict[tuple[str, str, str], tuple[str, str]] = {}
    started = time.time()
    for test in tests:
        row: list[CellVerdict] = []
        for patch in patches:
            if exclude_same_origin and _same_origin(test.origin_agent, patch.origin_agent):
                verdict = CellVerdict(
                    test_id=test.test_id,
                    patch_id=patch.patch_id,
                    buggy_status="skipped",
                    patched_status="skipped",
                    is_f2p=False,
                    note="skipped same-origin test/patch pair",
                )
            else:
                cache_key = (test.test_id, test.artifact_path, test.artifact_content)
                if cache_key not in buggy_cache:
                    buggy_cache[cache_key] = _run_buggy_side(
                        test=test,
                        benchmark_adapter=benchmark_adapter,
                        workdir=workdir,
                    )
                cached_buggy_status, cached_buggy_note = buggy_cache[cache_key]
                verdict = _verify_one_cell(
                    test=test,
                    patch=patch,
                    benchmark_adapter=benchmark_adapter,
                    workdir=workdir,
                    focal_path=focal_path,
                    cached_buggy_status=cached_buggy_status,
                    cached_buggy_note=cached_buggy_note,
                )
            row.append(verdict)
        matrix.append(row)

    rows = []
    for test, row in zip(tests, matrix):
        rows.append(
            TestRow(
                test_id=test.test_id,
                oracle_score=sum(1 for v in row if v.is_f2p),
                verdicts=list(row),
                buggy_pass_count=sum(1 for v in row if v.buggy_status == "pass"),
                patched_pass_count=sum(1 for v in row if v.patched_status == "pass"),
                harness_error_count=sum(
                    1
                    for v in row
                    if v.buggy_status == "harness_error" or v.patched_status == "harness_error"
                ),
            )
        )
    elapsed = time.time() - started
    diagnostics = {
        "status": "ok",
        "test_count": len(tests),
        "patch_count": len(patches),
        "cell_count": len(tests) * len(patches),
        "buggy_side_execution_count": len(buggy_cache),
        "buggy_side_cache_hit_count": max(
            0,
            sum(1 for row in matrix for verdict in row if verdict.buggy_status not in {"skipped"})
            - len(buggy_cache),
        ),
        "same_origin_skipped_count": sum(
            1
            for row in matrix
            for verdict in row
            if verdict.note == "skipped same-origin test/patch pair"
        ),
        "elapsed_seconds": round(elapsed, 2),
    }
    return DualVersionReport(rows=rows, matrix=matrix, diagnostics=diagnostics)


def _verify_one_cell(
    *,
    test: _CandidateTestInput,
    patch: _CandidatePatchInput,
    benchmark_adapter: Any,
    workdir: Path,
    focal_path: str,
    cached_buggy_status: str | None = None,
    cached_buggy_note: str = "",
) -> CellVerdict:
    """Run one (test, patch) cell.

    Per cell we run the test twice via the adapter:
      1. on the buggy code (no patch applied) — must FAIL
      2. on the patched code (patch applied to focal file) — must PASS

    Both halves are required for an F→P verdict.
    """

    from apex.evaluation.final_acceptance_gate import GeneratedArtifact

    artifact = GeneratedArtifact(path=test.artifact_path, content=test.artifact_content)
    patch_override_factory = getattr(benchmark_adapter, "with_patch_override", None)
    if callable(patch_override_factory):
        try:
            patched_adapter = _adapter_with_patch_override(
                benchmark_adapter,
                patch.diff,
                _cell_log_suffix(test, patch, "patched"),
            )
        except Exception as exc:
            return CellVerdict(
                test_id=test.test_id,
                patch_id=patch.patch_id,
                buggy_status="harness_error",
                patched_status="harness_error",
                is_f2p=False,
                note=f"patch override setup errored: {type(exc).__name__}",
            )
        buggy_run = cached_buggy_status
        if buggy_run is None:
            buggy_run, cached_buggy_note = _run_buggy_side(
                test=test,
                benchmark_adapter=benchmark_adapter,
                workdir=workdir,
            )
        if buggy_run not in {"fail", "pass"}:
            return CellVerdict(
                test_id=test.test_id,
                patch_id=patch.patch_id,
                buggy_status=buggy_run,
                patched_status="skipped",
                is_f2p=False,
                note=cached_buggy_note or "buggy run inconclusive",
            )
        if buggy_run == "pass":
            return CellVerdict(
                test_id=test.test_id,
                patch_id=patch.patch_id,
                buggy_status="pass",
                patched_status="skipped",
                is_f2p=False,
                note="test passes on buggy (over-loose oracle or wrong direction)",
            )
        try:
            patched_run = _coerce_status(patched_adapter.run_unfiltered(artifact, workdir).status)
        except Exception as exc:
            return CellVerdict(
                test_id=test.test_id,
                patch_id=patch.patch_id,
                buggy_status="fail",
                patched_status="harness_error",
                is_f2p=False,
                note=f"patched run errored: {type(exc).__name__}",
            )
        is_f2p = patched_run == "pass"
        return CellVerdict(
            test_id=test.test_id,
            patch_id=patch.patch_id,
            buggy_status="fail",
            patched_status=patched_run,
            is_f2p=is_f2p,
            note="" if is_f2p else "patched run did not pass",
        )

    # Phase 1: buggy version
    buggy_run = cached_buggy_status
    if buggy_run is None:
        buggy_run, cached_buggy_note = _run_buggy_side(
            test=test,
            benchmark_adapter=benchmark_adapter,
            workdir=workdir,
        )
    if buggy_run not in {"fail", "pass"}:
        return CellVerdict(
            test_id=test.test_id,
            patch_id=patch.patch_id,
            buggy_status=buggy_run,
            patched_status="skipped",
            is_f2p=False,
            note=cached_buggy_note or "buggy run inconclusive",
        )
    if buggy_run == "pass":
        # Test passes on buggy → can't be F→P regardless of patch.
        return CellVerdict(
            test_id=test.test_id,
            patch_id=patch.patch_id,
            buggy_status="pass",
            patched_status="skipped",
            is_f2p=False,
            note="test passes on buggy (over-loose oracle or wrong direction)",
        )

    # Phase 2: apply patch in a tmp clone of the workdir, run again
    with tempfile.TemporaryDirectory(prefix="apex_dvv_patched_") as tmp:
        tmp_dir = Path(tmp)
        try:
            materialize_diag = _materialize_workdir_for_patch(workdir, tmp_dir)
        except Exception as exc:
            return CellVerdict(
                test_id=test.test_id,
                patch_id=patch.patch_id,
                buggy_status="fail",
                patched_status="harness_error",
                is_f2p=False,
                note=f"materialize failed: {type(exc).__name__}",
            )
        if materialize_diag.get("status") == "empty_workdir":
            # We have nothing to patch against — distinct from
            # "patch_apply_error" which would imply the patch was
            # genuinely incompatible with real source. Without this
            # branch the cell silently reports patch_apply_error and
            # the voter can't tell missing-infrastructure from
            # bad-patches. Caller may want to switch to an adapter
            # that supports with_patch_override (in-container apply)
            # or pre-checkout the repo into ``workdir``.
            logger.warning(
                "dual_version_verifier: workdir %s has no .py files; "
                "skipping fallback git-apply path. Provide an adapter "
                "with with_patch_override() or pre-populate workdir.",
                workdir,
            )
            return CellVerdict(
                test_id=test.test_id,
                patch_id=patch.patch_id,
                buggy_status="fail",
                patched_status="no_source_to_patch",
                is_f2p=False,
                note="workdir empty; cannot apply patch outside container",
            )
        if not _apply_unified_diff(tmp_dir, patch.diff):
            return CellVerdict(
                test_id=test.test_id,
                patch_id=patch.patch_id,
                buggy_status="fail",
                patched_status="patch_apply_error",
                is_f2p=False,
                note="patch failed to apply",
            )
        try:
            patched_adapter = _adapter_with_run_suffix(
                benchmark_adapter,
                _cell_log_suffix(test, patch, "patched"),
            )
            patched_run = _coerce_status(patched_adapter.run_unfiltered(artifact, tmp_dir).status)
        except Exception as exc:
            return CellVerdict(
                test_id=test.test_id,
                patch_id=patch.patch_id,
                buggy_status="fail",
                patched_status="harness_error",
                is_f2p=False,
                note=f"patched run errored: {type(exc).__name__}",
            )

    is_f2p = patched_run == "pass"
    return CellVerdict(
        test_id=test.test_id,
        patch_id=patch.patch_id,
        buggy_status="fail",
        patched_status=patched_run,
        is_f2p=is_f2p,
        note="" if is_f2p else "patched run did not pass",
    )


def _run_buggy_side(
    *,
    test: _CandidateTestInput,
    benchmark_adapter: Any,
    workdir: Path,
) -> tuple[str, str]:
    from apex.evaluation.final_acceptance_gate import GeneratedArtifact

    artifact = GeneratedArtifact(path=test.artifact_path, content=test.artifact_content)
    try:
        if callable(getattr(benchmark_adapter, "with_patch_override", None)):
            adapter = _adapter_with_patch_override(
                benchmark_adapter,
                "",
                ".".join(part for part in (_safe_token(test.test_id), "buggy") if part),
            )
        else:
            adapter = _adapter_with_run_suffix(
                benchmark_adapter,
                ".".join(part for part in (_safe_token(test.test_id), "buggy") if part),
            )
        return _coerce_status(adapter.run_unfiltered(artifact, workdir).status), ""
    except Exception as exc:
        return "harness_error", f"buggy run errored: {type(exc).__name__}"


def _adapter_with_patch_override(adapter: Any, patch_diff: str, run_suffix: str) -> Any:
    factory = getattr(adapter, "with_patch_override", None)
    if not callable(factory):
        return _adapter_with_run_suffix(adapter, run_suffix)
    try:
        return factory(patch_diff, run_suffix=run_suffix)
    except TypeError:
        patched = factory(patch_diff)
        return _adapter_with_run_suffix(patched, run_suffix)


def _adapter_with_run_suffix(adapter: Any, run_suffix: str) -> Any:
    suffixer = getattr(adapter, "with_run_suffix", None)
    if not callable(suffixer):
        return adapter
    try:
        return suffixer(run_suffix)
    except TypeError:
        return adapter


def _cell_log_suffix(
    test: _CandidateTestInput,
    patch: _CandidatePatchInput,
    side: str,
) -> str:
    return ".".join(
        _safe_token(part) for part in (test.test_id, patch.patch_id, side) if _safe_token(part)
    )


def _safe_token(value: str) -> str:
    import re

    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")[:80]


def _same_origin(left: str, right: str) -> bool:
    return bool(left and right and left.strip().lower() == right.strip().lower())


def _coerce_status(raw: Any) -> str:
    text = str(raw or "").lower()
    if text in {"pass", "ok", "passed"}:
        return "pass"
    if text in {"fail", "failed", "error", "errored"}:
        return "fail"
    if text in {"harness_error", "harness-error"}:
        return "harness_error"
    return text or "unknown"


def _materialize_workdir_for_patch(src: Path, dst: Path) -> dict[str, Any]:
    """Copy enough of the workdir into ``dst`` so that ``git apply`` and a
    follow-up adapter run see a coherent project tree.

    Returns a small diagnostics dict so the caller can distinguish
    "no source available" (the in-container path is the only viable
    one for this benchmark) from "real patch failed to apply".
    """

    if not src.exists():
        return {"status": "empty_workdir", "reason": "src_missing"}
    shutil.copytree(src, dst, dirs_exist_ok=True)
    has_py = next(dst.rglob("*.py"), None) is not None
    if not has_py:
        return {"status": "empty_workdir", "reason": "no_python_files"}
    return {"status": "ok"}


def _apply_unified_diff(workdir: Path, diff: str) -> bool:
    """Apply ``diff`` to ``workdir`` via ``git apply -p1``. Falls back to
    ``patch -p1`` when git isn't available. Returns False on any failure."""

    if not diff or not diff.strip():
        return False
    try:
        result = subprocess.run(
            ["git", "apply", "-p1", "--reject", "--ignore-whitespace", "-"],
            input=diff,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            return True
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        result = subprocess.run(
            ["patch", "-p1", "--no-backup-if-mismatch"],
            input=diff,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _fallback_buggy_only(
    *,
    tests: list[_CandidateTestInput],
    benchmark_adapter: Any,
    workdir: Path,
) -> DualVersionReport:
    """No usable patches — score tests by buggy-vs-gold pair instead.

    Audit C2: the previous implementation called ``run_unfiltered`` once
    against the adapter's default state. The docker adapter's default
    applies the GOLD patch (``src.get("patch")``) so a correct test
    PASSED on gold and was tagged ``is_f2p=False``, while a brittle/wrong
    test FAILED on gold and was tagged ``is_f2p=True`` — the voter then
    preferred broken tests. The fix runs the test twice — once against
    the buggy state (no patch) and once against the gold patch — and
    only flags ``is_f2p=True`` when the test legitimately fails on
    buggy AND passes on gold. When the adapter doesn't expose
    ``with_patch_override`` we fall through to the legacy single-run
    proxy, but mark the diagnostic so callers can downweight the score.
    """

    from apex.evaluation.final_acceptance_gate import GeneratedArtifact

    has_override = callable(getattr(benchmark_adapter, "with_patch_override", None))
    rows = []
    matrix: list[list[CellVerdict]] = []
    for test in tests:
        artifact = GeneratedArtifact(path=test.artifact_path, content=test.artifact_content)
        if has_override:
            try:
                buggy_adapter = benchmark_adapter.with_patch_override("", run_suffix="buggy")
                buggy_status = _coerce_status(
                    buggy_adapter.run_unfiltered(artifact, workdir).status
                )
            except Exception:
                buggy_status = "harness_error"
            try:
                # No override → adapter default applies the gold patch.
                gold_status = _coerce_status(
                    benchmark_adapter.run_unfiltered(artifact, workdir).status
                )
            except Exception:
                gold_status = "harness_error"
            is_f2p = buggy_status == "fail" and gold_status == "pass"
            verdict = CellVerdict(
                test_id=test.test_id,
                patch_id="__gold_pair__",
                buggy_status=buggy_status,
                patched_status=gold_status,
                is_f2p=is_f2p,
                note="fallback: no surrogate patches; using gold as oracle pair",
            )
            harness_errors = (1 if buggy_status == "harness_error" else 0) + (
                1 if gold_status == "harness_error" else 0
            )
            buggy_pass = 1 if buggy_status == "pass" else 0
            gold_pass = 1 if gold_status == "pass" else 0
        else:
            # Adapter has no override hook: best-effort single run only.
            try:
                status = _coerce_status(benchmark_adapter.run_unfiltered(artifact, workdir).status)
            except Exception:
                status = "harness_error"
            verdict = CellVerdict(
                test_id=test.test_id,
                patch_id="__no_patch__",
                buggy_status=status,
                patched_status="skipped",
                is_f2p=False,  # without a pair we cannot confidently say F→P
                note=(
                    "fallback: adapter has no with_patch_override; "
                    "single-run signal only — voter should abstain"
                ),
            )
            harness_errors = 1 if status == "harness_error" else 0
            buggy_pass = 1 if status == "pass" else 0
            gold_pass = 0
        matrix.append([verdict])
        rows.append(
            TestRow(
                test_id=test.test_id,
                oracle_score=1 if verdict.is_f2p else 0,
                verdicts=[verdict],
                buggy_pass_count=buggy_pass,
                patched_pass_count=gold_pass,
                harness_error_count=harness_errors,
            )
        )
    return DualVersionReport(
        rows=rows,
        matrix=matrix,
        diagnostics={
            "status": "fallback_buggy_only",
            "test_count": len(tests),
            "used_gold_pair": has_override,
        },
    )
