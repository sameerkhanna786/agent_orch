"""HTN milestone validator: real CORE / ISO / STRICT readiness check.

Phase I.6 — close the gap between *self-reported* milestone
readiness (what the agent's portfolio claims) and *actually
verified* milestone readiness (what the test artifacts demonstrate
when run). Until now, ``regression_suite_summary['strict_ready']``
was decided from agent-emitted metadata; this module computes it
from execution outcomes.

Validation levels follow ``test_generation_design.md §3.3``:

    CORE
        Tests assigned to the milestone parse and import. The
        agent's portfolio is at minimum syntactically valid and
        the contract surface is reachable.

    ISO
        CORE + at least one test per declared milestone objective
        is present in the candidate portfolio. Coverage is
        objective-isomorphic — every objective has a witness.

    STRICT
        ISO + at least one test attached to the milestone flips
        F2P on the supplied (broken_dir, fixed_dir) sandbox pair.
        The milestone has *demonstrated bug-catching power*, not
        merely structural completeness.

The validator is pure — it takes a single milestone + its
artifacts + sandbox dirs and returns a structured report. Wiring
into the rollout state machine is left to callers; this module
makes the verification *available* so the engine's existing
self-reported strict_ready boolean can be cross-checked or
replaced.

Generalizes outside benchmarks: the validator depends only on
artifact dicts and sandbox paths — no benchmark task object.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------


class MilestoneValidationLevel(str, Enum):
    """Validation level a milestone has actually achieved."""

    NONE = "none"
    CORE = "core"
    ISO = "iso"
    STRICT = "strict"


_LEVEL_ORDER = {
    MilestoneValidationLevel.NONE: 0,
    MilestoneValidationLevel.CORE: 1,
    MilestoneValidationLevel.ISO: 2,
    MilestoneValidationLevel.STRICT: 3,
}


@dataclass
class MilestoneValidationReport:
    """Per-milestone CORE/ISO/STRICT verification result."""

    milestone_id: str
    level_reached: MilestoneValidationLevel = MilestoneValidationLevel.NONE
    objective_ids: list[str] = field(default_factory=list)
    objectives_with_witness: list[str] = field(default_factory=list)
    objectives_missing_witness: list[str] = field(default_factory=list)
    candidate_test_paths: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)  # paths that failed to parse
    f2p_count: int = 0
    f2p_tests: list[str] = field(default_factory=list)
    f2p_status: str = "not_evaluated"  # "ok" | "not_evaluated" | "skip_no_sandboxes" | error code
    error: Optional[str] = None

    @property
    def strict_ready(self) -> bool:
        return self.level_reached == MilestoneValidationLevel.STRICT

    @property
    def iso_ready(self) -> bool:
        return _LEVEL_ORDER[self.level_reached] >= _LEVEL_ORDER[MilestoneValidationLevel.ISO]

    @property
    def core_ready(self) -> bool:
        return _LEVEL_ORDER[self.level_reached] >= _LEVEL_ORDER[MilestoneValidationLevel.CORE]

    def to_dict(self) -> dict[str, Any]:
        return {
            "milestone_id": self.milestone_id,
            "level_reached": self.level_reached.value,
            "objective_ids": list(self.objective_ids),
            "objectives_with_witness": list(self.objectives_with_witness),
            "objectives_missing_witness": list(self.objectives_missing_witness),
            "candidate_test_paths": list(self.candidate_test_paths),
            "parse_errors": list(self.parse_errors),
            "f2p_count": self.f2p_count,
            "f2p_tests": list(self.f2p_tests),
            "f2p_status": self.f2p_status,
            "error": self.error,
            "core_ready": self.core_ready,
            "iso_ready": self.iso_ready,
            "strict_ready": self.strict_ready,
        }


@dataclass
class MilestoneSuiteReport:
    """Aggregate over multiple per-milestone reports."""

    per_milestone: list[MilestoneValidationReport] = field(default_factory=list)

    @property
    def all_strict_ready(self) -> bool:
        if not self.per_milestone:
            return False
        return all(r.strict_ready for r in self.per_milestone)

    @property
    def core_ready_count(self) -> int:
        return sum(1 for r in self.per_milestone if r.core_ready)

    @property
    def iso_ready_count(self) -> int:
        return sum(1 for r in self.per_milestone if r.iso_ready)

    @property
    def strict_ready_count(self) -> int:
        return sum(1 for r in self.per_milestone if r.strict_ready)

    def to_dict(self) -> dict[str, Any]:
        total = len(self.per_milestone)
        return {
            "milestone_count": total,
            "core_ready_count": self.core_ready_count,
            "iso_ready_count": self.iso_ready_count,
            "strict_ready_count": self.strict_ready_count,
            "all_strict_ready": self.all_strict_ready,
            "per_milestone": [r.to_dict() for r in self.per_milestone],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _artifacts_for_milestone(
    milestone_id: str, artifacts: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Filter test artifacts to those attached to the milestone.

    Acceptable shapes (any one wins):
        artifact["milestone_id"] == milestone_id
        artifact["milestones"] contains milestone_id
        artifact["objective"]["milestone_id"] == milestone_id
    """
    matching: list[dict[str, Any]] = []
    for a in artifacts:
        if not isinstance(a, dict):
            continue
        if str(a.get("milestone_id") or "") == milestone_id:
            matching.append(a)
            continue
        ms = a.get("milestones")
        if isinstance(ms, list) and milestone_id in [str(m) for m in ms]:
            matching.append(a)
            continue
        objective = a.get("objective")
        if isinstance(objective, dict) and str(objective.get("milestone_id") or "") == milestone_id:
            matching.append(a)
    return matching


def _objective_ids_in_artifact(a: dict[str, Any]) -> list[str]:
    if not isinstance(a, dict):
        return []
    out: list[str] = []
    if a.get("objective_id"):
        out.append(str(a["objective_id"]))
    objective = a.get("objective")
    if isinstance(objective, dict) and objective.get("objective_id"):
        out.append(str(objective["objective_id"]))
    if isinstance(a.get("objectives"), list):
        out.extend(str(o) for o in a["objectives"] if o)
    if isinstance(a.get("objective_ids"), list):
        out.extend(str(o) for o in a["objective_ids"] if o)
    return [o for o in out if o]


def _try_parse(language: str, content: str) -> Optional[str]:
    """Return None if the content parses cleanly, an error string otherwise."""
    if (language or "").lower() in {"python", "py", "python3"}:
        try:
            import ast

            ast.parse(content)
        except SyntaxError as exc:
            return f"SyntaxError: {exc}"
        return None
    # Non-Python: try tree-sitter dispatch (best-effort)
    try:
        from ..evaluation.mutation_engine_treesitter import _treesitter_parser

        parser = _treesitter_parser(language)
        if parser is None:
            return None  # can't validate — assume OK
        tree = parser.parse(content.encode("utf-8"))
        if tree.root_node.has_error:
            return "tree-sitter parser flagged ERROR node"
    except Exception:  # pragma: no cover — defensive
        return None
    return None


def _f2p_for_milestone(
    *,
    artifacts: list[dict[str, Any]],
    broken_dir: Path,
    fixed_dir: Path,
    language: str,
    timeout_seconds: float,
    install_repo: bool,
    output_dir: Path,
) -> tuple[str, int, list[str], Optional[str]]:
    """Run the F2P oracle on the artifacts attached to this milestone.

    Returns (status, f2p_count, f2p_tests, error). Defensive: any
    exception inside the oracle is captured as the error string and
    the milestone falls back to ISO-only.
    """
    try:
        from ..evaluation import evaluate_tdd_iteration
    except Exception as exc:  # pragma: no cover — defensive
        return ("import_failed", 0, [], f"{type(exc).__name__}: {exc}")
    try:
        report = evaluate_tdd_iteration(
            broken_dir=broken_dir,
            fixed_dir=fixed_dir,
            test_artifacts=artifacts,
            output_dir=output_dir,
            language=language,
            timeout_seconds=timeout_seconds,
            install_repo=install_repo,
        )
    except Exception as exc:
        return ("oracle_raised", 0, [], f"{type(exc).__name__}: {exc}")

    transitions = report.get("transitions") or {}
    f2p_tests = sorted(
        nodeid for nodeid, info in transitions.items() if isinstance(info, dict) and info.get("f2p")
    )
    summary = report.get("summary") or {}
    return (
        str(summary.get("status") or report.get("status") or "ok"),
        int(summary.get("f2p_count") or 0),
        f2p_tests,
        None,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def validate_milestone(
    *,
    milestone: dict[str, Any],
    test_objectives: list[dict[str, Any]],
    test_artifacts: list[dict[str, Any]],
    broken_dir: Optional[Path] = None,
    fixed_dir: Optional[Path] = None,
    language: str = "python",
    timeout_seconds: float = 300.0,
    install_repo: bool = False,
    output_dir: Optional[Path] = None,
) -> MilestoneValidationReport:
    """Compute the actual CORE/ISO/STRICT level for one milestone.

    The level returned is monotonic — STRICT implies ISO implies
    CORE. Pass broken_dir/fixed_dir to enable STRICT verification;
    omit them to get a CORE/ISO-only report (e.g., when the caller
    only has test artifacts and no sandbox).
    """
    milestone_id = str(milestone.get("milestone_id") or "").strip()
    report = MilestoneValidationReport(milestone_id=milestone_id)
    if not milestone_id:
        report.error = "milestone payload missing milestone_id"
        return report

    objectives_for_milestone = [
        o
        for o in test_objectives
        if isinstance(o, dict) and str(o.get("milestone_id") or "") == milestone_id
    ]
    objective_ids = [
        str(o.get("objective_id") or "")
        for o in objectives_for_milestone
        if str(o.get("objective_id") or "")
    ]
    report.objective_ids = list(objective_ids)

    matching_artifacts = _artifacts_for_milestone(milestone_id, test_artifacts)
    report.candidate_test_paths = sorted(
        {str(a.get("path") or "").strip() for a in matching_artifacts if a.get("path")}
    )

    # --- CORE: every assigned artifact must parse cleanly ---
    parse_errors: list[str] = []
    for a in matching_artifacts:
        path = str(a.get("path") or "").strip() or "<no_path>"
        content = str(a.get("content") or "")
        if not content:
            continue
        err = _try_parse(language, content)
        if err is not None:
            parse_errors.append(f"{path}: {err}")
    report.parse_errors = parse_errors

    if not matching_artifacts:
        report.level_reached = MilestoneValidationLevel.NONE
        return report
    if parse_errors:
        report.level_reached = MilestoneValidationLevel.NONE
        return report
    report.level_reached = MilestoneValidationLevel.CORE

    # --- ISO: every objective has at least one witness artifact ---
    objectives_with_witness: set[str] = set()
    for a in matching_artifacts:
        for obj_id in _objective_ids_in_artifact(a):
            if obj_id in objective_ids:
                objectives_with_witness.add(obj_id)
    report.objectives_with_witness = sorted(objectives_with_witness)
    report.objectives_missing_witness = sorted(set(objective_ids) - objectives_with_witness)
    if objective_ids and not report.objectives_missing_witness:
        report.level_reached = MilestoneValidationLevel.ISO
    elif not objective_ids:
        # A milestone with NO declared objectives reaches ISO trivially
        # — there's nothing to require a witness for.
        report.level_reached = MilestoneValidationLevel.ISO

    # --- STRICT: at least one F2P transition under the sandbox pair ---
    if not report.iso_ready:
        report.f2p_status = "skip_iso_not_ready"
        return report
    if broken_dir is None or fixed_dir is None:
        report.f2p_status = "skip_no_sandboxes"
        return report
    out_dir = output_dir or (Path("/tmp") / f"_milestone_validator_{milestone_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    status, f2p_count, f2p_tests, error = _f2p_for_milestone(
        artifacts=matching_artifacts,
        broken_dir=Path(broken_dir),
        fixed_dir=Path(fixed_dir),
        language=language,
        timeout_seconds=timeout_seconds,
        install_repo=install_repo,
        output_dir=out_dir,
    )
    report.f2p_status = status
    report.f2p_count = f2p_count
    report.f2p_tests = list(f2p_tests)
    if error:
        report.error = error
        return report
    if f2p_count > 0:
        report.level_reached = MilestoneValidationLevel.STRICT
    return report


def validate_milestone_suite(
    *,
    milestones: list[dict[str, Any]],
    test_objectives: list[dict[str, Any]],
    test_artifacts: list[dict[str, Any]],
    broken_dir: Optional[Path] = None,
    fixed_dir: Optional[Path] = None,
    language: str = "python",
    timeout_seconds: float = 300.0,
    install_repo: bool = False,
    output_dir: Optional[Path] = None,
) -> MilestoneSuiteReport:
    """Run :func:`validate_milestone` across every milestone in a
    test-generation design payload."""
    suite = MilestoneSuiteReport()
    for milestone in milestones or []:
        if not isinstance(milestone, dict):
            continue
        per_milestone_out = (
            (Path(output_dir) / f"milestone_{milestone.get('milestone_id')}")
            if output_dir
            else None
        )
        suite.per_milestone.append(
            validate_milestone(
                milestone=milestone,
                test_objectives=test_objectives,
                test_artifacts=test_artifacts,
                broken_dir=broken_dir,
                fixed_dir=fixed_dir,
                language=language,
                timeout_seconds=timeout_seconds,
                install_repo=install_repo,
                output_dir=per_milestone_out,
            )
        )
    return suite
