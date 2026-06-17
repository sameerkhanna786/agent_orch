"""Multi-surrogate F2P oracle for testgen quality scoring (Phase I.4).

The classic F2P oracle (``f2p_oracle.evaluate_f2p_on_sandboxes``)
needs a paired ``broken_dir`` and ``fixed_dir`` — i.e., the caller
must already have a working fix. For real-world TDD workflows the
caller may have only a problem statement and a broken repo. The
e-Otter++ pattern (ICSE 2026, "Targeted Test Generation") closes
this gap: synthesize N candidate fixes via the agent, treat each as
a *surrogate gold patch*, and score the candidate test portfolio
against ALL surrogates. The aggregated score is more robust than
any single-surrogate F2P run because:

    * **Consensus F2P**: tests that flip f2p under EVERY surrogate
      almost certainly capture the bug (high precision).
    * **Union F2P**: tests that flip f2p under AT LEAST ONE
      surrogate are at least relevant (high recall).
    * **Weighted consensus**: per-surrogate votes are weighted by
      the surrogate's own confidence (its broken-sandbox pass count
      relative to the total candidate test count). Surrogates whose
      fix actually exercises more of the suite carry more weight than
      ones that fix only a fraction.
    * **Per-surrogate disagreement** is itself a signal — a test
      that f2p's only on one surrogate may be over-fitting to that
      particular fix shape.

This module is the engine. The ``apex.modes`` surface wires it into
``run_testgen_with_fix(gold_patch=None, surrogate_patcher=...)``.

Generalizes outside benchmarks because it depends only on (repo,
problem_statement, surrogate_patcher) — no benchmark task object.

Phase 4A item 4.7 raised the default from N=4 to N=8 and added
multi-model diversity: surrogates round-robin through a list of
distinct CLI agent backends so the ensemble doesn't suffer from the
same blind spots N times. Per the project directive ("never reduce
model size / power"), the default is 8 — strong consensus signal,
bounded wallclock cost.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


# A surrogate_patcher takes (broken_repo_path, problem_statement,
# surrogate_index) and returns a unified-diff patch string (or None
# when the agent could not produce a fix). The surrogate_index lets
# the caller diversify across runs (different seeds, prompts, etc.).
SurrogatePatcher = Callable[[Path, str, int], Optional[str]]


@dataclass
class SurrogateFixCandidate:
    """One candidate fix produced by the surrogate patcher."""

    index: int
    patch_text: str
    apply_status: str = "ok"  # "ok" | "apply_failed" | "empty"
    apply_error: Optional[str] = None
    f2p_summary: dict[str, Any] = field(default_factory=dict)
    # repo-relative test nodeids that flipped F2P under this surrogate
    f2p_nodeids: list[str] = field(default_factory=list)
    # Phase 4A item 4.7 — model identifier the surrogate was generated
    # with. Empty when the caller didn't supply ``surrogate_models``.
    model: str = ""
    # Per-surrogate confidence weight in [0, 1]. Computed from the
    # surrogate's broken-sandbox pass count over the total candidate
    # test count: a surrogate that exercises more of the suite carries
    # more weight in the weighted consensus. Defaults to 1.0 when the
    # F2P payload doesn't carry pass-count detail (back-compat).
    confidence: float = 1.0


@dataclass
class SurrogateOracleReport:
    """Aggregated multi-surrogate F2P report."""

    n_surrogates_requested: int
    n_surrogates_produced: int
    n_surrogates_applied: int
    n_surrogates_with_any_f2p: int
    consensus_f2p_nodeids: list[str] = field(default_factory=list)
    union_f2p_nodeids: list[str] = field(default_factory=list)
    # Phase 4A item 4.7 — weighted-consensus selection. A test belongs
    # to ``weighted_consensus_f2p_nodeids`` when its weighted vote
    # share (sum of confidences of surrogates that f2p-flipped it,
    # divided by sum of all applied surrogate confidences) clears
    # ``weighted_consensus_threshold`` (default 0.5).
    weighted_consensus_f2p_nodeids: list[str] = field(default_factory=list)
    weighted_consensus_threshold: float = 0.5
    weighted_consensus_scores: dict[str, float] = field(default_factory=dict)
    per_surrogate: list[SurrogateFixCandidate] = field(default_factory=list)
    status: str = "ok"
    error: Optional[str] = None

    @property
    def consensus_f2p_count(self) -> int:
        return len(self.consensus_f2p_nodeids)

    @property
    def union_f2p_count(self) -> int:
        return len(self.union_f2p_nodeids)

    @property
    def weighted_consensus_f2p_count(self) -> int:
        return len(self.weighted_consensus_f2p_nodeids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_surrogates_requested": self.n_surrogates_requested,
            "n_surrogates_produced": self.n_surrogates_produced,
            "n_surrogates_applied": self.n_surrogates_applied,
            "n_surrogates_with_any_f2p": self.n_surrogates_with_any_f2p,
            "consensus_f2p_count": self.consensus_f2p_count,
            "union_f2p_count": self.union_f2p_count,
            "weighted_consensus_f2p_count": self.weighted_consensus_f2p_count,
            "weighted_consensus_threshold": self.weighted_consensus_threshold,
            "weighted_consensus_scores": dict(self.weighted_consensus_scores),
            "consensus_f2p_nodeids": list(self.consensus_f2p_nodeids),
            "union_f2p_nodeids": list(self.union_f2p_nodeids),
            "weighted_consensus_f2p_nodeids": list(self.weighted_consensus_f2p_nodeids),
            "per_surrogate": [
                {
                    "index": c.index,
                    "model": c.model,
                    "apply_status": c.apply_status,
                    "apply_error": c.apply_error,
                    "f2p_summary": dict(c.f2p_summary),
                    "f2p_nodeids": list(c.f2p_nodeids),
                    "confidence": float(c.confidence),
                }
                for c in self.per_surrogate
            ],
            "status": self.status,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clone(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "git",
            "clone",
            "--shared",
            "--no-hardlinks",
            "--quiet",
            str(source),
            str(dest),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        # Fall back to copytree for non-git sources
        shutil.copytree(source, dest)


def _apply_patch_3way(repo_dir: Path, patch_text: str) -> tuple[bool, str]:
    """Apply patch with `git apply` (with 3-way fallback). Returns (ok, error)."""
    if not patch_text or not patch_text.strip():
        return False, "empty patch"
    plain = subprocess.run(
        ["git", "-C", str(repo_dir), "apply", "--whitespace=nowarn", "-"],
        input=patch_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if plain.returncode == 0:
        return True, ""
    threeway = subprocess.run(
        ["git", "-C", str(repo_dir), "apply", "--3way", "--whitespace=nowarn", "-"],
        input=patch_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if threeway.returncode == 0:
        return True, ""
    return False, (
        f"git apply rc={plain.returncode}; 3way rc={threeway.returncode}; "
        f"stderr_tail={(plain.stderr or '').strip()[-300:]}"
    )


def _f2p_nodeids_from_summary(summary: dict[str, Any]) -> list[str]:
    """Extract the per-test F2P nodeids from an F2P payload.

    The F2P oracle's summary may carry per-test detail in two shapes:
    a flat ``f2p_tests`` list, or a nested ``per_test_status`` dict
    keyed by nodeid where transitions appear as 'fail->pass'. Try both.
    """
    raw_list = summary.get("f2p_tests")
    if isinstance(raw_list, list):
        return [str(n) for n in raw_list if n]
    nodeids: list[str] = []
    transitions = summary.get("per_test_transitions") or summary.get("per_test_status")
    if isinstance(transitions, dict):
        for nodeid, transition in transitions.items():
            if str(transition or "").lower() in {"fail->pass", "f2p", "f->p"}:
                nodeids.append(str(nodeid))
    return nodeids


def _f2p_nodeids_from_report(report: dict[str, Any]) -> list[str]:
    """Extract F2P-flipping nodeids from an evaluate_tdd_iteration report.

    The canonical place is the top-level ``transitions`` dict whose
    values are ``{"broken": ..., "fixed": ..., "f2p": bool, "kind": str}``.
    Falls back to ``_f2p_nodeids_from_summary`` for older payload shapes
    so callers stay compatible across releases.
    """
    transitions = report.get("transitions")
    if isinstance(transitions, dict) and transitions:
        return sorted(
            nodeid
            for nodeid, info in transitions.items()
            if isinstance(info, dict) and info.get("f2p")
        )
    return _f2p_nodeids_from_summary(report.get("summary") or {})


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_via_surrogate_oracle(
    *,
    broken_repo_path: str | Path,
    problem_statement: str,
    test_artifacts: list[dict[str, Any]],
    surrogate_patcher: SurrogatePatcher,
    output_dir: str | Path,
    n_surrogates: int = 8,
    language: str = "python",
    install_repo: bool = False,
    timeout_seconds: float = 300.0,
    surrogate_models: Optional[list[str]] = None,
    weighted_consensus_threshold: float = 0.5,
) -> SurrogateOracleReport:
    """Score a candidate test suite via N surrogate-fix F2P runs.

    For each surrogate index ``i`` in ``range(n_surrogates)`` we:
        1. Invoke ``surrogate_patcher(broken_repo, problem_statement, i)``
           to obtain a candidate fix.
        2. Clone the broken repo into ``output_dir / "_surrogates" / "i"``
           and apply the surrogate patch (with `git apply --3way`).
        3. Materialize the candidate test artifacts in BOTH the broken
           sandbox (per-surrogate clone) and the surrogate-fixed sandbox.
        4. Run :func:`apex.evaluation.evaluate_tdd_iteration` to compute
           an F2P payload for that surrogate.

    Then we aggregate:
        * ``consensus_f2p_nodeids``: tests that flipped F2P under EVERY
          successfully applied surrogate (high precision).
        * ``union_f2p_nodeids``: tests that flipped F2P under AT LEAST
          ONE surrogate (high recall).

    Defensive about every failure mode (patcher returned None, patch
    failed to apply, F2P timed out) — surfaces the failure on the
    per-surrogate record without aborting the whole pass.
    """
    broken_repo = Path(broken_repo_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    surrogates_root = output / "_surrogates"
    surrogates_root.mkdir(parents=True, exist_ok=True)

    models = [str(m).strip() for m in (surrogate_models or []) if str(m or "").strip()]
    threshold = float(weighted_consensus_threshold or 0.0)

    report = SurrogateOracleReport(
        n_surrogates_requested=int(n_surrogates),
        n_surrogates_produced=0,
        n_surrogates_applied=0,
        n_surrogates_with_any_f2p=0,
        weighted_consensus_threshold=threshold,
    )
    if not test_artifacts:
        report.status = "no_test_artifacts"
        return report
    if n_surrogates < 1:
        report.status = "no_surrogates_requested"
        return report

    from . import evaluate_tdd_iteration  # avoid import cycle at module load

    successful_candidates: list[SurrogateFixCandidate] = []

    for index in range(int(n_surrogates)):
        # Phase 4A item 4.7: round-robin model assignment so the
        # ensemble doesn't sample the same backend N times.
        model_id = models[index % len(models)] if models else ""
        candidate = SurrogateFixCandidate(index=index, patch_text="", model=model_id)
        report.per_surrogate.append(candidate)
        try:
            patch_text = _invoke_patcher(
                surrogate_patcher,
                broken_repo,
                problem_statement,
                index,
                model_id,
            )
        except Exception as exc:  # pragma: no cover — defensive
            candidate.apply_status = "patcher_raised"
            candidate.apply_error = f"{type(exc).__name__}: {exc}"
            continue
        if not patch_text or not patch_text.strip():
            candidate.apply_status = "empty"
            continue
        candidate.patch_text = patch_text
        report.n_surrogates_produced += 1

        per_surrogate_root = surrogates_root / f"surrogate_{index}"
        broken_dir = per_surrogate_root / "broken"
        fixed_dir = per_surrogate_root / "fixed"
        try:
            _clone(broken_repo, broken_dir)
            _clone(broken_repo, fixed_dir)
        except Exception as exc:  # pragma: no cover — defensive
            candidate.apply_status = "clone_failed"
            candidate.apply_error = f"{type(exc).__name__}: {exc}"
            continue

        ok, err = _apply_patch_3way(fixed_dir, patch_text)
        if not ok:
            candidate.apply_status = "apply_failed"
            candidate.apply_error = err
            continue
        candidate.apply_status = "ok"
        report.n_surrogates_applied += 1

        try:
            f2p_report = evaluate_tdd_iteration(
                broken_dir=broken_dir,
                fixed_dir=fixed_dir,
                test_artifacts=test_artifacts,
                output_dir=per_surrogate_root / "_tdd_report",
                language=language,
                timeout_seconds=timeout_seconds,
                install_repo=install_repo,
            )
        except Exception as exc:  # pragma: no cover — defensive
            candidate.apply_status = "f2p_raised"
            candidate.apply_error = f"{type(exc).__name__}: {exc}"
            continue

        summary = dict(f2p_report.get("summary") or {})
        candidate.f2p_summary = summary
        candidate.f2p_nodeids = _f2p_nodeids_from_report(f2p_report)
        candidate.confidence = _surrogate_confidence(f2p_report)
        if summary.get("any_f2p") or candidate.f2p_nodeids:
            report.n_surrogates_with_any_f2p += 1
            successful_candidates.append(candidate)

    if successful_candidates:
        union: set[str] = set()
        for c in successful_candidates:
            union |= set(c.f2p_nodeids)
        report.union_f2p_nodeids = sorted(union)
        consensus = set(successful_candidates[0].f2p_nodeids)
        for c in successful_candidates[1:]:
            consensus &= set(c.f2p_nodeids)
        report.consensus_f2p_nodeids = sorted(consensus)
        # Weighted consensus: per-test, sum the confidences of
        # surrogates that f2p-flipped that test, divide by the total
        # confidence across all applied surrogates. A test passes the
        # weighted-consensus bar when its share clears ``threshold``.
        applied_candidates = [c for c in report.per_surrogate if c.apply_status == "ok"]
        total_weight = sum(max(0.0, c.confidence) for c in applied_candidates)
        weighted_scores: dict[str, float] = {}
        if total_weight > 0:
            for nodeid in union:
                vote = 0.0
                for c in successful_candidates:
                    if nodeid in set(c.f2p_nodeids):
                        vote += max(0.0, c.confidence)
                weighted_scores[nodeid] = round(vote / total_weight, 4)
        report.weighted_consensus_scores = weighted_scores
        report.weighted_consensus_f2p_nodeids = sorted(
            nodeid for nodeid, score in weighted_scores.items() if score >= threshold
        )
    if report.n_surrogates_produced == 0:
        report.status = "no_surrogates_produced"
    elif report.n_surrogates_applied == 0:
        report.status = "no_surrogates_applied"
    else:
        report.status = "ok"
    return report


def _invoke_patcher(
    patcher: SurrogatePatcher,
    repo: Path,
    problem: str,
    index: int,
    model: str,
) -> Optional[str]:
    """Call the surrogate patcher, supporting both legacy (3-arg) and
    Phase 4A (4-arg ``model``) signatures.

    The ``model`` argument identifies the CLI agent backend the surrogate
    should be generated with (e.g. ``"codex_cli:gpt-5.5"``). Patchers
    that don't yet accept a model parameter are called with the legacy
    3-arg signature transparently — the model assignment then only
    appears in diagnostics.
    """

    try:
        return patcher(repo, problem, index, model)  # type: ignore[call-arg]
    except TypeError:
        return patcher(repo, problem, index)


def _surrogate_confidence(f2p_report: dict[str, Any]) -> float:
    """Per-surrogate confidence in [0, 1].

    Phase 4A item 4.7: weight each surrogate's vote by its independent
    confidence — the surrogate's own broken-sandbox pass count over the
    total candidate test count. Surrogates whose fix actually exercises
    more of the suite carry more weight than those that only fix one
    obscure test.

    Falls back to 1.0 (uniform) when the F2P payload doesn't carry the
    detail. This keeps backwards-compatibility with older payload shapes
    while letting modern F2P reports contribute richer signal.
    """

    summary = dict(f2p_report.get("summary") or {})
    transitions = f2p_report.get("transitions") or {}
    if isinstance(transitions, dict) and transitions:
        total = len(transitions)
        broken_pass = sum(
            1
            for info in transitions.values()
            if isinstance(info, dict)
            and str(info.get("broken") or "").lower() in {"pass", "passed", "ok"}
        )
        if total > 0:
            return round(broken_pass / total, 4)
    pass_count_keys = (
        "broken_pass_count",
        "n_pass_broken",
        "broken_passing",
    )
    total_keys = ("test_count", "n_tests", "total_tests")
    pass_count = next(
        (int(summary.get(k)) for k in pass_count_keys if k in summary),
        None,
    )
    total = next(
        (int(summary.get(k)) for k in total_keys if k in summary),
        None,
    )
    if pass_count is not None and total:
        return round(max(0, pass_count) / max(1, total), 4)
    return 1.0
