"""
Multi-stage patch selection.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ..acceptance import (
    quick_verification_expected_coverage_ratio,
    quick_verification_has_local_full_scope_pass,
    quick_verification_has_scored_expected_suite_pass,
    quick_verification_has_strong_signal,
    quick_verification_signal_score,
    rollout_has_authoritative_acceptance,
    verification_has_explicit_validity_rejection,
)
from ..agents.artifacts import coerce_localization_artifact, coerce_patch_artifact
from ..agents.selector_agent import SelectorAgent
from ..controller_policy import (
    canonical_expected_test_count,
    canonical_expected_test_ids,
    canonical_test_inventory,
    visible_test_edit_protection_enabled,
)
from ..core.config import ApexConfig, SelectionStrategy
from ..core.component_ablation import (
    behavioral_arms_summary,
    clarification_abstain_enabled,
    component_disabled,
)
from ..core.evidence_ledger import build_candidate_evidence_ledger
from ..core.failure_classifier import FailureClass as CoreFailureClass
from ..core.filesystem import copy_tree
from ..core.git_utils import (
    clone_git_repo_with_overlay,
    expand_changed_paths,
    is_git_repo,
)
from ..core.git_utils import (
    list_changed_files as list_git_changed_files,
)
from ..core.pytest_report_utils import (
    VisibleTestEditDisposition,
    analyze_visible_test_edit,
    incomplete_test_files_from_context,
)
from ..core.subprocess_utils import run_process_command
from ..evaluation.contracts import CandidateValidity
from ..evaluation.checkpointing import atomic_write_json
from ..evaluation.run_artifacts import load_rollout_live_states, write_task_live_state
from ..orchestration.clarification import assess_clarification_need
from ..orchestration.verification_amplifier import (
    VerificationAmplifier,
)
from ..rollout.engine import RolloutResult, _quick_verification_payload_from_rollout_status
from ..core.verification_taxonomy import (
    VerificationFailureKind,
    classify_candidate_verification,
)
from .adversarial_review import review_candidate
from .final_acceptance_reviewer import FinalAcceptanceReviewer, PerspectiveReviewer
from .learned_critic import extract_execution_features, load_eg_critic
from ..core.component_ablation import component_optional_enabled
from .process_quality import score_process_quality
from .semantic_clustering import (
    SemanticSignature,
    compute_semantic_signature,
    semantic_similarity,
)
from .verifier import PatchVerifier, PruneResult, TestResult, VerificationResult

if TYPE_CHECKING:
    from ..planning.manager import IssuePlan

logger = logging.getLogger("apex.selection")

# When every rollout is marked success=False, salvage rollouts whose quick
# verification still produced a high observed pass rate (env-conditional skip
# noise or coverage-gap demotion that doesn't reflect actual patch quality).
_ZERO_SUCCESS_FALLBACK_QV_PASS_RATE_FLOOR = 0.85
_SELECTION_GIT_TIMEOUT_RETURN_CODE = 124
_SELECTION_GIT_QUICK_TIMEOUT_SECONDS = 30
_SELECTION_GIT_DEFAULT_TIMEOUT_SECONDS = 300
_SELECTION_GIT_WORKTREE_REMOVE_TIMEOUT_SECONDS = 120


def _run_selection_git(
    args: list[str],
    *,
    cwd: Path,
    timeout: int = _SELECTION_GIT_DEFAULT_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    command = ["git", *args]
    try:
        return run_process_command(command, cwd=cwd, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.output if isinstance(exc.output, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return subprocess.CompletedProcess(
            command,
            _SELECTION_GIT_TIMEOUT_RETURN_CODE,
            stdout,
            (stderr or "") + f"\ngit command timed out after {timeout:.1f}s",
        )
    except OSError as exc:
        return subprocess.CompletedProcess(command, 127, "", str(exc))


def _signature_has_signal(signature: SemanticSignature) -> bool:
    """Decisive-Edge B.3: True iff a semantic signature has *any*
    non-empty component the distance function can score on.

    An "empty" signature (every tuple field empty) means the diff
    contained no detectable Python edits and the non-Python heuristic
    extracted nothing — the selector then falls back to legacy text
    similarity rather than scoring two empty signatures as 0.0
    distance (which would collapse unrelated empty signatures into
    one cluster).
    """
    # ``file_set_normalized`` is intentionally NOT consulted here:
    # a signature whose only content is the touched-file set carries
    # no information that distinguishes two different non-Python diffs
    # touching the same file. In that case the selector falls back to
    # legacy text similarity (the spec'd behaviour for non-Python
    # patches in Decisive-Edge B.3).
    return bool(
        signature.changed_function_names
        or signature.changed_call_sites
        or signature.added_imports
        or signature.removed_imports
        or signature.modified_control_flow
        or signature.modified_data_structures
        or signature.operator_kinds
        or signature.constant_signature
    )


@dataclass
class PatchCluster:
    """Semantically similar candidate patches."""

    cluster_id: int
    patches: list[RolloutResult] = field(default_factory=list)
    signature: str = ""
    payload: str = ""
    verification: Optional[VerificationResult] = None
    cross_validation_score: float = 0.0
    vote_count: int = 0
    critic_score: float = 0.0
    critic_summary: str = ""
    critic_focus_files: list[str] = field(default_factory=list)
    critic_features: dict[str, float] = field(default_factory=dict)
    critic_weight: float = 0.0
    evidence_mode: str = "authoritative"
    verification_authority: str = ""
    public_signal_score: float = 0.0
    backend_anomaly_penalty: float = 0.0
    evidence_ledger_score: float = 0.5
    evidence_ledger: dict[str, Any] = field(default_factory=dict)
    process_quality_score: float = 0.5
    process_quality: dict[str, Any] = field(default_factory=dict)
    adversarial_risk_score: float = 0.0
    adversarial_review: dict[str, Any] = field(default_factory=dict)
    verification_taxonomy: dict[str, Any] = field(default_factory=dict)
    clarification_assessment: dict[str, Any] = field(default_factory=dict)
    # Feature E: perspective-diverse model-critic tiebreaker score (0..1 mean of
    # the per-lens scores) and the raw per-lens map. Defaults are neutral/empty
    # so an unscored cluster contributes nothing to ranking.
    perspective_score: float = 0.0
    perspective_scores: dict[str, float] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.patches)

    @property
    def representative(self) -> RolloutResult:
        return self.patches[0]

    @property
    def verification_score(self) -> float:
        return self.verification.overall_score if self.verification else 0.0

    @property
    def combined_score(self) -> float:
        return float(self.ranking_details()["combined_score"])

    @property
    def accepted(self) -> bool:
        return bool(self.verification and self.verification.accepted)

    def ranking_details(self) -> dict[str, Any]:
        mode = str(self.evidence_mode or "authoritative").strip().lower()
        if mode not in {"authoritative", "weak_public", "structural_only"}:
            mode = "authoritative"

        vote_fraction = self.vote_count / max(self.size, 1)
        size_signal = min(self.size / 3.0, 1.0)
        verification_score = max(0.0, min(float(self.verification_score), 1.0))
        consensus_score = max(0.0, min(float(self.cross_validation_score), 1.0))
        public_signal_score = max(0.0, min(float(self.public_signal_score), 1.0))
        anomaly_penalty = max(0.0, min(float(self.backend_anomaly_penalty), 0.35))
        evidence_score = max(0.0, min(float(self.evidence_ledger_score), 1.0))
        process_score = max(0.0, min(float(self.process_quality_score), 1.0))
        adversarial_penalty = max(0.0, min(float(self.adversarial_risk_score), 1.0)) * 0.22

        if mode == "authoritative":
            base_score = (
                0.57 * verification_score
                + 0.08 * consensus_score
                + 0.14 * size_signal
                + 0.06 * vote_fraction
                + 0.15 * public_signal_score
            )
            critic_floor = 0.0
            critic_cap = 0.35
            ranking_reason = (
                "authoritative_verifier_acceptance"
                if self.accepted
                else "authoritative_verifier_evidence"
            )
        elif mode == "weak_public":
            base_score = (
                0.22 * verification_score
                + 0.16 * consensus_score
                + 0.14 * size_signal
                + 0.06 * vote_fraction
                + 0.42 * public_signal_score
            )
            critic_floor = 0.24
            critic_cap = 0.40
            ranking_reason = "weak_public_signal_plus_critic"
        else:
            base_score = (
                0.12 * verification_score
                + 0.16 * consensus_score
                + 0.18 * size_signal
                + 0.04 * vote_fraction
                + 0.50 * public_signal_score
            )
            critic_floor = 0.30
            critic_cap = 0.45
            ranking_reason = "critic_consensus_without_authoritative_tests"

        base_score = max(0.0, min(base_score, 1.0))
        effective_critic_weight = 0.0
        combined_score = base_score
        if self.critic_score > 0.0 and (self.critic_weight > 0.0 or critic_floor > 0.0):
            effective_critic_weight = min(
                max(max(self.critic_weight, 0.0), critic_floor),
                critic_cap,
            )
            combined_score = ((1.0 - effective_critic_weight) * base_score) + (
                effective_critic_weight * self.critic_score
            )

        evidence_adjustment = 0.08 * (evidence_score - 0.5)
        process_adjustment = 0.06 * (process_score - 0.5)
        combined_score = max(
            0.0,
            min(
                combined_score
                + evidence_adjustment
                + process_adjustment
                - anomaly_penalty
                - adversarial_penalty,
                1.0,
            ),
        )
        return {
            "evidence_mode": mode,
            "verification_authority": str(self.verification_authority or ""),
            "ranking_reason": ranking_reason,
            "base_score": round(base_score, 4),
            "combined_score": round(combined_score, 4),
            "verification_score": round(verification_score, 4),
            "cross_validation_score": round(consensus_score, 4),
            "public_signal_score": round(public_signal_score, 4),
            "backend_anomaly_penalty": round(anomaly_penalty, 4),
            "evidence_ledger_score": round(evidence_score, 4),
            "process_quality_score": round(process_score, 4),
            "adversarial_risk_penalty": round(adversarial_penalty, 4),
            "critic_score": round(max(0.0, min(float(self.critic_score), 1.0)), 4),
            "effective_critic_weight": round(effective_critic_weight, 4),
            "size_signal": round(size_signal, 4),
            "vote_fraction": round(vote_fraction, 4),
        }


def _patch_is_substantive(
    patch_text: str,
    allow_test_only: bool = False,
) -> tuple[bool, str]:
    """Decide whether a unified diff represents real, semantically meaningful work.

    Returns a ``(is_substantive, reason)`` tuple. ``reason`` is a short
    diagnostic string suitable for logging / pruning attribution; when
    ``is_substantive`` is True the reason describes *why* the patch was
    accepted (e.g., ``"non_test_source_change"``), and when False the
    reason names the rejection cause (e.g., ``"whitespace_only"``,
    ``"comment_only"``, ``"docstring_only"``, ``"blank_line_only"``,
    ``"empty_diff"``, ``"test_only_in_strict_mode"``).

    Modes:
        ``allow_test_only=False`` (default, used by the codegen-with-tests
        flow): the patch must touch at least one non-test source file with
        a substantive (non-comment / non-whitespace / non-docstring-only)
        change.

        ``allow_test_only=True`` (testgen-with-fix flow): the patch may
        consist of test-file edits alone, but those edits must still be
        substantive — pure-whitespace / pure-comment / docstring-only
        edits to test files are still rejected.

    Implementation notes:
        Parses the diff with ``unidiff`` when available. ``unidiff`` is a
        hard dependency in apex's environment (pyproject lists it), but
        we keep a defensive line-by-line fallback so the helper still
        produces a verdict if the import is unavailable in some hermetic
        test fixture.
    """

    text = str(patch_text or "")
    if not text.strip():
        return (False, "empty_diff")

    try:
        from unidiff import PatchSet  # type: ignore
        from unidiff.errors import UnidiffParseError  # type: ignore
    except ImportError:  # pragma: no cover - unidiff is in pyproject
        return _patch_is_substantive_fallback(text, allow_test_only=allow_test_only)

    try:
        patch_set = PatchSet(text)
    except (UnidiffParseError, Exception):  # pragma: no cover - defensive
        # If unidiff can't parse the diff, fall back to the line-by-line
        # scanner rather than failing-open (which would let a corrupt
        # diff slip through pruning).
        return _patch_is_substantive_fallback(text, allow_test_only=allow_test_only)

    saw_any_added_or_removed_line = False
    has_substantive_source = False
    has_substantive_test = False
    for patched_file in patch_set:
        rel_path = patched_file.target_file or patched_file.source_file or ""
        # ``unidiff`` prefixes paths with ``a/``/``b/``; strip the prefix
        # so the test-path heuristic sees the actual repo-relative path.
        for prefix in ("a/", "b/"):
            if rel_path.startswith(prefix):
                rel_path = rel_path[len(prefix) :]
                break
        is_test = SelectionCritic._is_test_path(rel_path) if rel_path else False
        is_python = rel_path.endswith(".py")
        for hunk in patched_file:
            removed_substantive: list[str] = []
            added_substantive: list[str] = []
            for line in hunk:
                if not (line.is_added or line.is_removed):
                    continue
                saw_any_added_or_removed_line = True
                content = str(line.value or "")
                if not _diff_line_is_substantive(content, is_python=is_python):
                    continue
                if line.is_added:
                    added_substantive.append(content.strip())
                else:
                    removed_substantive.append(content.strip())
            # Cancel matched (removed, added) pairs whose stripped
            # contents are identical — those are pure whitespace /
            # re-indentation churn. What remains after cancellation is
            # the patch's true semantic delta.
            removed_counter: Counter[str] = Counter(removed_substantive)
            added_counter: Counter[str] = Counter(added_substantive)
            common = removed_counter & added_counter
            removed_counter.subtract(common)
            added_counter.subtract(common)
            net_added = sum(count for count in added_counter.values() if count > 0)
            net_removed = sum(count for count in removed_counter.values() if count > 0)
            if net_added == 0 and net_removed == 0:
                continue
            if is_test:
                has_substantive_test = True
            else:
                has_substantive_source = True

    if not saw_any_added_or_removed_line:
        # ``unidiff`` returned no added/removed lines, which can happen
        # when the diff has malformed hunk headers (``@@`` without line
        # ranges, common in synthetic test fixtures). Try the
        # line-by-line fallback before declaring the diff empty.
        return _patch_is_substantive_fallback(text, allow_test_only=allow_test_only)

    if has_substantive_source:
        return (True, "non_test_source_change")
    if has_substantive_test:
        if allow_test_only:
            return (True, "test_only_substantive_change")
        return (False, "test_only_in_strict_mode")
    return (False, "non_substantive_patch")


def _diff_line_is_substantive(line_value: str, *, is_python: bool) -> bool:
    """Decide whether a single +/- diff line represents real content.

    Strips the line and rejects:
        * empty / whitespace-only changes
        * pure-comment changes (``#`` for Python, ``//``/``/*`` for
          C-family heuristic)
        * docstring opener/closer / pure docstring text lines
    """

    stripped = line_value.strip()
    if not stripped:
        return False
    if is_python:
        if stripped.startswith("#"):
            return False
        # Treat triple-quote docstring openers/closers/body as
        # non-substantive. This is a *heuristic* — a multi-line string
        # used for behaviour (e.g., a SQL literal) will look the same.
        # That's an acceptable cost: docstring-only patches are common
        # in agent failure modes, and the alternative is parsing every
        # diff with a real Python tokenizer (expensive on hot path).
        if stripped.startswith(('"""', "'''")):
            return False
        if stripped.endswith(('"""', "'''")) and len(stripped) <= 6:
            return False
    else:
        if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            return False
        if stripped.startswith("#"):
            return False
    return True


def _patch_is_substantive_fallback(
    patch_text: str,
    *,
    allow_test_only: bool,
) -> tuple[bool, str]:
    """Defensive line-by-line parser used when ``unidiff`` is unavailable.

    Reads the diff one line at a time, tracking the current target file
    via ``+++ b/<path>`` markers. Same substantiality semantics as the
    unidiff path, just without hunk awareness.
    """
    current_target = ""
    saw_any_added_or_removed_line = False
    has_substantive_source = False
    has_substantive_test = False
    # Aggregate per-file substantive added/removed content so we can
    # cancel whitespace-only edit pairs (``-hello`` / ``+hello ``)
    # before deciding substantiality.
    per_file_added: dict[str, list[str]] = {}
    per_file_removed: dict[str, list[str]] = {}
    for raw_line in patch_text.splitlines():
        if raw_line.startswith("+++ "):
            target = raw_line[4:].strip()
            for prefix in ("b/", "a/"):
                if target.startswith(prefix):
                    target = target[len(prefix) :]
                    break
            # ``+++ /dev/null`` indicates a deletion; record an empty path
            # which falls through to the source/test classification.
            current_target = "" if target == "/dev/null" else target
            continue
        if raw_line.startswith(("--- ", "diff ", "index ", "@@", "\\")):
            continue
        if not raw_line:
            continue
        marker = raw_line[0]
        if marker not in {"+", "-"}:
            continue
        saw_any_added_or_removed_line = True
        is_python = current_target.endswith(".py")
        if not _diff_line_is_substantive(raw_line[1:], is_python=is_python):
            continue
        bucket = per_file_added if marker == "+" else per_file_removed
        bucket.setdefault(current_target, []).append(raw_line[1:].strip())
    for path in set(per_file_added) | set(per_file_removed):
        added_counter: Counter[str] = Counter(per_file_added.get(path, []))
        removed_counter: Counter[str] = Counter(per_file_removed.get(path, []))
        common = added_counter & removed_counter
        added_counter.subtract(common)
        removed_counter.subtract(common)
        net_added = sum(count for count in added_counter.values() if count > 0)
        net_removed = sum(count for count in removed_counter.values() if count > 0)
        if net_added == 0 and net_removed == 0:
            continue
        is_test = SelectionCritic._is_test_path(path) if path else False
        if is_test:
            has_substantive_test = True
        else:
            has_substantive_source = True
    if not saw_any_added_or_removed_line:
        return (False, "empty_diff")
    if has_substantive_source:
        return (True, "non_test_source_change")
    if has_substantive_test:
        if allow_test_only:
            return (True, "test_only_substantive_change")
        return (False, "test_only_in_strict_mode")
    return (False, "non_substantive_patch")


class _SynthesisAstError(ValueError):
    """Raised when synthesis cannot AST-parse a candidate's file."""


def _summarize_python_module(source: str) -> dict[str, Any]:
    """Return defined names, imports, and free Load names in a module.

    ``defined_names``: top-level FunctionDef/AsyncFunctionDef/ClassDef
        targets and top-level Assign / AnnAssign target names.
    ``import_from``: list of (module, [name, ...]) tuples from
        ``from X import a, b``. ``module`` is the literal module text.
    ``imported_names``: flattened set of plain ``import X`` names
        plus aliases.
    ``used_names``: set of free ``Name(ctx=Load)`` identifiers that
        appear anywhere in the module.
    """
    tree = ast.parse(source)
    defined_names: set[str] = set()
    import_from: list[tuple[str, list[str]]] = []
    imported_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined_names.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for sub in target.elts:
                        if isinstance(sub, ast.Name):
                            defined_names.add(sub.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined_names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            module_text = node.module or ""
            names = [alias.asname or alias.name for alias in node.names if alias.name != "*"]
            import_from.append((module_text, names))
            imported_names.update(names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name.split(".", 1)[0])
    used_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used_names.add(node.id)
    return {
        "defined_names": defined_names,
        "import_from": import_from,
        "imported_names": imported_names,
        "used_names": used_names,
    }


@dataclass
class CriticAssessment:
    """Structured reranking output for one candidate cluster."""

    score: float
    summary: str = ""
    focus_files: list[str] = field(default_factory=list)
    feature_scores: dict[str, float] = field(default_factory=dict)


def _completion_test_edit_context(
    issue_plan: Optional["IssuePlan"],
) -> tuple[bool, set[str]]:
    if issue_plan is None:
        return False, set()
    protect_visible_tests = visible_test_edit_protection_enabled(issue_plan)
    allocator_features = dict((issue_plan.allocator_features or {}) or {})
    completion_like = bool(
        allocator_features.get("is_completion_task")
        or allocator_features.get("mentions_public_api")
        or issue_plan.test_context.incomplete_source_files
        or issue_plan.test_context.incomplete_test_files
        or protect_visible_tests
    )
    allowed_test_files = set(incomplete_test_files_from_context(issue_plan.test_context))
    return completion_like, allowed_test_files


# Decisive-Edge D.2: literature-prior critic reranker weights. The
# original :meth:`SelectionCritic.assess_cluster` inlined this dict into
# the ``score = sum(...)`` call. Hoisted to a module constant so the
# calibration script (``apex/scripts/calibrate_testgen_ranking.py``)
# can refit it against historical run outcomes and ship the result as
# ``apex/configs/critic_weights_calibrated.json``. The keys here are the
# canonical critic feature names; ``DEFAULT_CRITIC_FEATURE_KEYS``
# enumerates them in a stable order for the calibration / config layer.
DEFAULT_CRITIC_WEIGHTS: dict[str, float] = {
    "issue_alignment": 0.10,
    "risk_coverage": 0.06,
    "localization_alignment": 0.00,
    "consensus_alignment": 0.11,
    "source_change_quality": 0.06,
    "patch_focus": 0.07,
    "test_alignment": 0.09,
    "obligation_coverage": 0.09,
    "hypothesis_alignment": 0.05,
    "task_state_focus_alignment": 0.04,
    "artifact_confidence": 0.05,
    "outcome_signal": 0.14,
    "test_edit_discipline": 0.08,
    "conflict_discipline": 0.06,
}
DEFAULT_CRITIC_FEATURE_KEYS: tuple[str, ...] = tuple(DEFAULT_CRITIC_WEIGHTS.keys())


def _default_critic_weights_calibrated_path() -> Path:
    """Return the on-disk default for the calibrated critic weights JSON.

    Resolves to ``apex/configs/critic_weights_calibrated.json`` next to
    the package. Importable as a function (not a constant) so test
    suites that patch the package layout still see the right path.
    """
    return Path(__file__).resolve().parents[1] / "configs" / "critic_weights_calibrated.json"


def _coerce_critic_weights(payload: Any) -> Optional[dict[str, float]]:
    """Validate and normalize a critic weights payload.

    Accepts either the full schema (``{"weights": {...}, ...}``) or a
    bare ``{key: float, ...}`` dict for back-compat with raw operator
    edits. Unknown keys are dropped silently; missing keys fall back to
    :data:`DEFAULT_CRITIC_WEIGHTS`. Non-finite values are dropped.
    Returns ``None`` if the payload is structurally invalid (e.g. JSON
    array, no usable keys) so the caller can fall back to defaults.
    """
    if not isinstance(payload, dict):
        return None
    raw = payload.get("weights") if "weights" in payload else payload
    if not isinstance(raw, dict):
        return None
    resolved = dict(DEFAULT_CRITIC_WEIGHTS)
    used_any = False
    for key, value in raw.items():
        if key not in DEFAULT_CRITIC_WEIGHTS:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number != number or number in (float("inf"), float("-inf")):
            continue
        resolved[key] = number
        used_any = True
    if not used_any:
        return None
    return resolved


def load_calibrated_critic_weights(
    path: Optional[Path] = None,
) -> Optional[dict[str, float]]:
    """Load calibrated critic weights from disk if available.

    Returns ``None`` when the file is missing, unreadable, or contains
    an unrecognized payload — the caller should then fall back to
    :data:`DEFAULT_CRITIC_WEIGHTS`. Returns the validated weights dict
    when the file is present and well-formed.
    """
    target = Path(path) if path is not None else _default_critic_weights_calibrated_path()
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        import json

        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        logger.warning(
            "Calibrated critic weights at %s could not be parsed; falling back to defaults", target
        )
        return None
    coerced = _coerce_critic_weights(payload)
    if coerced is None:
        logger.warning(
            "Calibrated critic weights at %s have no usable keys; falling back to defaults", target
        )
        return None
    return coerced


class SelectionCritic:
    """Evidence-aware critic that reranks candidates beyond raw verification."""

    def __init__(self, selector: "PatchSelector"):
        self.selector = selector
        # Decisive-Edge D.2: loader for calibrated reranker weights.
        # Falls back to the literature-prior :data:`DEFAULT_CRITIC_WEIGHTS`
        # when no calibrated file is present (the default placeholder
        # ships ``synthetic: true`` defaults equal to the prior, so this
        # is plumbing — behaviour is unchanged until real calibration
        # data overwrites the file).
        override_path = getattr(
            getattr(selector, "config", None),
            "selection",
            None,
        )
        explicit_path = getattr(override_path, "critic_weights_calibrated_path", None)
        loaded = load_calibrated_critic_weights(
            Path(explicit_path) if explicit_path else None,
        )
        if loaded is not None:
            self._weights: dict[str, float] = loaded
            self._weights_source = "calibrated"
        else:
            self._weights = dict(DEFAULT_CRITIC_WEIGHTS)
            self._weights_source = "default"

    def assess_cluster(
        self,
        cluster: PatchCluster,
        *,
        issue_plan: Optional["IssuePlan"],
        all_candidates: list[RolloutResult],
    ) -> CriticAssessment:
        representative = cluster.representative
        changed_files = set(self.selector._candidate_changed_files(representative))
        evidence = self._build_cross_rollout_evidence(all_candidates)
        localization = coerce_localization_artifact(representative.localization_artifact)
        patch_artifact = coerce_patch_artifact(representative.patch_artifact)

        issue_files = set((issue_plan.relevant_files if issue_plan else []) or [])
        risk_files = set((issue_plan.risk_files if issue_plan else []) or [])
        localized_files = set((localization.files if localization else []) or [])
        task_state_context = (issue_plan.task_state_context if issue_plan else {}) or {}
        completion_like, allowed_test_files = _completion_test_edit_context(issue_plan)
        quick_verification = (
            representative.quick_verification
            if isinstance(representative.quick_verification, dict)
            else {}
        )
        obligation_files = {
            str(path)
            for item in list(task_state_context.get("open_obligations") or [])
            if isinstance(item, dict)
            for path in list(item.get("file_paths") or [])
            if path
        }
        obligation_tests = {
            str(test_id)
            for item in list(task_state_context.get("open_obligations") or [])
            if isinstance(item, dict)
            for test_id in list(item.get("test_ids") or [])
            if test_id
        }
        supported_hypothesis_items = [
            item
            for item in list(task_state_context.get("supported_hypotheses") or [])
            if isinstance(item, dict)
        ]
        supported_hypothesis_files = {
            str(path)
            for item in supported_hypothesis_items
            for path in list(item.get("file_paths") or [])
            if path
        }
        graph_focus_files = {
            str(path) for path in list(task_state_context.get("focus_files") or []) if path
        }
        visible_test_targets = set(
            ((issue_plan.test_context.failing_test_ids if issue_plan else []) or [])
            + ((issue_plan.test_context.focus_test_files if issue_plan else []) or [])
        )
        tests_run = set((patch_artifact.tests_run if patch_artifact else []) or [])
        if representative.test_descriptions:
            tests_run.update(representative.test_descriptions)

        feature_scores = {
            "issue_alignment": self._coverage_score(changed_files, issue_files),
            "risk_coverage": self._risk_coverage_score(changed_files, risk_files),
            "localization_alignment": self._localization_alignment_score(
                changed_files,
                localized_files,
            ),
            "consensus_alignment": self._consensus_alignment(
                changed_files,
                localized_files,
                candidate=representative,
                evidence=evidence,
            ),
            "source_change_quality": self._source_change_quality(changed_files),
            "patch_focus": self._patch_focus_score(changed_files),
            "test_alignment": self._test_alignment_score(
                tests_run=tests_run,
                visible_targets=visible_test_targets,
            ),
            "obligation_coverage": self._obligation_coverage_score(
                changed_files=changed_files,
                tests_run=tests_run,
                obligation_files=obligation_files,
                obligation_tests=obligation_tests,
            ),
            "hypothesis_alignment": self._coverage_score(
                changed_files,
                supported_hypothesis_files,
            ),
            "task_state_focus_alignment": self._coverage_score(
                changed_files,
                graph_focus_files,
            ),
            "artifact_confidence": self._artifact_confidence_score(patch_artifact),
            "outcome_signal": self._outcome_signal(cluster),
            "test_edit_discipline": self._test_edit_discipline_score(
                changed_files=changed_files,
                allowed_test_files=allowed_test_files,
                completion_like=completion_like,
                quick_verification=quick_verification,
            ),
            "conflict_discipline": self._conflict_discipline_score(
                changed_files=changed_files,
                supported_hypotheses=supported_hypothesis_items,
                outcome_signal=self._outcome_signal(cluster),
            ),
        }

        # Decisive-Edge D.2: weights live on the critic so calibration
        # can override them via apex/configs/critic_weights_calibrated.json.
        # Missing keys fall back to the literature prior so partial
        # calibrations (e.g. a refit of just consensus_alignment) keep
        # the rest of the score intact.
        score = sum(
            self._weights.get(name, DEFAULT_CRITIC_WEIGHTS[name]) * feature_scores[name]
            for name in DEFAULT_CRITIC_FEATURE_KEYS
        )
        score = max(0.0, min(score, 1.0))

        focus_counter: Counter[str] = Counter()
        for path in issue_files:
            focus_counter[path] += 2
        for path in risk_files:
            focus_counter[path] += 3
        for path in obligation_files:
            focus_counter[path] += 4
        for path in supported_hypothesis_files:
            focus_counter[path] += 3
        for path in graph_focus_files:
            focus_counter[path] += 2
        for path, count in evidence["changed_files"].items():
            focus_counter[path] += count
        ordered_focus = [
            path for path, _ in focus_counter.most_common() if path not in changed_files
        ]
        focus_files = list(dict.fromkeys(ordered_focus + sorted(changed_files) + list(risk_files)))[
            :8
        ]

        summary = self._build_summary(
            changed_files=changed_files,
            feature_scores=feature_scores,
            localized_files=localized_files,
            issue_files=issue_files,
            has_task_state=bool(task_state_context),
        )
        return CriticAssessment(
            score=score,
            summary=summary,
            focus_files=focus_files,
            feature_scores=feature_scores,
        )

    def _build_cross_rollout_evidence(
        self,
        all_candidates: list[RolloutResult],
    ) -> dict[str, Any]:
        evidence = {
            "changed_files": Counter(),
            "localization_files": Counter(),
            "symbols": Counter(),
            "changed_rollouts": defaultdict(set),
            "changed_families": defaultdict(set),
            "localization_families": defaultdict(set),
            "execution_backed_files": Counter(),
        }
        for candidate in all_candidates:
            changed_files = set(self.selector._candidate_changed_files(candidate))
            patch_artifact = coerce_patch_artifact(candidate.patch_artifact)
            if patch_artifact:
                changed_files.update(path for path in patch_artifact.changed_files if path)
            rollout_id = getattr(candidate, "rollout_id", None)
            model_family = self._candidate_model_family(candidate)
            execution_backed = self._candidate_has_execution_signal(candidate)
            for path in changed_files:
                evidence["changed_files"][path] += 1
                if isinstance(rollout_id, int):
                    evidence["changed_rollouts"][path].add(rollout_id)
                if model_family:
                    evidence["changed_families"][path].add(model_family)
                if execution_backed:
                    evidence["execution_backed_files"][path] += 1

            localization = coerce_localization_artifact(candidate.localization_artifact)
            if localization:
                for path in set(localization.files):
                    evidence["localization_files"][path] += 1
                    if model_family:
                        evidence["localization_families"][path].add(model_family)
                for symbol in set(localization.symbols):
                    evidence["symbols"][symbol] += 1
        return evidence

    def _coverage_score(self, changed_files: set[str], targets: set[str]) -> float:
        if not targets:
            return 0.5
        if not changed_files:
            return 0.0
        overlap = len(changed_files.intersection(targets))
        if overlap <= 0:
            return 0.0
        return min(1.0, overlap / max(min(len(changed_files), len(targets)), 1))

    def _localization_alignment_score(
        self,
        changed_files: set[str],
        localized_files: set[str],
    ) -> float:
        if not localized_files:
            return 0.5
        if not changed_files:
            return 0.0
        overlap = len(changed_files.intersection(localized_files))
        if overlap <= 0:
            return 0.0
        precision = overlap / max(len(changed_files), 1)
        recall = overlap / max(len(localized_files), 1)
        if precision <= 0.0 or recall <= 0.0:
            return 0.0
        return max(0.0, min((2.0 * precision * recall) / (precision + recall), 1.0))

    def _risk_coverage_score(self, changed_files: set[str], risk_files: set[str]) -> float:
        if not risk_files:
            return 0.5
        return 1.0 if changed_files.intersection(risk_files) else 0.2

    def _consensus_alignment(
        self,
        changed_files: set[str],
        localized_files: set[str],
        *,
        candidate: RolloutResult,
        evidence: dict[str, Any],
    ) -> float:
        if not changed_files:
            return 0.0
        candidate_family = self._candidate_model_family(candidate)
        candidate_has_execution = self._candidate_has_execution_signal(candidate)
        per_file_scores: list[float] = []
        for path in changed_files:
            support = max(0, int(evidence["changed_files"].get(path, 0) or 0) - 1)
            family_support = set(evidence["changed_families"].get(path) or set())
            if candidate_family:
                family_support.discard(candidate_family)
            execution_support = int(evidence["execution_backed_files"].get(path, 0) or 0)
            if candidate_has_execution:
                execution_support = max(0, execution_support - 1)
            score = (
                (0.35 * min(support / 3.0, 1.0))
                + (0.35 * min(len(family_support) / 2.0, 1.0))
                + (0.30 * min(execution_support / 2.0, 1.0))
            )
            if support > 0 and not family_support and execution_support <= 0:
                score = min(score, 0.25)
            per_file_scores.append(score)
        return sum(per_file_scores) / len(per_file_scores)

    def _conflict_discipline_score(
        self,
        *,
        changed_files: set[str],
        supported_hypotheses: list[dict[str, Any]],
        outcome_signal: float,
    ) -> float:
        if not changed_files:
            return 0.0
        conflicted_files: set[str] = set()
        weak_support_files: set[str] = set()
        for item in supported_hypotheses:
            conflict_score = (
                float(item.get("conflict_score"))
                if isinstance(item.get("conflict_score"), (int, float))
                else 0.0
            )
            contradiction_score = (
                float(item.get("contradiction_score"))
                if isinstance(item.get("contradiction_score"), (int, float))
                else 0.0
            )
            independent_support = (
                float(item.get("independent_support_score"))
                if isinstance(item.get("independent_support_score"), (int, float))
                else 0.0
            )
            paths = {str(path) for path in list(item.get("file_paths") or []) if str(path).strip()}
            if conflict_score >= 0.35 or contradiction_score >= 0.25:
                conflicted_files.update(paths)
            if independent_support < 0.34:
                weak_support_files.update(paths)

        if not conflicted_files and not weak_support_files:
            return 1.0

        overlap_conflicted = len(changed_files.intersection(conflicted_files)) / max(
            len(changed_files), 1
        )
        overlap_weak = len(changed_files.intersection(weak_support_files)) / max(
            len(changed_files), 1
        )
        score = 1.0 - (0.7 * overlap_conflicted) - (0.3 * overlap_weak)
        if outcome_signal >= 0.8:
            score = min(1.0, score + 0.2)
        elif outcome_signal <= 0.35 and overlap_conflicted > 0.0:
            score -= 0.1
        return max(0.0, min(score, 1.0))

    def _source_change_quality(self, changed_files: set[str]) -> float:
        if not changed_files:
            return 0.0
        return 1.0 if any(not self._is_test_path(path) for path in changed_files) else 0.1

    def _patch_focus_score(self, changed_files: set[str]) -> float:
        count = len(changed_files)
        if count == 0:
            return 0.0
        if count <= 4:
            return 1.0
        if count <= 6:
            return 0.7
        return 0.35

    def _test_edit_discipline_score(
        self,
        *,
        changed_files: set[str],
        allowed_test_files: set[str],
        completion_like: bool,
        quick_verification: dict[str, Any],
    ) -> float:
        test_changes = {path for path in changed_files if self._is_test_path(path)}
        if not test_changes:
            return 1.0

        source_changes = {path for path in changed_files if not self._is_test_path(path)}
        unexpected_test_changes = {path for path in test_changes if path not in allowed_test_files}
        collection_error_failures = sum(
            1
            for test_id in list(quick_verification.get("failed_tests") or [])
            if self._looks_like_collection_error_test_id(str(test_id))
        )
        pass_rate = quick_verification_signal_score(quick_verification)

        if test_changes and test_changes.issubset(allowed_test_files):
            score = 0.85 if pass_rate is not None and pass_rate >= 0.999 else 0.7
        elif not source_changes:
            score = 0.2
        else:
            score = 0.7

        if unexpected_test_changes:
            score -= 0.2 if source_changes else 0.35
        if pass_rate is not None and completion_like and pass_rate < 0.999:
            score -= 0.15
        if collection_error_failures:
            score -= min(0.4, 0.18 * collection_error_failures)

        return max(0.0, min(score, 1.0))

    def _test_alignment_score(
        self,
        *,
        tests_run: set[str],
        visible_targets: set[str],
    ) -> float:
        if not visible_targets:
            return 0.7 if tests_run else 0.5
        if not tests_run:
            return 0.2
        overlap = len(tests_run.intersection(visible_targets))
        if overlap <= 0:
            return 0.25
        return min(1.0, overlap / max(min(len(tests_run), len(visible_targets)), 1))

    def _artifact_confidence_score(self, patch_artifact: Any) -> float:
        confidence = getattr(patch_artifact, "confidence", None)
        if isinstance(confidence, (float, int)):
            return max(0.0, min(float(confidence), 1.0))
        return 0.5

    def _obligation_coverage_score(
        self,
        *,
        changed_files: set[str],
        tests_run: set[str],
        obligation_files: set[str],
        obligation_tests: set[str],
    ) -> float:
        scores: list[float] = []
        if obligation_files:
            scores.append(self._coverage_score(changed_files, obligation_files))
        if obligation_tests:
            scores.append(
                self._test_alignment_score(
                    tests_run=tests_run,
                    visible_targets=obligation_tests,
                )
            )
        if not scores:
            return 0.5
        return sum(scores) / len(scores)

    def _outcome_signal(self, cluster: PatchCluster) -> float:
        verification = cluster.verification
        if verification is None or verification.test_result is None:
            return 0.5
        result = verification.test_result
        signals: list[float] = []
        if result.regression_passes is not None:
            signals.append(1.0 if result.regression_passes else 0.0)
        if result.reproduction_passes is not None:
            signals.append(1.0 if result.reproduction_passes else 0.0)
        if result.pass_rate is not None:
            signals.append(max(0.0, min(float(result.pass_rate), 1.0)))
        return sum(signals) / len(signals) if signals else 0.5

    def _build_summary(
        self,
        *,
        changed_files: set[str],
        feature_scores: dict[str, float],
        localized_files: set[str],
        issue_files: set[str],
        has_task_state: bool,
    ) -> str:
        positives: list[str] = []
        negatives: list[str] = []
        if (
            feature_scores["consensus_alignment"] >= 0.65
            and feature_scores.get("conflict_discipline", 0.0) >= 0.6
        ):
            positives.append(
                "Independent, execution-backed evidence converges on the edited files."
            )
        elif feature_scores["consensus_alignment"] >= 0.65:
            positives.append("Cross-rollout evidence converges on the edited files.")
        if feature_scores["risk_coverage"] >= 0.75:
            positives.append("The patch covers the highest-risk files.")
        if has_task_state and feature_scores["obligation_coverage"] >= 0.7:
            positives.append("The patch covers the highest-pressure unresolved obligations.")
        if feature_scores["issue_alignment"] < 0.3 and issue_files:
            negatives.append("Edited files have weak overlap with the planned issue focus.")
        if feature_scores["source_change_quality"] < 0.3:
            negatives.append("The patch is too test-heavy and lacks enough source edits.")
        if has_task_state and feature_scores["hypothesis_alignment"] < 0.25:
            negatives.append(
                "The patch has weak alignment with the strongest supported hypotheses."
            )
        if feature_scores["outcome_signal"] < 0.35:
            negatives.append("Artifact and verification signals still suggest unresolved behavior.")
        if feature_scores.get("conflict_discipline", 1.0) < 0.45:
            negatives.append(
                "Most agreement is still weakly corroborated or contested across rollouts."
            )
        if feature_scores["test_edit_discipline"] < 0.45:
            negatives.append(
                "The patch edits visible tests without stable confirmation and still shows collection/import breakage."
            )
        if not positives and changed_files:
            positives.append(
                "The patch changes concrete implementation files rather than only metadata."
            )
        parts = positives[:2] + negatives[:2]
        return " ".join(parts) if parts else "Candidate evidence is mixed."

    @staticmethod
    def _candidate_model_family(candidate: RolloutResult) -> str:
        if bool(getattr(candidate, "is_synthetic", False)):
            return "synthetic"
        token = str(getattr(candidate, "llm_model", "") or "").strip().lower()
        if not token:
            return ""
        if token.startswith("openai/") or "gpt" in token or "codex" in token:
            return "openai"
        if token.startswith("anthropic/") or "claude" in token or token == "opus":
            return "anthropic"
        if token.startswith("gemini") or token.startswith("google/"):
            return "google"
        if token.startswith("meta/") or "llama" in token or "avocado" in token:
            return "meta"
        return token.split("/", 1)[0].split("-", 1)[0]

    @staticmethod
    def _candidate_has_execution_signal(candidate: RolloutResult) -> bool:
        if rollout_has_authoritative_acceptance(candidate):
            return True
        verification = candidate.verification if isinstance(candidate.verification, dict) else {}
        if bool(verification.get("accepted")):
            return True
        if (
            isinstance(verification.get("overall_score"), (int, float))
            and float(verification.get("overall_score")) >= 0.85
        ):
            return True
        # Phase 2 10.P: tighten quick-verify gate to require full-scope
        # execution evidence. The previous lenient setting let in-rollout
        # claims of partial-scope passes promote weak candidates into
        # selection across all benchmarks.
        return quick_verification_has_strong_signal(
            candidate.quick_verification if isinstance(candidate.quick_verification, dict) else {},
            require_full_scope=True,
        )

    @staticmethod
    def _is_test_path(path: str) -> bool:
        parts = {part.lower() for part in Path(path).parts}
        return "tests" in parts or "test" in parts or Path(path).name.startswith("test_")

    @staticmethod
    def _is_pytest_collected_test_path(path: str) -> bool:
        """Stricter than _is_test_path: only files pytest will collect.

        Pytest discovers tests matching ``test_*.py`` or ``*_test.py`` (its
        default pattern). The visible-test-edit policy must use *this*
        check, not the broad ``_is_test_path``, otherwise it punishes
        agents for creating scratch / fixture files in ``tests/`` that
        pytest never imports — which costs real solves on benchmarks
        like commit0/bitstring (834/834 tests passed, but the rollout
        was pruned for creating ``tests/temp_bitstring_unit_testing_file``
        with no extension).
        """

        name = Path(path).name
        if not name.endswith(".py"):
            return False
        if name.startswith("test_") or name.endswith("_test.py"):
            return True
        return False

    @staticmethod
    def _looks_like_collection_error_test_id(test_id: str) -> bool:
        normalized = test_id.strip()
        return bool(normalized) and normalized.endswith(".py") and "::" not in normalized


class PatchSelector:
    """Prune, deduplicate, verify, and select the best rollout patch."""

    def __init__(
        self,
        config: ApexConfig,
        repo_path: str,
        verifier: Optional[PatchVerifier] = None,
    ):
        self.config = config
        self.repo_path = repo_path
        if verifier is None:
            primary_llm = self.config.llm_configs[0] if self.config.llm_configs else None
            runtime_env_overrides = dict(getattr(primary_llm, "cli_env_overrides", {}) or {})
            try:
                verifier = PatchVerifier(
                    repo_path,
                    timeout=self.config.selection.verification_timeout_seconds,
                    full_test_timeout=self.config.selection.full_test_timeout_seconds,
                    custom_test_timeout=self.config.selection.custom_test_timeout_seconds,
                    runtime_env_overrides=runtime_env_overrides,
                    verification_helper_files=list(self.config.selection.verification_helper_files),
                )
            except TypeError as exc:
                if "runtime_env_overrides" not in str(
                    exc
                ) and "verification_helper_files" not in str(exc):
                    raise
                helper_fallback_succeeded = False
                if "verification_helper_files" in str(exc):
                    try:
                        verifier = PatchVerifier(
                            repo_path,
                            timeout=self.config.selection.verification_timeout_seconds,
                            full_test_timeout=self.config.selection.full_test_timeout_seconds,
                            custom_test_timeout=self.config.selection.custom_test_timeout_seconds,
                            runtime_env_overrides=runtime_env_overrides,
                        )
                    except TypeError as inner:
                        if "runtime_env_overrides" not in str(inner):
                            raise
                    else:
                        helper_fallback_succeeded = True
                if not helper_fallback_succeeded:
                    try:
                        verifier = PatchVerifier(
                            repo_path,
                            timeout=self.config.selection.verification_timeout_seconds,
                            full_test_timeout=self.config.selection.full_test_timeout_seconds,
                            custom_test_timeout=self.config.selection.custom_test_timeout_seconds,
                        )
                    except TypeError as inner:
                        if "timeout" not in str(inner) and "full_test_timeout" not in str(inner):
                            raise
                        verifier = PatchVerifier(repo_path)
        self.verifier = verifier
        # Propagate the sandbox-disabled ablation knob if the verifier
        # accepts it (the test suites inject thin stand-ins that may not).
        if hasattr(self.verifier, "cross_validation_sandbox_disabled"):
            self.verifier.cross_validation_sandbox_disabled = (
                self.config.selection.cross_validation_sandbox_disabled
            )
        self._prune_results: dict[int, Any] = {}
        self._ephemeral_worktrees: list[Path] = []
        self._ephemeral_workspace_modes: dict[Path, str] = {}
        self._baseline_text_cache: dict[tuple[str, str], Optional[str]] = {}
        # APEX Decisive-Edge A.2: Verification Amplifier hookpoint.
        # Off by default (preserves legacy behavior + cost). The
        # orchestrator (or a test) sets this when it wants tied
        # selectable clusters to be broken with on-the-fly
        # discriminating tests instead of relying on critic / AST
        # heuristics. See apex.orchestration.verification_amplifier.
        self.verification_amplifier: Optional[VerificationAmplifier] = None
        # WS3C: optional fresh-context LLM final-acceptance reviewer (injected by
        # the orchestrator when enabled; None == off, the default).
        self.final_acceptance_reviewer: Optional[FinalAcceptanceReviewer] = None
        # Feature E: optional perspective-diverse model critic. When set (by the
        # orchestrator / a test), it acts as a LOW-PRIORITY tiebreaker among
        # execution-verified clusters in deterministic ranking. None == off.
        self.perspective_reviewer: Optional[PerspectiveReviewer] = None
        # Issue description captured for the duration of select_best_patch so the
        # deterministic-ranking chokepoint can score perspectives without
        # threading the description through every internal call signature.
        self._perspective_issue_description: str = ""
        # WS2C: lazily-loaded execution-grounded learned critic (tie-break only).
        self._eg_critic: Any = None
        self._eg_critic_loaded: bool = False
        # Confidence threshold below which an amplifier verdict is
        # NOT trusted and the selector falls back to existing logic.
        self.verification_amplifier_confidence_threshold: float = 0.6

    @staticmethod
    def _candidate_component_ablation(candidate: RolloutResult) -> dict[str, Any]:
        metadata = getattr(candidate, "search_metadata", None) if candidate is not None else None
        if not isinstance(metadata, dict):
            return {}
        assignment = metadata.get("component_ablation")
        return dict(assignment) if isinstance(assignment, dict) else {}

    @classmethod
    def _cluster_component_ablation(cls, cluster: PatchCluster) -> dict[str, Any]:
        return cls._candidate_component_ablation(cluster.representative)

    @classmethod
    def _component_disabled_for_candidate(
        cls,
        candidate: RolloutResult,
        component: str,
    ) -> bool:
        return component_disabled(cls._candidate_component_ablation(candidate), component)

    @classmethod
    def _component_disabled_for_clusters(
        cls,
        clusters: list[PatchCluster],
        component: str,
    ) -> bool:
        assignments = [cls._cluster_component_ablation(cluster) for cluster in clusters]
        active = [assignment for assignment in assignments if bool(assignment.get("enabled"))]
        return bool(active) and all(
            component_disabled(assignment, component) for assignment in active
        )

    def _selection_status_output_dir(self) -> Optional[Path]:
        output_dir = str(getattr(self.config, "output_dir", "") or "").strip()
        if not output_dir:
            return None
        return Path(output_dir)

    def _write_task_selection_state(
        self,
        stage: str,
        *,
        clear_rollout_state: bool = False,
        extra_fields: Optional[dict[str, Any]] = None,
    ) -> None:
        output_dir = self._selection_status_output_dir()
        if output_dir is None:
            return
        payload: dict[str, Any] = {
            "phase": "selection",
            "status": "active",
            "process_pid": os.getpid(),
            "last_progress_at": time.time(),
            "stage": stage,
            "current_stage": stage,
        }
        if clear_rollout_state:
            payload["_clear_keys"] = [
                "active_rollout_ids",
                "active_rollout_count",
                "completed_rollout_count",
                "error_rollout_count",
                "terminal_rollout_count",
                "total_rollout_count",
                "current_rollout_id",
                "progress_timeout_seconds",
                "hard_timeout_seconds",
                "model",
                "success",
                "final_tests_passed",
            ]
        if isinstance(extra_fields, dict):
            payload.update(extra_fields)
        try:
            write_task_live_state(output_dir, payload)
        except Exception as exc:
            logger.debug("Failed to persist selection live state in %s: %s", output_dir, exc)

    def _candidate_handoff_artifact_dir(self, candidate: RolloutResult) -> Optional[Path]:
        output_dir = self._selection_status_output_dir()
        if output_dir is None:
            return None
        return output_dir / "_candidate_artifacts" / f"rollout_{candidate.rollout_id}"

    @staticmethod
    def _load_json_artifact(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _persist_reconciled_candidate_handoff_artifacts(
        self,
        candidate: RolloutResult,
    ) -> None:
        artifact_dir = self._candidate_handoff_artifact_dir(candidate)
        if artifact_dir is None or not artifact_dir.is_dir():
            return
        try:
            if isinstance(candidate.quick_verification, dict):
                atomic_write_json(
                    artifact_dir / "quick_verification.json",
                    dict(candidate.quick_verification),
                )
            validity = getattr(candidate, "validity", None)
            if isinstance(validity, CandidateValidity):
                atomic_write_json(artifact_dir / "validity.json", validity.as_dict())
        except OSError as exc:
            logger.debug(
                "Failed to persist reconciled candidate handoff artifacts for rollout %s: %s",
                getattr(candidate, "rollout_id", "<unknown>"),
                exc,
            )

    @staticmethod
    def _quick_verification_artifact_is_stronger(
        existing: Optional[dict[str, Any]],
        artifact: Optional[dict[str, Any]],
    ) -> bool:
        if not isinstance(artifact, dict) or not artifact:
            return False
        if not isinstance(existing, dict) or not existing:
            return True

        artifact_full_pass = quick_verification_has_local_full_scope_pass(artifact)
        existing_full_pass = quick_verification_has_local_full_scope_pass(existing)
        if artifact_full_pass != existing_full_pass:
            return artifact_full_pass

        artifact_score = quick_verification_signal_score(artifact)
        existing_score = quick_verification_signal_score(existing)
        if isinstance(artifact_score, (int, float)) and not isinstance(
            existing_score,
            (int, float),
        ):
            return True
        if (
            isinstance(artifact_score, (int, float))
            and isinstance(
                existing_score,
                (int, float),
            )
            and float(artifact_score) > float(existing_score)
        ):
            return True

        artifact_ratio = quick_verification_expected_coverage_ratio(artifact)
        existing_ratio = quick_verification_expected_coverage_ratio(existing)
        if isinstance(artifact_ratio, (int, float)) and not isinstance(
            existing_ratio,
            (int, float),
        ):
            return True
        if (
            isinstance(artifact_ratio, (int, float))
            and isinstance(
                existing_ratio,
                (int, float),
            )
            and float(artifact_ratio) > float(existing_ratio)
        ):
            return True

        return artifact_full_pass and len(artifact) > len(existing)

    @staticmethod
    def _candidate_validity_from_artifact(
        payload: Any,
    ) -> Optional[CandidateValidity]:
        if not isinstance(payload, dict) or not payload:
            return None

        def _as_bool(key: str, default: bool = False) -> bool:
            value = payload.get(key)
            return bool(default if value is None else value)

        missing_expected_raw = payload.get("missing_expected_test_count")
        try:
            missing_expected = max(0, int(missing_expected_raw or 0))
        except (TypeError, ValueError):
            missing_expected = 0
        reasons = [
            str(reason)
            for reason in list(payload.get("reasons") or [])
            if str(reason or "").strip()
        ]
        expected_coverage_preserved = payload.get("expected_coverage_preserved")
        if not isinstance(expected_coverage_preserved, bool):
            expected_coverage_preserved = None
        quality_gate_passed = payload.get("quality_gate_passed")
        if not isinstance(quality_gate_passed, bool):
            quality_gate_passed = None
        return CandidateValidity(
            has_patch=_as_bool("has_patch"),
            worktree_materialized=_as_bool("worktree_materialized"),
            expected_coverage_preserved=expected_coverage_preserved,
            missing_expected_test_count=missing_expected,
            protected_tests_unchanged=_as_bool("protected_tests_unchanged", True),
            collection_critical_files_unchanged=_as_bool(
                "collection_critical_files_unchanged",
                True,
            ),
            quick_verification_passed=_as_bool("quick_verification_passed"),
            quality_gate_passed=quality_gate_passed,
            backend_protocol_error=_as_bool("backend_protocol_error"),
            coverage_collapse_terminal=_as_bool("coverage_collapse_terminal"),
            provenance_violation=_as_bool("provenance_violation"),
            reasons=reasons,
        )

    def _live_state_quick_verification_artifact(
        self,
        candidate: RolloutResult,
    ) -> dict[str, Any]:
        output_dir = self._selection_status_output_dir()
        if output_dir is None:
            return {}
        try:
            rollout_states = load_rollout_live_states(output_dir)
        except Exception as exc:  # noqa: BLE001 - diagnostics must not break selection
            logger.debug("Failed to load rollout live states for candidate hydration: %s", exc)
            return {}

        def _as_int(value: Any) -> Optional[int]:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        rollout_id = _as_int(getattr(candidate, "rollout_id", None))
        if rollout_id is None:
            return {}
        state = next(
            (
                payload
                for payload in rollout_states
                if _as_int(payload.get("rollout_id")) == rollout_id
            ),
            None,
        )
        if not isinstance(state, dict):
            return {}
        return _quick_verification_payload_from_rollout_status(state)

    def _hydrate_candidate_handoff_artifacts(self, candidate: RolloutResult) -> None:
        artifact_dir = self._candidate_handoff_artifact_dir(candidate)
        hydrated_fields: list[str] = []
        if artifact_dir is not None and artifact_dir.is_dir():
            patch_path = artifact_dir / "patch.diff"
            if not str(candidate.patch or "").strip() and patch_path.is_file():
                try:
                    patch_text = patch_path.read_text(encoding="utf-8")
                except OSError:
                    patch_text = ""
                if patch_text.strip():
                    candidate.patch = patch_text
                    hydrated_fields.append("patch")

            changed_files_payload = self._load_json_artifact(artifact_dir / "changed_files.json")
            if isinstance(changed_files_payload, list):
                changed_files = [
                    str(path).strip() for path in changed_files_payload if str(path or "").strip()
                ]
                if changed_files and (
                    not list(candidate.changed_files or [])
                    or len(changed_files) > len(candidate.changed_files)
                ):
                    candidate.changed_files = changed_files
                    hydrated_fields.append("changed_files")

            quick_verification_payload = self._load_json_artifact(
                artifact_dir / "quick_verification.json"
            )
            existing_quick_verification = (
                candidate.quick_verification
                if isinstance(candidate.quick_verification, dict)
                else {}
            )
            if self._quick_verification_artifact_is_stronger(
                existing_quick_verification,
                quick_verification_payload,
            ):
                candidate.quick_verification = dict(quick_verification_payload)
                hydrated_fields.append("quick_verification")

            if candidate.validity is None:
                validity_payload = self._load_json_artifact(artifact_dir / "validity.json")
                validity = self._candidate_validity_from_artifact(validity_payload)
                if validity is not None:
                    candidate.validity = validity
                    hydrated_fields.append("validity")

        live_state_quick_verification = self._live_state_quick_verification_artifact(candidate)
        existing_quick_verification = (
            candidate.quick_verification if isinstance(candidate.quick_verification, dict) else {}
        )
        if self._quick_verification_artifact_is_stronger(
            existing_quick_verification,
            live_state_quick_verification,
        ):
            candidate.quick_verification = dict(live_state_quick_verification)
            hydrated_fields.append("quick_verification_live_state")

        if not hydrated_fields:
            return
        metadata = (
            dict(candidate.search_metadata) if isinstance(candidate.search_metadata, dict) else {}
        )
        metadata["candidate_handoff_artifacts_hydrated"] = True
        metadata["candidate_handoff_hydrated_fields"] = sorted(set(hydrated_fields))
        candidate.search_metadata = metadata

    def _selection_count_fields(
        self,
        *,
        candidate_count: int,
        cluster_count: int,
        verified_cluster_count: Optional[int] = None,
        selectable_cluster_count: Optional[int] = None,
        selector_pool_count: Optional[int] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "selection_candidate_count": candidate_count,
            "selection_cluster_count": cluster_count,
        }
        if verified_cluster_count is not None:
            payload["selection_verified_cluster_count"] = verified_cluster_count
        if selectable_cluster_count is not None:
            payload["selection_selectable_cluster_count"] = selectable_cluster_count
        if selector_pool_count is not None:
            payload["selection_selector_pool_count"] = selector_pool_count
        return payload

    def _is_standalone_anchor_candidate(self, candidate: RolloutResult) -> bool:
        metadata = getattr(candidate, "search_metadata", None)
        if not isinstance(metadata, dict):
            return False
        value = metadata.get("standalone_agent_anchor")
        if isinstance(value, bool):
            return value
        return str(value or "").strip().lower() in {"1", "true", "yes", "on"}

    def _accepted_from_payload(self, candidate: RolloutResult) -> bool:
        verification = candidate.verification if isinstance(candidate.verification, dict) else {}
        quick = (
            candidate.quick_verification if isinstance(candidate.quick_verification, dict) else {}
        )
        if self._candidate_has_explicit_hard_validity_rejection(candidate):
            return False
        if verification_has_explicit_validity_rejection(verification):
            return False
        return bool(
            candidate.internally_accepted
            or verification.get("accepted")
            or verification.get("internally_accepted")
            or candidate.officially_accepted is True
            or verification.get("officially_accepted") is True
            or quick_verification_has_strong_signal(quick, require_full_scope=True)
        )

    def _anchor_strength(self, candidate: RolloutResult) -> tuple[Any, ...]:
        verification = candidate.verification if isinstance(candidate.verification, dict) else {}
        test_result = verification.get("test_result")
        if not isinstance(test_result, dict):
            test_result = {}
        prune_result = verification.get("prune_result")
        if not isinstance(prune_result, dict):
            prune_result = {}
        quick = (
            candidate.quick_verification if isinstance(candidate.quick_verification, dict) else {}
        )
        cross_scores = verification.get("cross_validation_scores")
        cross_values = [
            float(item) for item in list(cross_scores or []) if isinstance(item, (int, float))
        ]
        cross_score = sum(cross_values) / len(cross_values) if cross_values else 0.0
        pass_rate = test_result.get("pass_rate")
        if not isinstance(pass_rate, (int, float)):
            pass_rate = quick.get("pass_rate")
        if not isinstance(pass_rate, (int, float)):
            pass_rate = 0.0
        missing_expected = test_result.get("missing_expected_test_count")
        if not isinstance(missing_expected, (int, float)):
            missing_expected = quick.get("missing_expected_test_count")
        if not isinstance(missing_expected, (int, float)):
            missing_expected = 0
        regressed = prune_result.get("regressed_tests")
        regressed_count = len(regressed) if isinstance(regressed, list) else 0
        expected_preserved = test_result.get("expected_coverage_preserved")
        if expected_preserved is None:
            expected_preserved = True
        return (
            1
            if candidate.officially_accepted is True
            or verification.get("officially_accepted") is True
            else 0,
            1 if self._accepted_from_payload(candidate) else 0,
            1 if expected_preserved is not False else 0,
            round(float(pass_rate or 0.0), 6),
            -int(missing_expected or 0),
            round(float(verification.get("overall_score") or 0.0), 6),
            round(float(cross_score or 0.0), 6),
            -int(regressed_count),
        )

    def _ensure_anchor_verification(
        self,
        candidate: RolloutResult,
        test_command: Optional[str],
        *,
        baseline_result: Optional[Any],
        issue_plan: Optional["IssuePlan"],
    ) -> None:
        if isinstance(candidate.verification, dict) and "accepted" in candidate.verification:
            return
        verification = self._verify_candidate_or_confirmed_handoff(
            candidate,
            test_command,
            baseline_result=baseline_result,
            issue_plan=issue_plan,
        )
        self._reconcile_verification_with_candidate_handoff(candidate, verification)
        self._reconcile_candidate_validity_with_verification(candidate, verification)
        verification.accepted = self._verification_meets_acceptance_bar(
            candidate,
            verification,
            test_command,
            issue_plan=issue_plan,
        )
        candidate.verification = verification.to_dict()
        candidate.internally_accepted = bool(verification.accepted)

    def _maybe_preserve_standalone_anchor(
        self,
        rollout_results: list[RolloutResult],
        selected: Optional[RolloutResult],
        test_command: Optional[str],
        *,
        baseline_result: Optional[Any],
        issue_plan: Optional["IssuePlan"],
    ) -> Optional[RolloutResult]:
        if not bool(getattr(self.config.selection, "preserve_standalone_anchor", True)):
            return selected
        anchors = [
            candidate
            for candidate in rollout_results
            if self._is_standalone_anchor_candidate(candidate) and bool(candidate.patch)
        ]
        if not anchors:
            return selected
        for anchor in anchors:
            try:
                self._ensure_anchor_verification(
                    anchor,
                    test_command,
                    baseline_result=baseline_result,
                    issue_plan=issue_plan,
                )
            except Exception as exc:  # noqa: BLE001 - selection should not crash on guard audit
                anchor.selection_diagnostics = {
                    **dict(anchor.selection_diagnostics or {}),
                    "standalone_anchor_guard": {
                        "status": "verification_failed",
                        "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                    },
                }
        anchors = [
            anchor
            for anchor in anchors
            if self._accepted_from_payload(anchor)
            and self._candidate_validity_allows_submission(anchor, issue_plan=issue_plan)
            and not self._candidate_test_edit_policy_reason(
                anchor,
                issue_plan=issue_plan,
                verification=None,
            )
        ]
        if not anchors:
            return selected
        anchor = max(anchors, key=self._anchor_strength)
        anchor_strength = self._anchor_strength(anchor)
        selected_strength = self._anchor_strength(selected) if selected is not None else None
        if selected is not None and selected is anchor:
            anchor.selection_diagnostics = {
                **dict(anchor.selection_diagnostics or {}),
                "standalone_anchor_guard": {
                    "status": "selected_anchor",
                    "anchor_strength": list(anchor_strength),
                },
            }
            return selected
        if (
            selected is not None
            and selected_strength is not None
            and selected_strength > anchor_strength
        ):
            selected.selection_diagnostics = {
                **dict(selected.selection_diagnostics or {}),
                "standalone_anchor_guard": {
                    "status": "overrode_anchor_with_stronger_evidence",
                    "anchor_rollout_id": anchor.rollout_id,
                    "anchor_strength": list(anchor_strength),
                    "selected_strength": list(selected_strength),
                },
            }
            return selected
        anchor.selection_diagnostics = {
            **dict(anchor.selection_diagnostics or {}),
            "standalone_anchor_guard": {
                "status": "preserved_anchor_on_tie_or_weaker_evidence",
                "previous_selected_rollout_id": (
                    selected.rollout_id if selected is not None else None
                ),
                "anchor_strength": list(anchor_strength),
                "selected_strength": list(selected_strength)
                if selected_strength is not None
                else None,
            },
        }
        verification_payload = dict(anchor.verification or {})
        verification_payload.update(
            {
                "accepted": True,
                "selected_for_submission": True,
                "internally_accepted": True,
                "selection_authority": "standalone_anchor_guard",
            }
        )
        anchor.verification = verification_payload
        anchor.selected_for_submission = True
        anchor.internally_accepted = True
        anchor.salvaged_for_external_scoring = False
        return anchor

    def _repo_can_use_git_snapshot_clone(self, repo_path: Path) -> bool:
        return is_git_repo(repo_path)

    def select_best_patch(
        self,
        rollout_results: list[RolloutResult],
        issue_description: str,
        test_command: Optional[str] = None,
        baseline_result: Optional[Any] = None,
        issue_plan: Optional["IssuePlan"] = None,
    ) -> Optional[RolloutResult]:
        selected: Optional[RolloutResult] = None
        # Feature E: stash the issue description so the deterministic-ranking
        # chokepoint can run the perspective-diverse tiebreaker without altering
        # internal call signatures. Fails open to "" if unset.
        self._perspective_issue_description = str(issue_description or "")
        try:
            for result in rollout_results:
                self._hydrate_candidate_handoff_artifacts(result)

            # Phase 2 cleanup: rollouts that failed for an environment /
            # harness reason (network, install, OOM, harness bug, ...) are
            # not APEX misses — they're "didn't run" outcomes and must not
            # block the round from succeeding. The same logic exists in
            # ``apex/evaluation/multi_candidate.select_best_testgen_candidate``
            # for the testgen pipeline; mirror it here so codegen-with-tests
            # selection has consistent semantics.
            env_failed_rollouts: list[RolloutResult] = []
            apex_pool: list[RolloutResult] = []
            for result in rollout_results:
                cls = getattr(result, "failure_class", None)
                if cls is not None and isinstance(cls, CoreFailureClass) and not cls.charges_apex:
                    env_failed_rollouts.append(result)
                else:
                    apex_pool.append(result)
            if env_failed_rollouts and not apex_pool:
                # Every rollout was an env / harness failure. Surface a
                # warning and abstain — selection has nothing to choose
                # from, but this should not be charged as an APEX miss.
                logger.warning(
                    "All %s rollouts failed for environment/harness reasons "
                    "(failure_class=%s); selection abstains rather than "
                    "blaming APEX.",
                    len(env_failed_rollouts),
                    sorted(
                        {
                            (
                                rollout.failure_class.value
                                if isinstance(rollout.failure_class, CoreFailureClass)
                                else "unknown"
                            )
                            for rollout in env_failed_rollouts
                        }
                    ),
                )
                return None
            # Mark env-failed rollouts as not selected so the Phase 1c
            # shim doesn't accidentally promote one of them when none of
            # the apex_pool candidates produced a usable patch.
            for env_rollout in env_failed_rollouts:
                env_rollout.selected_for_submission = False

            candidates = [result for result in apex_pool if result.success and result.patch]
            if not candidates:
                # When every rollout is marked success=False (typical when the
                # QV coverage-gap gate or a sandbox-policy demotion fires on
                # otherwise-passing rollouts) returning None throws away real
                # answers. Salvage the strongest near-success rollout — any
                # patch with QV pass_rate >= the near-miss floor — and run it
                # through the same fallback path that handles pruning collapse.
                # NOTE (Phase 2C followup): Phase 2C orchestrator agent will
                # rip out salvage-as-success and replace it with explicit
                # abstention. This call site is intentionally retained as
                # deprecated so the orchestrator can swap behaviour.
                near_success = [
                    result
                    for result in apex_pool
                    if result.patch
                    and self._candidate_validity_allows_submission(
                        result,
                        issue_plan=issue_plan,
                    )
                    and isinstance(result.quick_verification, dict)
                    and isinstance(result.quick_verification.get("pass_rate"), (int, float))
                    and float(result.quick_verification.get("pass_rate") or 0.0)
                    >= _ZERO_SUCCESS_FALLBACK_QV_PASS_RATE_FLOOR
                ]
                if near_success:
                    selected = self._fallback_pruned_candidate(
                        near_success,
                        issue_plan=issue_plan,
                        reason="zero_successful_rollouts",
                    )
                    selected = self._maybe_preserve_standalone_anchor(
                        rollout_results,
                        selected,
                        test_command,
                        baseline_result=baseline_result,
                        issue_plan=issue_plan,
                    )
                    return selected
                external_scoring_candidates = [
                    result
                    for result in apex_pool
                    if result.patch
                    and result.worktree_path
                    and self._candidate_validity_allows_external_scoring(
                        result,
                        issue_plan=issue_plan,
                    )
                ]
                if external_scoring_candidates:
                    selected = self._fallback_pruned_candidate(
                        external_scoring_candidates,
                        issue_plan=issue_plan,
                        reason="zero_successful_rollouts_external_scoring",
                    )
                    selected = self._maybe_preserve_standalone_anchor(
                        rollout_results,
                        selected,
                        test_command,
                        baseline_result=baseline_result,
                        issue_plan=issue_plan,
                    )
                    return selected
                return None
            self._write_task_selection_state(
                "selection_started",
                clear_rollout_state=True,
                extra_fields={"selection_candidate_count": len(candidates)},
            )
            initial_candidates = list(candidates)

            baseline = baseline_result
            if (
                baseline is None
                and test_command
                and self.config.selection.enable_regression_pruning
                and hasattr(self.verifier, "capture_baseline")
            ):
                self._write_task_selection_state(
                    "capturing_selection_baseline",
                    extra_fields={"selection_candidate_count": len(candidates)},
                )
                baseline = self.verifier.capture_baseline(self.repo_path, test_command)

            self._write_task_selection_state(
                "pruning_candidates",
                extra_fields={"selection_candidate_count": len(candidates)},
            )
            candidates = self._prune_candidates(
                candidates,
                baseline,
                test_command,
                issue_plan=issue_plan,
            )
            if not candidates:
                logger.warning("No candidates survived pruning.")
                selected = self._fallback_pruned_candidate(
                    initial_candidates,
                    issue_plan=issue_plan,
                    reason="all_candidates_pruned",
                )
                selected = self._maybe_preserve_standalone_anchor(
                    rollout_results,
                    selected,
                    test_command,
                    baseline_result=baseline,
                    issue_plan=issue_plan,
                )
                return selected

            if len(candidates) == 1:
                selected = self._verify_single_candidate(
                    candidates[0],
                    test_command,
                    baseline_result=baseline,
                    issue_plan=issue_plan,
                )
                if selected is None:
                    logger.warning("Only candidate was rejected by selection policy.")
                    return None
                selected = self._maybe_preserve_standalone_anchor(
                    rollout_results,
                    selected,
                    test_command,
                    baseline_result=baseline,
                    issue_plan=issue_plan,
                )
                return selected

            clusters = self._cluster_candidates(candidates)
            strategy = self.config.selection.strategy

            if strategy == SelectionStrategy.AST_CLUSTER:
                best_cluster = sorted(
                    clusters,
                    key=lambda cluster: (-cluster.size, cluster.cluster_id),
                )[0]
                selected = best_cluster.representative
                logger.info(
                    "Selected rollout %s from cluster %s using cluster-only strategy (size=%s)",
                    selected.rollout_id,
                    best_cluster.cluster_id,
                    best_cluster.size,
                )
                selected = self._maybe_preserve_standalone_anchor(
                    rollout_results,
                    selected,
                    test_command,
                    baseline_result=baseline,
                    issue_plan=issue_plan,
                )
                return selected

            self._write_task_selection_state(
                "verifying_clusters",
                extra_fields=self._selection_count_fields(
                    candidate_count=len(candidates),
                    cluster_count=len(clusters),
                    verified_cluster_count=0,
                ),
            )
            self._verify_clusters(
                clusters,
                candidates,
                test_command,
                baseline_result=baseline,
                issue_plan=issue_plan,
            )
            decisive_full_coverage_clusters = [
                cluster
                for cluster in clusters
                if self._cluster_has_decisive_full_expected_coverage_acceptance(cluster)
            ]
            ranking_state_fields = self._selection_count_fields(
                candidate_count=len(candidates),
                cluster_count=len(clusters),
                verified_cluster_count=len(clusters),
            )
            if decisive_full_coverage_clusters:
                ranking_state_fields.update(
                    {
                        "selection_synthesis_skipped": True,
                        "selection_synthesis_skip_reason": (
                            "decisive_full_expected_coverage_acceptance"
                        ),
                        "selection_synthesis_skip_cluster_count": len(
                            decisive_full_coverage_clusters
                        ),
                    }
                )
            else:
                self._write_task_selection_state(
                    "synthesizing_candidates",
                    extra_fields=self._selection_count_fields(
                        candidate_count=len(candidates),
                        cluster_count=len(clusters),
                        verified_cluster_count=len(clusters),
                    ),
                )
                synthetic_cluster = self._attempt_patch_synthesis(
                    clusters,
                    candidates,
                    test_command,
                    issue_plan=issue_plan,
                )
                if synthetic_cluster is not None:
                    clusters.append(synthetic_cluster)
                    ranking_state_fields = self._selection_count_fields(
                        candidate_count=len(candidates),
                        cluster_count=len(clusters),
                        verified_cluster_count=len(clusters),
                    )
            self._write_task_selection_state(
                "ranking_candidates",
                extra_fields=ranking_state_fields,
            )
            self._apply_selection_critic(
                clusters,
                candidates,
                issue_plan=issue_plan,
            )
            self._refresh_cluster_selection_context(
                clusters,
                test_command=test_command,
                issue_plan=issue_plan,
            )
            preferred_synthetic = self._preferred_verified_synthetic_cluster(
                clusters,
                issue_plan=issue_plan,
            )
            if preferred_synthetic is not None:
                self._write_task_selection_state(
                    "finalizing_selection",
                    extra_fields=self._selection_count_fields(
                        candidate_count=len(candidates),
                        cluster_count=len(clusters),
                        verified_cluster_count=len(clusters),
                        selectable_cluster_count=len(clusters),
                    ),
                )
                selected = self._finalize_cluster_selection(
                    preferred_synthetic,
                    test_command,
                )
                self._annotate_selection_mode(
                    selected,
                    mode="verified_synthetic_merge",
                    reason=(
                        "accepted_synthetic_candidate_had_stronger_verifier_score_"
                        "than_each_component_rollout"
                    ),
                    candidate_cluster_count=len(clusters),
                )
                logger.info(
                    "Selected verified synthetic rollout %s from cluster %s "
                    "(score=%.2f) before selector voting.",
                    selected.rollout_id,
                    preferred_synthetic.cluster_id,
                    preferred_synthetic.verification_score,
                )
                selected = self._maybe_preserve_standalone_anchor(
                    rollout_results,
                    selected,
                    test_command,
                    baseline_result=baseline,
                    issue_plan=issue_plan,
                )
                return selected
            selectable_clusters = self._prefer_accepted_clusters(
                clusters,
                issue_plan=issue_plan,
            )
            if not selectable_clusters:
                logger.warning("No candidate clusters satisfied the selection policy.")
                return None

            # APEX Decisive-Edge A.2: Verification Amplifier short-circuit.
            # If the orchestrator wired up a VerificationAmplifier on this
            # selector AND we have 2+ selectable clusters all passing the
            # existing test suite, ask the amplifier to generate
            # pair-wise discriminating tests and pick a winner with
            # genuinely-discriminating evidence instead of falling back
            # to AST/critic heuristics.
            amplified = self._maybe_apply_verification_amplifier(
                selectable_clusters,
                issue_description=issue_description,
                test_command=test_command,
                candidates=candidates,
                clusters=clusters,
            )
            if amplified is not None:
                selected = amplified
                selected = self._maybe_preserve_standalone_anchor(
                    rollout_results,
                    selected,
                    test_command,
                    baseline_result=baseline,
                    issue_plan=issue_plan,
                )
                return selected

            if (
                strategy in {SelectionStrategy.LLM_JUDGE, SelectionStrategy.MULTI_STAGE}
                and len(selectable_clusters) > 1
            ):
                selector_vote_disabled = self._component_disabled_for_clusters(
                    selectable_clusters,
                    "selector_vote",
                )
                selector_vote_redundant = self._selector_vote_is_redundant(
                    selectable_clusters,
                    test_command=test_command,
                )
                if selector_vote_disabled or selector_vote_redundant:
                    # Decisive verifier acceptance is already the authority; do
                    # not spend model-critic latency to break execution-equal ties.
                    best_cluster = (
                        self._best_cluster_by_decisive_acceptance(selectable_clusters)
                        if selector_vote_redundant
                        else self._best_cluster_by_deterministic_ranking(
                            selectable_clusters,
                            include_perspective=True,
                        )
                    )
                    self._write_task_selection_state(
                        "finalizing_selection",
                        extra_fields=self._selection_count_fields(
                            candidate_count=len(candidates),
                            cluster_count=len(clusters),
                            verified_cluster_count=len(clusters),
                            selectable_cluster_count=len(selectable_clusters),
                        ),
                    )
                    selected = self._finalize_cluster_selection(best_cluster, test_command)
                    self._annotate_selection_mode(
                        selected,
                        mode=(
                            "component_ablation_deterministic_selection"
                            if selector_vote_disabled
                            else "verification_fast_path"
                        ),
                        reason=(
                            "component_ablation_disabled_selector_vote"
                            if selector_vote_disabled
                            else "all_selectable_clusters_decisively_accepted"
                        ),
                        candidate_cluster_count=len(selectable_clusters),
                    )
                    logger.info(
                        "Skipped selector agent and selected rollout %s from cluster %s after decisive verifier acceptance across %s clusters.",
                        selected.rollout_id,
                        best_cluster.cluster_id,
                        len(selectable_clusters),
                    )
                    selected = self._maybe_preserve_standalone_anchor(
                        rollout_results,
                        selected,
                        test_command,
                        baseline_result=baseline,
                        issue_plan=issue_plan,
                    )
                    return selected
                candidate_pool = self._build_selector_candidate_pool(selectable_clusters)
                selector_state_fields = self._selection_count_fields(
                    candidate_count=len(candidates),
                    cluster_count=len(clusters),
                    verified_cluster_count=len(clusters),
                    selectable_cluster_count=len(selectable_clusters),
                    selector_pool_count=len(candidate_pool),
                )
                self._write_task_selection_state(
                    "selector_voting",
                    extra_fields=selector_state_fields,
                )

                def _selector_vote_progress(extra_fields: dict[str, Any]) -> None:
                    payload = dict(selector_state_fields)
                    payload.update(extra_fields)
                    self._write_task_selection_state(
                        "selector_voting",
                        extra_fields=payload,
                    )

                try:
                    winner = SelectorAgent(
                        self.config,
                        self.repo_path,
                    ).select_with_majority_voting(
                        candidate_pool,
                        issue_description,
                        max_voters=self.config.selection.selector_max_voters,
                        test_command=test_command,
                        progress_callback=_selector_vote_progress,
                    )
                    if winner not in candidate_pool:
                        raise ValueError("selector returned cluster outside candidate pool")
                except Exception as exc:  # noqa: BLE001 - verifier evidence is the fallback authority
                    winner = self._best_cluster_by_deterministic_ranking(candidate_pool)
                    diagnostics = (
                        dict(winner.representative.selection_diagnostics)
                        if isinstance(winner.representative.selection_diagnostics, dict)
                        else {}
                    )
                    diagnostics["selector_vote_fallback"] = {
                        "status": "deterministic_fallback",
                        "reason": f"{type(exc).__name__}: {str(exc)[:240]}",
                    }
                    winner.representative.selection_diagnostics = diagnostics
                self._write_task_selection_state(
                    "finalizing_selection",
                    extra_fields=selector_state_fields,
                )
                selected = self._finalize_cluster_selection(winner, test_command)
                logger.info(
                    "Selected rollout %s from cluster %s via selector agent",
                    selected.rollout_id,
                    winner.cluster_id,
                )
                selected = self._maybe_preserve_standalone_anchor(
                    rollout_results,
                    selected,
                    test_command,
                    baseline_result=baseline,
                    issue_plan=issue_plan,
                )
                return selected

            best_cluster = self._best_cluster_by_deterministic_ranking(selectable_clusters)
            self._write_task_selection_state(
                "finalizing_selection",
                extra_fields=self._selection_count_fields(
                    candidate_count=len(candidates),
                    cluster_count=len(clusters),
                    verified_cluster_count=len(clusters),
                    selectable_cluster_count=len(selectable_clusters),
                ),
            )
            selected = self._finalize_cluster_selection(best_cluster, test_command)

            logger.info(
                "Selected rollout %s from cluster %s (score=%.2f, size=%s)",
                selected.rollout_id,
                best_cluster.cluster_id,
                best_cluster.combined_score,
                best_cluster.size,
            )
            selected = self._maybe_preserve_standalone_anchor(
                rollout_results,
                selected,
                test_command,
                baseline_result=baseline,
                issue_plan=issue_plan,
            )
            return selected
        finally:
            # Phase 2 bonus item (1.10 wiring): mark every non-selected
            # rollout as ``selected_for_submission=False`` so the
            # downstream artifact pipeline can trust the flag. The
            # selected winner gets ``True`` from
            # ``_finalize_cluster_selection`` /
            # ``_verify_single_candidate`` / ``_fallback_pruned_candidate``;
            # everything else is explicitly cleared here. Env-failed
            # rollouts are also covered because we already cleared their
            # flag at the top of the function.
            self._stamp_selection_flags(rollout_results, selected)
            keep_path = None
            if selected is not None and selected.worktree_path:
                selected_path = Path(selected.worktree_path).resolve()
                if any(
                    workspace.resolve() == selected_path for workspace in self._ephemeral_worktrees
                ):
                    keep_path = selected.worktree_path
            self._cleanup_synthetic_worktrees(keep_path=keep_path)

    def _stamp_selection_flags(
        self,
        rollout_results: list[RolloutResult],
        selected: Optional[RolloutResult],
    ) -> None:
        """Set ``selected_for_submission`` definitively on every rollout.

        After Phase 2.B this is the single source of truth for the flag
        on the codegen-with-tests path. The Phase 1c
        ``_ensure_selection_flag`` shim in eval runners remains as a
        defensive last-resort default — but for any rollout that flowed
        through ``select_best_patch`` the shim should be a no-op.
        """
        selected_id = int(getattr(selected, "rollout_id", 0)) if selected is not None else None
        selected_can_submit = self._selected_rollout_should_submit(selected)
        # Identity check is safer than rollout_id comparison because
        # synthetic candidates use negative ids that may collide with
        # other synthetic groups across calls.
        for result in rollout_results:
            if selected is not None and result is selected:
                # Final winner; the wiring earlier already set True, but
                # write it again to be defensive against partial failures.
                result.selected_for_submission = selected_can_submit
                continue
            # Synthetic candidate winners may not appear in
            # ``rollout_results`` (they're constructed inside the
            # selector); only stamp when the result here is genuinely
            # not the winner.
            if (
                selected is not None
                and selected_id is not None
                and int(getattr(result, "rollout_id", 0)) == selected_id
                and bool(getattr(result, "is_synthetic", False))
                == bool(getattr(selected, "is_synthetic", False))
            ):
                # Same logical winner reached us by id — be defensive.
                result.selected_for_submission = selected_can_submit
                continue
            result.selected_for_submission = False

    @staticmethod
    def _selected_rollout_should_submit(selected: Optional[RolloutResult]) -> bool:
        if selected is None:
            return False
        verification = (
            dict(selected.verification) if isinstance(selected.verification, dict) else {}
        )
        if verification.get("selected_for_submission") is False:
            return False
        if PatchSelector._candidate_has_explicit_hard_validity_rejection(selected):
            return False
        if verification_has_explicit_validity_rejection(verification):
            return False
        if verification.get("selected_for_submission") is True:
            return True
        return rollout_has_authoritative_acceptance(selected)

    @staticmethod
    def _candidate_has_explicit_hard_validity_rejection(candidate: RolloutResult) -> bool:
        validity = PatchSelector._candidate_validity_payload(candidate)
        if not validity:
            return False
        if validity.get("expected_coverage_preserved") is False:
            return True
        missing_expected = validity.get("missing_expected_test_count")
        if (
            isinstance(missing_expected, int)
            and not isinstance(missing_expected, bool)
            and missing_expected > 0
        ):
            return True
        return verification_has_explicit_validity_rejection({"validity": validity})

    @staticmethod
    def _issue_plan_evidence_mode(issue_plan: Optional["IssuePlan"]) -> str:
        if issue_plan is None:
            return ""
        try:
            policy = issue_plan.evaluation_constraints.resolved_evidence_policy()
            return str(getattr(policy, "mode", "") or "").strip()
        except Exception:
            test_context = getattr(issue_plan, "test_context", None)
            return str(getattr(test_context, "evidence_mode", "") or "").strip()

    def _gold_suite_visible_mode(self, issue_plan: Optional["IssuePlan"]) -> bool:
        return self._issue_plan_evidence_mode(issue_plan) == "gold_suite_visible"

    def _candidate_validity_allows_external_scoring(
        self,
        candidate: RolloutResult,
        *,
        issue_plan: Optional["IssuePlan"],
    ) -> bool:
        if not self._gold_suite_visible_mode(issue_plan):
            return True
        validity = getattr(candidate, "validity", None)
        return bool(validity and validity.eligible_for_external_scoring)

    def _candidate_validity_allows_submission(
        self,
        candidate: RolloutResult,
        *,
        issue_plan: Optional["IssuePlan"],
    ) -> bool:
        if not self._gold_suite_visible_mode(issue_plan):
            return True
        validity = getattr(candidate, "validity", None)
        if not (validity and validity.eligible_for_submission):
            return False
        worktree_path = str(candidate.worktree_path or "").strip()
        return bool(worktree_path and Path(worktree_path).is_dir())

    def _fallback_candidate_materialization_path(
        self,
        candidate: RolloutResult,
    ) -> Path:
        output_dir = self._selection_status_output_dir()
        if output_dir is not None:
            return output_dir / "selected_candidate" / f"rollout_{candidate.rollout_id}"
        workspace_root = Path(tempfile.mkdtemp(prefix="apex_selected_candidate_"))
        stable_path = workspace_root / f"rollout_{candidate.rollout_id}"
        self._ephemeral_worktrees.append(stable_path)
        self._ephemeral_workspace_modes[stable_path.resolve()] = "git_clone"
        return stable_path

    def _fallback_candidate_baseline(self, candidate: RolloutResult) -> str:
        baseline = str(candidate.baseline_commit or "").strip()
        if baseline:
            return baseline
        result = _run_selection_git(
            ["rev-parse", "HEAD"],
            cwd=Path(self.repo_path),
            timeout=_SELECTION_GIT_QUICK_TIMEOUT_SECONDS,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return "HEAD"

    def _remove_materialized_candidate_path(self, stable_path: Path) -> None:
        if not stable_path.exists():
            return
        _run_selection_git(
            ["worktree", "remove", "--force", str(stable_path)],
            cwd=Path(self.repo_path),
            timeout=_SELECTION_GIT_WORKTREE_REMOVE_TIMEOUT_SECONDS,
        )
        if stable_path.exists():
            shutil.rmtree(stable_path, ignore_errors=True)

    def _rematerialize_candidate(
        self,
        candidate: RolloutResult,
        stable_path: Path,
    ) -> Optional[Path]:
        patch_text = str(candidate.patch or "")
        if not patch_text.strip():
            return None
        self._remove_materialized_candidate_path(stable_path)
        stable_path.parent.mkdir(parents=True, exist_ok=True)
        baseline = self._fallback_candidate_baseline(candidate)
        checkout = _run_selection_git(
            ["worktree", "add", "--detach", str(stable_path), baseline],
            cwd=Path(self.repo_path),
            timeout=_SELECTION_GIT_DEFAULT_TIMEOUT_SECONDS,
        )
        if checkout.returncode != 0:
            logger.warning(
                "Failed to materialize fallback candidate rollout %s at %s: %s",
                candidate.rollout_id,
                stable_path,
                (checkout.stderr or checkout.stdout or "").strip() or "git worktree add failed",
            )
            return None
        patch_file: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                suffix=".patch",
                prefix="apex-candidate-",
                dir=str(stable_path.parent),
                delete=False,
            ) as handle:
                handle.write(patch_text)
                patch_file = Path(handle.name)
            apply_result = _run_selection_git(
                ["apply", "--3way", "--whitespace=nowarn", str(patch_file)],
                cwd=stable_path,
                timeout=_SELECTION_GIT_DEFAULT_TIMEOUT_SECONDS,
            )
        finally:
            if patch_file is not None:
                patch_file.unlink(missing_ok=True)
        if apply_result.returncode != 0:
            logger.warning(
                "Failed to apply fallback candidate patch for rollout %s at %s: %s",
                candidate.rollout_id,
                stable_path,
                (apply_result.stderr or apply_result.stdout or "").strip() or "git apply failed",
            )
            self._remove_materialized_candidate_path(stable_path)
            return None
        return stable_path.resolve()

    def _ensure_candidate_worktree_materialized(
        self,
        candidate: RolloutResult,
        *,
        reason: str,
    ) -> bool:
        current_worktree_path = str(candidate.worktree_path or "").strip()
        if current_worktree_path and Path(current_worktree_path).is_dir():
            return True
        if not str(candidate.patch or "").strip():
            return False
        if not is_git_repo(Path(self.repo_path)):
            return False

        stable_path = self._fallback_candidate_materialization_path(candidate)
        materialized_path = self._rematerialize_candidate(candidate, stable_path)
        if materialized_path is None:
            candidate.failure_reason = "candidate_materialization_failed"
            validity = getattr(candidate, "validity", None)
            if validity is not None:
                validity.worktree_materialized = False
                if "candidate_materialization_failed" not in validity.reasons:
                    validity.reasons.append("candidate_materialization_failed")
            return False

        candidate.worktree_path = str(materialized_path)
        validity = getattr(candidate, "validity", None)
        if validity is not None:
            validity.worktree_materialized = True
        metadata = (
            dict(candidate.search_metadata) if isinstance(candidate.search_metadata, dict) else {}
        )
        metadata["candidate_worktree_rematerialized"] = True
        metadata["candidate_worktree_rematerialized_reason"] = str(reason)
        metadata["candidate_worktree_rematerialized_path"] = str(materialized_path)
        candidate.search_metadata = metadata
        return True

    def _fallback_pruned_candidate(
        self,
        candidates: list[RolloutResult],
        *,
        issue_plan: Optional["IssuePlan"],
        reason: str,
    ) -> Optional[RolloutResult]:
        """Return the strongest non-accepted candidate after pruning collapse.

        This is intentionally conservative: it does NOT mark the candidate as
        accepted, it preserves visible-test policy checks, and it exists to
        keep residual-followup / benchmark-side candidate evaluation alive when
        regression pruning removes every candidate from consideration.

        DEPRECATED (Phase 2C orchestrator owner): the salvage-as-success
        path is scheduled to be replaced with explicit abstention in
        Phase 2C. Don't add new call sites; the Phase 2C agent will
        excise this method once the orchestrator-side abstention plumbing
        lands.
        """

        ranked: list[
            tuple[tuple[float, float, float, float, float, int], RolloutResult, dict[str, Any]]
        ] = []
        authoritative_scoring_nomination = reason == "zero_successful_rollouts_external_scoring"
        for candidate in candidates:
            if not candidate.patch:
                continue
            if authoritative_scoring_nomination:
                validity_allows_candidate = self._candidate_validity_allows_external_scoring(
                    candidate,
                    issue_plan=issue_plan,
                )
            else:
                validity_allows_candidate = self._candidate_validity_allows_submission(
                    candidate,
                    issue_plan=issue_plan,
                )
            if not validity_allows_candidate:
                continue
            policy_reason = self._candidate_test_edit_policy_reason(
                candidate,
                issue_plan=issue_plan,
            )
            if policy_reason:
                continue
            prune_payload = self._serialize_prune_result(
                self._prune_results.get(candidate.rollout_id, {})
            )
            regressed_tests = list(prune_payload.get("regressed_tests") or [])
            still_passing = list(prune_payload.get("still_passing") or [])
            quick_verification = (
                candidate.quick_verification
                if isinstance(candidate.quick_verification, dict)
                else {}
            )
            quick_signal = quick_verification_signal_score(quick_verification)
            quick_pass_rate = quick_verification.get("pass_rate")
            progress_score = getattr(candidate, "progress_score", None)
            sort_key = (
                float(quick_signal) if isinstance(quick_signal, (int, float)) else -1.0,
                float(quick_pass_rate) if isinstance(quick_pass_rate, (int, float)) else -1.0,
                float(progress_score) if isinstance(progress_score, (int, float)) else -1.0,
                float(len(still_passing) - len(regressed_tests)),
                # Merit tiebreak (sweep #6): when all execution-grounded signals
                # tie (common when QV is indeterminate), prefer the candidate that
                # changed MORE files over a do-nothing one, instead of falling
                # straight through to the lowest rollout_id. Strictly below every
                # execution signal; -rollout_id stays the final byte-stable key.
                float(len(self._candidate_changed_files(candidate))),
                -int(candidate.rollout_id),
            )
            ranked.append((sort_key, candidate, prune_payload))

        if not ranked:
            return None

        ranked.sort(key=lambda item: item[0], reverse=True)
        _, candidate, prune_payload = ranked[0]
        original_worktree_path = str(candidate.worktree_path or "").strip()
        if (
            (not original_worktree_path or not Path(original_worktree_path).is_dir())
            and is_git_repo(Path(self.repo_path))
            and not self._ensure_candidate_worktree_materialized(candidate, reason=reason)
        ):
            return None
        rematerialized_worktree_path = (
            str(candidate.worktree_path or "").strip()
            if str(candidate.worktree_path or "").strip() != original_worktree_path
            else None
        )
        selection_diagnostics = (
            dict(candidate.selection_diagnostics)
            if isinstance(candidate.selection_diagnostics, dict)
            else {}
        )
        quick_verification = (
            candidate.quick_verification if isinstance(candidate.quick_verification, dict) else {}
        )
        fallback_payload = {
            "reason": reason,
            "quick_signal_score": quick_verification_signal_score(quick_verification),
            "quick_observed_pass_rate": quick_verification.get("pass_rate"),
            "missing_expected_test_count": quick_verification.get("missing_expected_test_count"),
            "prune_result": prune_payload,
            "authoritative_scoring_nomination": authoritative_scoring_nomination,
        }
        if rematerialized_worktree_path:
            fallback_payload["rematerialized_worktree_path"] = rematerialized_worktree_path
        selection_diagnostics["fallback"] = fallback_payload
        # Attach stub-residue findings (cross-language) so the residual
        # followup can name the unimplemented functions in its prompt
        # instead of just saying "fix the failing tests."
        stub_findings = self._candidate_stub_residue_findings(candidate)
        if stub_findings:
            selection_diagnostics["stub_residue"] = [
                {
                    "path": getattr(f, "path", ""),
                    "symbol": getattr(f, "symbol", ""),
                    "reason": getattr(f, "reason", ""),
                }
                for f in stub_findings[:25]
            ]
        # Public-symbol survival losses (Python AST diff vs apex-base).
        # Catches the most common test-collection-breakage form where
        # the agent removes a function that conftest / other test
        # modules transitively import.
        symbol_losses = self._candidate_public_symbol_losses(candidate)
        if symbol_losses:
            selection_diagnostics["public_symbol_losses"] = [
                {
                    "path": getattr(loss, "path", ""),
                    "symbol": getattr(loss, "symbol", ""),
                    "kind": getattr(loss, "kind", ""),
                }
                for loss in symbol_losses[:25]
            ]
        candidate.selection_diagnostics = selection_diagnostics
        allow_explicit_salvage = bool(getattr(self.config.rollout, "allow_salvage", False))
        selected_for_submission = bool(
            allow_explicit_salvage and not authoritative_scoring_nomination
        )
        candidate.selected_for_submission = selected_for_submission
        candidate.internally_accepted = False
        candidate.salvaged_for_external_scoring = True
        verification_payload = (
            dict(candidate.verification) if isinstance(candidate.verification, dict) else {}
        )
        verification_payload.update(
            {
                "accepted": False,
                "internally_accepted": False,
                "selected_for_submission": selected_for_submission,
                "salvaged_for_external_scoring": True,
                "selection_authority": (
                    "authoritative_scoring_nomination"
                    if authoritative_scoring_nomination
                    else (
                        "explicit_salvage_submission"
                        if allow_explicit_salvage
                        else "repair_seed_only"
                    )
                ),
                "prune_result": prune_payload,
            }
        )
        candidate.verification = verification_payload
        logger.warning(
            "Selection fallback retained rollout %s after pruning collapse (%s).",
            candidate.rollout_id,
            reason,
        )
        return candidate

    def _build_selector_candidate_pool(self, clusters: list[PatchCluster]) -> list[PatchCluster]:
        ranked = sorted(
            clusters,
            key=lambda cluster: (
                cluster.combined_score,
                cluster.accepted,
                cluster.public_signal_score,
                cluster.critic_score,
                cluster.verification_score,
                cluster.cross_validation_score,
                cluster.size,
                -cluster.cluster_id,
            ),
            reverse=True,
        )
        pool_limit = min(len(ranked), max(2, self.config.selection.selector_max_voters))
        if len(ranked) <= pool_limit:
            return ranked

        cutoff_cluster = ranked[pool_limit - 1]
        cutoff_score = (
            cutoff_cluster.combined_score,
            cutoff_cluster.public_signal_score,
            cutoff_cluster.critic_score,
            cutoff_cluster.verification_score,
            cutoff_cluster.cross_validation_score,
            cutoff_cluster.size,
        )
        pool: list[PatchCluster] = []
        for cluster in ranked:
            cluster_score = (
                cluster.combined_score,
                cluster.public_signal_score,
                cluster.critic_score,
                cluster.verification_score,
                cluster.cross_validation_score,
                cluster.size,
            )
            if len(pool) < pool_limit or cluster_score >= cutoff_score:
                pool.append(cluster)
                continue
            break
        return pool

    def _eg_critic_active(self, clusters: list[PatchCluster]) -> bool:
        """WS2C: True only when the config flag is on AND a FITTED EG-critic
        artifact is loaded. (The shipped artifact is non-fitted, so this is False
        by default — two independent default-off gates.) The optional-component
        arm, when present on the clusters, is honored as an additional gate."""
        if not self.config.selection.enable_eg_critic_tiebreak:
            return False
        if not self._eg_critic_loaded:
            self._eg_critic = load_eg_critic(self.config.selection.eg_critic_weights_path)
            self._eg_critic_loaded = True
        if self._eg_critic is None or not getattr(self._eg_critic, "fitted", False):
            return False
        # If any cluster carries a component-ablation assignment, require the
        # eg_critic optional arm to be enabled there too (third default-off gate).
        assignments = [getattr(c.representative, "search_metadata", {}) or {} for c in clusters]
        arms = [
            a.get("component_ablation")
            for a in assignments
            if isinstance(a, dict) and isinstance(a.get("component_ablation"), dict)
        ]
        if arms and not any(component_optional_enabled(a, "eg_critic") for a in arms):
            return False
        return True

    def _eg_critic_tiebreak_value(self, cluster: PatchCluster) -> float:
        """Learned tie-break value for ``cluster`` (0.0 when inactive — a no-op in
        the ranking tuple so it only separates otherwise-execution-tied clusters)."""
        if self._eg_critic is None:
            return 0.0
        try:
            features = extract_execution_features(cluster.representative, cluster.verification)
            return float(self._eg_critic.score_features(features))
        except Exception:  # noqa: BLE001 - learned tie-break must never break ranking
            return 0.0

    def _best_cluster_by_deterministic_ranking(
        self,
        clusters: list[PatchCluster],
        *,
        include_perspective: bool = True,
    ) -> PatchCluster:
        # WS2C: the learned EG-critic enters ONLY as a low-priority tie-break
        # (after every execution-grounded key), so it can separate
        # otherwise-tied clusters but NEVER override execution evidence. Inactive
        # -> 0.0 for all clusters -> ordering is byte-identical to before.
        eg_active = self._eg_critic_active(clusters)
        # Feature E: when enabled for this ranking call, the
        # perspective-diverse model critic enters AFTER every
        # execution-grounded key and after the EG-critic, and BEFORE the final
        # -cluster_id deterministic stabiliser. It can only re-order clusters
        # that are otherwise tied on all execution evidence, so it NEVER
        # overrides concrete verification signals. Inactive / failed-open ->
        # 0.0 for every cluster, leaving the ordering byte-identical.
        perspective_scores = self._maybe_perspective_scores(clusters) if include_perspective else {}
        return max(
            clusters,
            key=lambda cluster: (
                cluster.combined_score,
                cluster.accepted,
                cluster.public_signal_score,
                cluster.critic_score,
                cluster.size,
                cluster.verification_score,
                (self._eg_critic_tiebreak_value(cluster) if eg_active else 0.0),
                perspective_scores.get(cluster.cluster_id, 0.0),
                # Merit tiebreak (sweep #6): among clusters tied on ALL execution
                # + critic signals, prefer the one whose representative changed
                # more files over a do-nothing one, instead of falling straight to
                # the lowest cluster_id. Strictly below every execution/critic key;
                # -cluster_id stays the final byte-stable stabiliser.
                float(len(self._candidate_changed_files(cluster.representative))),
                -cluster.cluster_id,
            ),
        )

    def _best_cluster_by_decisive_acceptance(
        self,
        clusters: list[PatchCluster],
    ) -> PatchCluster:
        """Rank clusters after all selectable candidates have verifier acceptance."""

        return max(
            clusters,
            key=lambda cluster: (
                cluster.accepted,
                cluster.verification_score,
                cluster.public_signal_score,
                cluster.combined_score,
                cluster.critic_score,
                cluster.size,
                float(len(self._candidate_changed_files(cluster.representative))),
                -cluster.cluster_id,
            ),
        )

    def _perspective_review_active(self, clusters: list[PatchCluster]) -> bool:
        """Feature E gate: enabled flag set, reviewer injected, and at least
        ``perspective_review_min_candidates`` ACCEPTED (accept-tier) clusters to
        re-order. Anything else -> the tiebreaker is a no-op."""
        if not getattr(self.config.selection, "enable_perspective_review", False):
            return False
        if self.perspective_reviewer is None:
            return False
        # Honour the configured threshold; a single-candidate tier can never be
        # re-ordered, so floor at 2 for any non-positive / nonsensical value.
        configured = int(
            getattr(self.config.selection, "perspective_review_min_candidates", 2) or 2
        )
        min_candidates = configured if configured >= 1 else 2
        accept_tier = [cluster for cluster in clusters if cluster.accepted]
        return len(accept_tier) >= min_candidates

    def _maybe_perspective_scores(
        self,
        clusters: list[PatchCluster],
    ) -> dict[int, float]:
        """Return ``{cluster_id: perspective_score}`` for the accept tier.

        Only execution-VERIFIED/ACCEPTED clusters are scored; everything else
        (and the whole map when the feature is inactive) defaults to ``0.0`` so
        the ranking tuple is unchanged. MUST FAIL OPEN: any error returns an
        empty map, leaving ranking untouched.
        """
        if not self._perspective_review_active(clusters):
            return {}
        scores: dict[int, float] = {}
        reviewer = self.perspective_reviewer
        try:
            for cluster in clusters:
                if not cluster.accepted:
                    continue
                representative = cluster.representative
                try:
                    changed_files = self._candidate_changed_files(representative)
                except Exception:  # noqa: BLE001 - file listing must not break ranking
                    changed_files = list(getattr(representative, "changed_files", []) or [])
                lens_scores = reviewer.score_candidate(
                    representative,
                    issue_description=self._perspective_issue_description,
                    changed_files=changed_files,
                    test_summary=self._perspective_test_summary(cluster),
                )
                aggregate = reviewer.aggregate(lens_scores)
                scores[cluster.cluster_id] = float(aggregate)
                # Store on the cluster + selection diagnostics for observability.
                cluster.perspective_score = float(aggregate)
                cluster.perspective_scores = dict(lens_scores)
                self._record_perspective_diagnostics(cluster, lens_scores, aggregate)
        except Exception:  # noqa: BLE001 - perspective tiebreaker must fail open
            logger.debug("Perspective review failed open; ranking unchanged", exc_info=True)
            return {}
        return scores

    @staticmethod
    def _perspective_test_summary(cluster: PatchCluster) -> str:
        verification = getattr(cluster, "verification", None)
        test_result = getattr(verification, "test_result", None)
        if test_result is None:
            return ""
        try:
            return (
                f"passed={getattr(test_result, 'passed', '')} "
                f"failed={getattr(test_result, 'failed', '')} "
                f"errors={getattr(test_result, 'errors', '')} "
                f"pass_rate={getattr(test_result, 'pass_rate', '')}"
            )
        except Exception:  # noqa: BLE001 - summary is best-effort
            return ""

    @staticmethod
    def _record_perspective_diagnostics(
        cluster: PatchCluster,
        lens_scores: dict[str, float],
        aggregate: float,
    ) -> None:
        representative = cluster.representative
        diagnostics = (
            dict(representative.selection_diagnostics)
            if isinstance(representative.selection_diagnostics, dict)
            else {}
        )
        diagnostics["perspective_review"] = {
            "used": True,
            "aggregate": float(aggregate),
            "lenses": {k: float(v) for k, v in (lens_scores or {}).items()},
        }
        representative.selection_diagnostics = diagnostics

    def _preferred_verified_synthetic_cluster(
        self,
        clusters: list[PatchCluster],
        *,
        issue_plan: Optional["IssuePlan"] = None,
    ) -> Optional[PatchCluster]:
        """Prefer a verified synthetic merge when it beats every component.

        Patch synthesis exists to combine compatible partial fixes. Once that
        merged candidate independently verifies better than each source
        rollout, critic features or selector voting should not demote it back
        to an incomplete component patch.
        """

        valid_synthetic: list[PatchCluster] = []
        non_synthetic_scores: list[float] = []
        for cluster in clusters:
            representative = cluster.representative
            if bool(getattr(representative, "is_synthetic", False)):
                if (
                    issue_plan is not None
                    and canonical_expected_test_count(issue_plan) > 0
                    and not self._verification_has_proven_full_expected_coverage(
                        cluster.verification
                    )
                ):
                    continue
                if (
                    cluster.accepted
                    and getattr(
                        getattr(cluster.verification, "test_result", None),
                        "expected_coverage_preserved",
                        None,
                    )
                    is not False
                    and not self._candidate_test_edit_policy_reason(
                        representative,
                        issue_plan=issue_plan,
                        verification=cluster.verification,
                    )
                ):
                    valid_synthetic.append(cluster)
                continue
            non_synthetic_scores.append(float(cluster.verification_score))

        if not valid_synthetic:
            return None

        best = max(
            valid_synthetic,
            key=lambda cluster: (
                cluster.verification_score,
                cluster.combined_score,
                len(self._candidate_changed_files(cluster.representative)),
                -cluster.cluster_id,
            ),
        )
        best_component_score = max(non_synthetic_scores, default=-1.0)
        if float(best.verification_score) <= best_component_score + 1e-9:
            return None
        return best

    def _selector_vote_is_redundant(
        self,
        clusters: list[PatchCluster],
        *,
        test_command: Optional[str],
    ) -> bool:
        if len(clusters) <= 1 or not str(test_command or "").strip():
            return False
        return all(self._cluster_has_decisive_acceptance(cluster) for cluster in clusters)

    def _cluster_has_decisive_acceptance(self, cluster: PatchCluster) -> bool:
        verification = cluster.verification
        # The selector acceptance bar is stamped on the cluster even when the raw
        # verifier object predates that decision.
        if verification is None or not (cluster.accepted or verification.accepted):
            return False

        test_result = verification.test_result
        if test_result is None or not test_result.regression_passes:
            return False
        if int(test_result.failed or 0) != 0 or int(test_result.errors or 0) != 0:
            return False
        if test_result.pass_rate < 0.999:
            return False

        expected_coverage_preserved = getattr(
            test_result,
            "expected_coverage_preserved",
            None,
        )
        if expected_coverage_preserved is False:
            return False

        expected_test_count = int(getattr(test_result, "expected_test_count", 0) or 0)
        if expected_test_count <= 0:
            return True

        matched_expected_test_count = int(
            getattr(test_result, "matched_expected_test_count", 0) or 0
        )
        missing_expected_test_count = int(
            getattr(test_result, "missing_expected_test_count", 0) or 0
        )
        return (
            expected_coverage_preserved is True
            and missing_expected_test_count == 0
            and matched_expected_test_count >= expected_test_count
        )

    def _cluster_has_decisive_full_expected_coverage_acceptance(
        self,
        cluster: PatchCluster,
    ) -> bool:
        """Return true when synthesis would only re-verify after a proved win."""

        verification = cluster.verification
        if verification is None:
            return False
        return bool(
            self._cluster_has_decisive_acceptance(cluster)
            and self._verification_has_proven_full_expected_coverage(verification)
        )

    # ------------------------------------------------------------------
    # APEX Decisive-Edge A.2: Verification Amplifier wiring
    # ------------------------------------------------------------------

    def _maybe_apply_verification_amplifier(
        self,
        selectable_clusters: list[PatchCluster],
        *,
        issue_description: str,
        test_command: Optional[str],
        candidates: list[RolloutResult],
        clusters: list[PatchCluster],
    ) -> Optional[RolloutResult]:
        """Run the verification amplifier on tied selectable clusters.

        Returns a :class:`RolloutResult` when amplification produced a
        high-confidence winner; ``None`` to fall back to the existing
        downstream selector logic (selector vote / deterministic
        ranking).

        The amplifier is opt-in: it is only invoked if
        ``self.verification_amplifier`` was set by the orchestrator (or
        a test). When unset, this method is a no-op so legacy callers
        see no behavior change.
        """
        amplifier = self.verification_amplifier
        if amplifier is None:
            return None
        if len(selectable_clusters) < 2:
            return None

        n_tied = len(selectable_clusters)
        patches: list[str] = []
        for cluster in selectable_clusters:
            patch_text = cluster.representative.patch or ""
            patches.append(patch_text)

        try:
            amplification = amplifier.amplify(
                patches=patches,
                task_context=issue_description,
                repo_path=Path(self.repo_path),
            )
        except Exception:
            logger.exception(
                "Verification amplifier raised; falling back to existing selection logic."
            )
            return None

        # Always log amplifier usage for downstream analysis.
        logger.info(
            "verification amplifier ran: n_candidates_tied=%d "
            "amplifier_used=True winner_index=%d confidence=%.3f "
            "cost_inferences=%d cost_test_runs=%d short_circuit=%s",
            n_tied,
            amplification.chosen_patch,
            amplification.confidence,
            amplification.cost_inferences,
            amplification.cost_test_runs,
            amplification.short_circuit_reason,
        )

        if amplification.short_circuited:
            return None

        confidence_floor = float(self.verification_amplifier_confidence_threshold)
        if amplification.confidence < confidence_floor:
            return None

        winner_idx = amplification.chosen_patch
        if winner_idx < 0 or winner_idx >= n_tied:
            return None

        winning_cluster = selectable_clusters[winner_idx]
        # Capture pre-amplifier ranking signals so we can preserve them
        # as ``tiebreak_evidence`` for downstream analysis.
        tiebreak_evidence = self._build_tiebreak_evidence_snapshot(selectable_clusters)

        selected = self._finalize_cluster_selection(winning_cluster, test_command)
        self._annotate_selection_mode(
            selected,
            mode="verification_amplifier",
            reason="discriminating_test_majority",
            candidate_cluster_count=n_tied,
        )

        diagnostics = (
            dict(selected.selection_diagnostics)
            if isinstance(selected.selection_diagnostics, dict)
            else {}
        )
        diagnostics["amplifier"] = {
            "used": True,
            "n_candidates_tied": int(n_tied),
            "winner_index": int(winner_idx),
            "confidence": float(amplification.confidence),
            "cost_inferences": int(amplification.cost_inferences),
            "cost_test_runs": int(amplification.cost_test_runs),
            "win_counts": list(amplification.win_counts),
            "matrix": amplification.discrimination_matrix.to_dict(),
        }
        diagnostics["tiebreak_evidence"] = tiebreak_evidence
        selected.selection_diagnostics = diagnostics
        return selected

    def _build_tiebreak_evidence_snapshot(
        self,
        clusters: list[PatchCluster],
    ) -> list[dict[str, Any]]:
        """Snapshot pre-amplifier scoring signals for each tied cluster."""
        snapshot: list[dict[str, Any]] = []
        for cluster in clusters:
            snapshot.append(
                {
                    "cluster_id": int(cluster.cluster_id),
                    "size": int(cluster.size),
                    "verification_score": float(cluster.verification_score),
                    "critic_score": float(cluster.critic_score),
                    "combined_score": float(cluster.combined_score),
                    "vote_count": int(cluster.vote_count),
                    "rollout_id": int(getattr(cluster.representative, "rollout_id", 0) or 0),
                }
            )
        return snapshot

    def _annotate_selection_mode(
        self,
        result: RolloutResult,
        *,
        mode: str,
        reason: str,
        candidate_cluster_count: int,
    ) -> None:
        diagnostics = (
            dict(result.selection_diagnostics)
            if isinstance(result.selection_diagnostics, dict)
            else {}
        )
        selector_diagnostics = (
            dict(diagnostics.get("selector"))
            if isinstance(diagnostics.get("selector"), dict)
            else {}
        )
        selector_diagnostics.update(
            {
                "mode": str(mode),
                "reason": str(reason),
                "candidate_cluster_count": int(max(candidate_cluster_count, 0)),
            }
        )
        diagnostics["selector"] = selector_diagnostics
        result.selection_diagnostics = diagnostics

    def _prune_candidates(
        self,
        candidates: list[RolloutResult],
        baseline: Any,
        test_command: Optional[str],
        *,
        issue_plan: Optional["IssuePlan"] = None,
    ) -> list[RolloutResult]:
        valid_candidates: list[RolloutResult] = []
        allow_test_only = self._allow_test_only_patches(issue_plan)
        for candidate in candidates:
            syntax_valid = self._candidate_has_valid_syntax(candidate)
            if not syntax_valid:
                logger.info("Pruned rollout %s due to syntax failure.", candidate.rollout_id)
                self._prune_results[candidate.rollout_id] = {
                    "is_valid": False,
                    "reason": "syntax_invalid",
                    "regressed_tests": [],
                    "still_passing": [],
                }
                continue

            # Phase 2.3: reject patches that don't represent substantive
            # work (whitespace-only, comment-only, docstring-only, blank-
            # line-only). These previously slipped through pruning,
            # received a None fingerprint, and could attach to a real
            # cluster by coincidence — biasing selection toward the
            # appearance of consensus that doesn't actually exist.
            substantive, substantive_reason = _patch_is_substantive(
                candidate.patch or "",
                allow_test_only=allow_test_only,
            )
            if not substantive:
                logger.info(
                    "Pruned rollout %s as non-substantive patch (%s).",
                    candidate.rollout_id,
                    substantive_reason,
                )
                self._prune_results[candidate.rollout_id] = {
                    "is_valid": False,
                    "reason": "non_substantive_patch",
                    "non_substantive_reason": substantive_reason,
                    "regressed_tests": [],
                    "still_passing": [],
                }
                continue

            test_edit_policy_reason = self._candidate_test_edit_policy_reason(
                candidate,
                issue_plan=issue_plan,
            )
            if test_edit_policy_reason and test_edit_policy_reason.startswith(
                "unexpected visible test edits:"
            ):
                sanitized_candidate = self._sanitize_candidate_visible_test_edits(
                    candidate,
                    issue_plan=issue_plan,
                )
                if sanitized_candidate is not None:
                    logger.info(
                        "Sanitized rollout %s by restoring protected visible tests: %s",
                        candidate.rollout_id,
                        test_edit_policy_reason,
                    )
                    candidate = sanitized_candidate
                    test_edit_policy_reason = self._candidate_test_edit_policy_reason(
                        candidate,
                        issue_plan=issue_plan,
                    )
                if test_edit_policy_reason and test_edit_policy_reason.startswith(
                    "unexpected visible test edits:"
                ):
                    logger.info(
                        "Pruned rollout %s due to visible-test policy violation: %s",
                        candidate.rollout_id,
                        test_edit_policy_reason,
                    )
                    self._prune_results[candidate.rollout_id] = {
                        "is_valid": False,
                        "reason": "unexpected_visible_test_edits",
                        "regressed_tests": [],
                        "still_passing": [],
                    }
                    continue

            if (
                baseline is not None
                and test_command
                and hasattr(self.verifier, "prune_by_regression")
            ):
                if self._candidate_full_scope_quick_pass_preserves_expected(candidate):
                    self._prune_results[candidate.rollout_id] = PruneResult(
                        is_valid=True,
                        regressed_tests=[],
                        still_passing=[],
                        reason="quick_verification_accepted",
                    )
                    valid_candidates.append(candidate)
                    continue
                prune_result = self._prune_candidate_by_regression(
                    candidate,
                    baseline,
                    test_command,
                )
                self._prune_results[candidate.rollout_id] = prune_result
                if not prune_result.is_valid:
                    logger.info(
                        "Pruned rollout %s due to regressions: %s",
                        candidate.rollout_id,
                        ", ".join(prune_result.regressed_tests) or prune_result.reason,
                    )
                    continue

            valid_candidates.append(candidate)
        return valid_candidates

    def _allow_test_only_patches(
        self,
        issue_plan: Optional["IssuePlan"],
    ) -> bool:
        """Whether the current selection round may accept test-only patches.

        Phase 2.3: the codegen-with-tests selector path requires a
        non-test source change (rejecting patches that only edit test
        files). The testgen-with-fix path legitimately ships test-only
        patches and must opt in.

        The selector itself runs in both modes (it's a shared library),
        so we infer the mode from the issue plan's allocator features.
        Specifically, ``is_completion_task`` issues that authorize
        ``incomplete_test_files`` indicate that test-only edits may be
        the entire fix (the task is "fill in this test scaffold"). When
        in doubt we default to strict mode — false negatives are
        recoverable through the residual-followup path; false positives
        ship empty-source noise as if it were a real fix.
        """
        if issue_plan is None:
            return False
        try:
            allocator_features = dict(getattr(issue_plan, "allocator_features", {}) or {})
        except (TypeError, AttributeError):
            allocator_features = {}
        if bool(allocator_features.get("testgen_with_fix")):
            return True
        # Allow when the issue explicitly authorizes test-file edits via
        # incomplete_test_files (completion-style scaffolds).
        try:
            test_context = getattr(issue_plan, "test_context", None)
            incomplete_tests = list(getattr(test_context, "incomplete_test_files", []) or [])
        except (TypeError, AttributeError):
            incomplete_tests = []
        return bool(incomplete_tests)

    def _candidate_has_valid_syntax(self, candidate: RolloutResult) -> bool:
        if not hasattr(self.verifier, "_check_syntax"):
            return True
        if not candidate.worktree_path:
            return True
        worktree = Path(candidate.worktree_path)
        changed_files = self._candidate_changed_files(candidate)
        return self.verifier._check_syntax(worktree, changed_files)

    def _verify_single_candidate(
        self,
        candidate: RolloutResult,
        test_command: Optional[str],
        *,
        baseline_result: Optional[Any] = None,
        issue_plan: Optional["IssuePlan"] = None,
    ) -> Optional[RolloutResult]:
        verification = self._verify_candidate_or_confirmed_handoff(
            candidate,
            test_command,
            baseline_result=baseline_result,
            issue_plan=issue_plan,
        )
        if candidate.rollout_id in self._prune_results:
            verification.prune_result = self._serialize_prune_result(
                self._prune_results[candidate.rollout_id]
            )
        self._apply_candidate_stub_residue_gate(candidate, verification)
        self._reconcile_verification_with_candidate_handoff(candidate, verification)
        self._reconcile_candidate_validity_with_verification(candidate, verification)
        verification.accepted = self._verification_meets_acceptance_bar(
            candidate,
            verification,
            test_command,
            issue_plan=issue_plan,
        )
        self._apply_evidence_bound_review(candidate, verification, issue_plan=issue_plan)
        candidate.verification = verification.to_dict()
        cluster = PatchCluster(
            cluster_id=0,
            patches=[candidate],
            verification=verification,
            critic_weight=self.config.selection.critic_weight,
        )
        # Decisive-Edge C.4: ``_apply_selection_critic`` itself honours
        # both ``use_critic`` and ``enable_critic_reranking``, so calling
        # it unconditionally here is safe (it returns immediately when
        # the gates are off) and lets the gate live in one place.
        self._apply_selection_critic([cluster], [candidate], issue_plan=issue_plan)
        self._refresh_cluster_selection_context(
            [cluster],
            test_command=test_command,
            issue_plan=issue_plan,
        )
        candidate.selection_diagnostics = self._cluster_selection_diagnostics(cluster)
        policy_reason = self._candidate_test_edit_policy_reason(
            candidate,
            issue_plan=issue_plan,
            verification=verification,
        )
        if policy_reason:
            logger.warning(
                "Rejected only candidate rollout %s due to test-edit policy: %s",
                candidate.rollout_id,
                policy_reason,
            )
            return None
        if not verification.accepted:
            candidate.selected_for_submission = False
            candidate.internally_accepted = False
            candidate.salvaged_for_external_scoring = False
            verification_payload = dict(candidate.verification or {})
            verification_payload.update(
                {
                    "accepted": False,
                    "internally_accepted": False,
                    "selected_for_submission": False,
                    "selection_authority": "internal_verifier_rejected",
                }
            )
            candidate.verification = verification_payload
            logger.warning(
                "Only candidate rollout %s did not meet the acceptance bar.",
                candidate.rollout_id,
            )
            return candidate
        if (
            test_command
            and verification.test_result
            and verification.test_result.pass_rate < self.config.selection.min_test_pass_rate
        ):
            logger.warning("Only candidate patch failed the minimum pass-rate threshold.")
        candidate.selected_for_submission = True
        candidate.internally_accepted = True
        candidate.salvaged_for_external_scoring = False
        verification_payload = dict(candidate.verification or {})
        verification_payload.update(
            {
                "internally_accepted": True,
                "selected_for_submission": True,
                "salvaged_for_external_scoring": False,
                "selection_authority": "internal_verifier",
            }
        )
        candidate.verification = verification_payload
        return candidate

    def _finalize_cluster_selection(
        self,
        cluster: PatchCluster,
        test_command: Optional[str],
    ) -> RolloutResult:
        best_patch = cluster.representative
        if cluster.verification:
            best_patch.verification = cluster.verification.to_dict()
        best_patch.selection_diagnostics = self._cluster_selection_diagnostics(cluster)
        accepted = bool(cluster.verification and cluster.verification.accepted)
        best_patch.selected_for_submission = accepted
        best_patch.internally_accepted = accepted
        best_patch.salvaged_for_external_scoring = False
        verification_payload = dict(best_patch.verification or {})
        verification_payload.update(
            {
                "internally_accepted": accepted,
                "selected_for_submission": accepted,
                "salvaged_for_external_scoring": False,
                "selection_authority": (
                    "internal_verifier" if accepted else "internal_verifier_rejected"
                ),
            }
        )
        best_patch.verification = verification_payload

        if test_command and cluster.verification:
            test_result = cluster.verification.test_result
            if test_result and test_result.pass_rate < self.config.selection.min_test_pass_rate:
                logger.warning(
                    "Best patch is below the configured pass-rate threshold: %.2f",
                    test_result.pass_rate,
                )
        return best_patch

    def _apply_selection_critic(
        self,
        clusters: list[PatchCluster],
        all_candidates: list[RolloutResult],
        *,
        issue_plan: Optional["IssuePlan"],
    ) -> None:
        # Decisive-Edge C.4: ``use_critic=False`` is the call-site gate —
        # when off, the SelectionCritic is NEVER constructed, no LLM is
        # invoked, and the selector falls back to verifier-only ranking
        # (pass-rate then lowest test-edit count). See SelectionConfig
        # docstring for the rationale; the A/B harness in
        # ``apex/scripts/ab_critic.py`` flips this per-arm.
        if not getattr(self.config.selection, "use_critic", True):
            return
        if not self.config.selection.enable_critic_reranking:
            return
        critic = SelectionCritic(self)
        for cluster in clusters:
            assessment = critic.assess_cluster(
                cluster,
                issue_plan=issue_plan,
                all_candidates=all_candidates,
            )
            cluster.critic_score = assessment.score
            cluster.critic_summary = assessment.summary
            cluster.critic_focus_files = list(assessment.focus_files)
            cluster.critic_features = dict(assessment.feature_scores)
            cluster.critic_weight = self.config.selection.critic_weight
            cluster.representative.selection_diagnostics = self._cluster_selection_diagnostics(
                cluster
            )

    def _apply_evidence_bound_review(
        self,
        candidate: RolloutResult,
        verification: VerificationResult,
        *,
        issue_plan: Optional["IssuePlan"],
    ) -> None:
        taxonomy = classify_candidate_verification(candidate, verification)
        verification.verification_taxonomy = taxonomy.to_dict()
        verification.repair_policy = taxonomy.repair_policy

        review = review_candidate(candidate, verification)
        if review.veto and verification.accepted:
            verification.accepted = False
            reason = "adversarial_review_veto: " + "; ".join(review.reasons[:3])
            if reason not in verification.validity_reasons:
                verification.validity_reasons.append(reason)
            taxonomy = classify_candidate_verification(candidate, verification)
            verification.verification_taxonomy = taxonomy.to_dict()
            verification.repair_policy = taxonomy.repair_policy

        # WS3C: fresh-context LLM final-acceptance reviewer (DEFAULT OFF). Layered
        # ON TOP of the deterministic veto and ONLY consulted for an already-
        # accepted candidate, so it can only DOWNGRADE (never upgrade). It fails
        # open (a reviewer error leaves acceptance unchanged).
        reviewer_verdict_dict: Optional[dict[str, Any]] = None
        if (
            self.config.selection.enable_final_acceptance_reviewer
            and self.final_acceptance_reviewer is not None
            and verification.accepted
        ):
            issue_description = ""
            if issue_plan is not None:
                issue_description = str(getattr(issue_plan, "summary", "") or "")
            verdict = self.final_acceptance_reviewer.review(
                candidate,
                verification,
                issue_description=issue_description,
            )
            reviewer_verdict_dict = verdict.to_dict()
            if not verdict.accept and not verdict.failed_open:
                verification.accepted = False
                reason = "final_acceptance_reviewer_reject: " + (verdict.reason or "")[:200]
                if reason not in verification.validity_reasons:
                    verification.validity_reasons.append(reason)
                taxonomy = classify_candidate_verification(candidate, verification)
                verification.verification_taxonomy = taxonomy.to_dict()
                verification.repair_policy = taxonomy.repair_policy

        ledger = build_candidate_evidence_ledger(candidate, verification=verification)
        process_quality = score_process_quality(candidate, verification)
        clarification = assess_clarification_need(
            issue_plan=issue_plan,
            verification_taxonomy=verification.verification_taxonomy,
            evidence_ledger=ledger.to_dict(),
        )
        # WS3I: clarification-abstain arm (DEFAULT OFF). When enabled, an
        # accepted candidate whose evidence is inconclusive (scorer disagreement /
        # unobserved expected inventory) is downgraded to not-accepted with a
        # validity reason, so the orchestrator abstains rather than shipping an
        # under-evidenced patch. Never UPGRADES; only ever flips accepted->False.
        clarification_abstained = False
        if (
            clarification_abstain_enabled(self.config)
            and clarification.should_abstain
            and verification.accepted
        ):
            verification.accepted = False
            reason = "clarification_abstain: " + (clarification.action or "needs_clarification")
            if reason not in verification.validity_reasons:
                verification.validity_reasons.append(reason)
            clarification_abstained = True
        diagnostics = (
            dict(candidate.selection_diagnostics)
            if isinstance(candidate.selection_diagnostics, dict)
            else {}
        )
        diagnostics.update(
            {
                "evidence_ledger": ledger.to_dict(),
                "adversarial_review": review.to_dict(),
                "process_quality": dict(process_quality),
                "verification_taxonomy": dict(verification.verification_taxonomy),
                "clarification_assessment": clarification.to_dict(),
                "clarification_abstained": clarification_abstained,
            }
        )
        if reviewer_verdict_dict is not None:
            diagnostics["final_acceptance_reviewer"] = reviewer_verdict_dict
        candidate.selection_diagnostics = diagnostics

    def _refresh_cluster_selection_context(
        self,
        clusters: list[PatchCluster],
        *,
        test_command: Optional[str],
        issue_plan: Optional["IssuePlan"] = None,
    ) -> None:
        for cluster in clusters:
            evidence_mode, verification_authority = self._cluster_evidence_mode(
                cluster,
                test_command=test_command,
            )
            cluster.evidence_mode = evidence_mode
            cluster.verification_authority = verification_authority
            cluster.public_signal_score = self._cluster_public_signal_score(cluster)
            cluster.backend_anomaly_penalty = self._cluster_backend_anomaly_penalty(cluster)
            ledger = build_candidate_evidence_ledger(
                cluster.representative,
                verification=cluster.verification,
            )
            process_quality = score_process_quality(cluster.representative, cluster.verification)
            review = review_candidate(cluster.representative, cluster.verification)
            taxonomy = classify_candidate_verification(
                cluster.representative,
                cluster.verification,
            )
            clarification = assess_clarification_need(
                issue_plan=issue_plan,
                verification_taxonomy=taxonomy.to_dict(),
                evidence_ledger=ledger.to_dict(),
            )
            cluster.evidence_ledger_score = ledger.score
            cluster.evidence_ledger = ledger.to_dict()
            cluster.process_quality_score = float(process_quality.get("score") or 0.0)
            cluster.process_quality = dict(process_quality)
            cluster.adversarial_risk_score = review.risk_score
            cluster.adversarial_review = review.to_dict()
            cluster.verification_taxonomy = taxonomy.to_dict()
            cluster.clarification_assessment = clarification.to_dict()
            cluster.representative.selection_diagnostics = self._cluster_selection_diagnostics(
                cluster
            )

    def _cluster_selection_diagnostics(self, cluster: PatchCluster) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "ranking": cluster.ranking_details(),
        }
        existing = (
            dict(cluster.representative.selection_diagnostics)
            if isinstance(cluster.representative.selection_diagnostics, dict)
            else {}
        )
        for key in ("stub_residue", "stub_residue_advisory"):
            if key in existing:
                diagnostics[key] = copy.deepcopy(existing[key])
        component_ablation = self._cluster_component_ablation(cluster)
        if component_ablation:
            diagnostics["component_ablation"] = component_ablation
        # WS3I: surface the behavioral-arm gate state for telemetry (gated-off
        # runs emit {enabled: False}).
        diagnostics["behavioral_arms"] = behavioral_arms_summary(self.config)
        # Decisive-Edge C.4: hide critic diagnostics when the call-site
        # gate is off so downstream consumers can use the diagnostic
        # surface to detect "the critic ran" vs. "the critic was
        # disabled" without scraping the config.
        critic_active = (
            getattr(self.config.selection, "use_critic", True)
            and self.config.selection.enable_critic_reranking
        )
        if critic_active:
            diagnostics["critic"] = {
                "score": cluster.critic_score,
                "summary": cluster.critic_summary,
                "focus_files": list(cluster.critic_focus_files),
                "features": dict(cluster.critic_features),
            }
        if cluster.evidence_ledger:
            diagnostics["evidence_ledger"] = dict(cluster.evidence_ledger)
        if cluster.adversarial_review:
            diagnostics["adversarial_review"] = dict(cluster.adversarial_review)
        if cluster.process_quality:
            diagnostics["process_quality"] = dict(cluster.process_quality)
        if cluster.verification_taxonomy:
            diagnostics["verification_taxonomy"] = dict(cluster.verification_taxonomy)
        if cluster.clarification_assessment:
            diagnostics["clarification_assessment"] = dict(cluster.clarification_assessment)
        anomaly = self._candidate_backend_anomaly(cluster.representative)
        if anomaly:
            diagnostics["backend_anomaly"] = dict(anomaly)
        return diagnostics

    def _cluster_evidence_mode(
        self,
        cluster: PatchCluster,
        *,
        test_command: Optional[str],
    ) -> tuple[str, str]:
        if self._cluster_has_authoritative_test_evidence(
            cluster,
            test_command=test_command,
        ):
            return "authoritative", "verifier_test_command"

        representative = cluster.representative
        quick_verification = (
            representative.quick_verification
            if isinstance(representative.quick_verification, dict)
            else {}
        )
        public_signal = (
            representative.search_metadata.get("public_signal")
            if isinstance(representative.search_metadata, dict)
            else None
        )
        if quick_verification or isinstance(public_signal, dict):
            return "weak_public", "rollout_public_signal"
        return "structural_only", "non_authoritative_structural_proxy"

    def _cluster_has_authoritative_test_evidence(
        self,
        cluster: PatchCluster,
        *,
        test_command: Optional[str],
    ) -> bool:
        if not str(test_command or "").strip():
            return False
        verification = cluster.verification
        if verification is None:
            return False
        if verification.accepted:
            return True
        test_result = verification.test_result
        if test_result is None:
            return False
        if bool(test_result.reproduction_passes or test_result.regression_passes):
            return True
        if bool(getattr(test_result, "regression_inconclusive", False)):
            return True
        if any(
            int(getattr(test_result, field, 0) or 0) > 0 for field in ("passed", "failed", "errors")
        ):
            return True
        if int(getattr(test_result, "expected_test_count", 0) or 0) > 0:
            return True
        if int(getattr(test_result, "collected_test_count", 0) or 0) > 0:
            return True
        return bool(str(getattr(test_result, "test_inventory_source", "") or "").strip())

    def _cluster_public_signal_score(self, cluster: PatchCluster) -> float:
        representative = cluster.representative
        quick_verification = (
            representative.quick_verification
            if isinstance(representative.quick_verification, dict)
            else {}
        )
        quick_signal = quick_verification_signal_score(quick_verification)
        progress_score = getattr(representative, "progress_score", 0.0)
        if not isinstance(progress_score, (int, float)):
            progress_score = 0.0
        patch_confidence = self._candidate_patch_confidence(representative)

        score = 0.0
        if isinstance(quick_signal, (int, float)):
            score += 0.60 * max(0.0, min(float(quick_signal), 1.0))
        score += 0.25 * max(0.0, min(float(progress_score), 1.0))
        score += 0.15 * max(0.0, min(float(patch_confidence), 1.0))

        public_signal = (
            representative.search_metadata.get("public_signal")
            if isinstance(representative.search_metadata, dict)
            else None
        )
        if isinstance(public_signal, dict):
            signal_score = public_signal.get("score")
            if isinstance(signal_score, (int, float)):
                score = max(score, float(signal_score))

        if quick_verification_has_local_full_scope_pass(quick_verification):
            score = max(
                score,
                0.82
                + (
                    0.13
                    * max(
                        0.0,
                        min(float(quick_signal or 1.0), 1.0),
                    )
                ),
            )

        failed = quick_verification.get("failed")
        errors = quick_verification.get("errors")
        returncode = quick_verification.get("returncode")
        scored_expected_suite_pass = quick_verification_has_scored_expected_suite_pass(
            quick_verification
        )
        if (
            isinstance(failed, int)
            and failed > 0
            or isinstance(errors, int)
            and errors > 0
            or isinstance(returncode, int)
            and returncode != 0
            and not scored_expected_suite_pass
        ):
            score = min(score, 0.35)
        if bool(quick_verification.get("timed_out")) or bool(
            quick_verification.get("full_scope_timed_out")
        ):
            score = min(score, 0.45)
        if quick_verification.get("scope") == "structural_precheck" and not bool(
            quick_verification.get("structural_recovered")
        ):
            score = min(score, 0.15)
        return max(0.0, min(score, 1.0))

    def _candidate_patch_confidence(self, candidate: RolloutResult) -> float:
        patch_artifact = (
            candidate.patch_artifact if isinstance(candidate.patch_artifact, dict) else {}
        )
        confidence = patch_artifact.get("confidence")
        if isinstance(confidence, (int, float)):
            return max(0.0, min(float(confidence), 1.0))
        return 0.0

    def _candidate_backend_anomaly(self, candidate: RolloutResult) -> dict[str, Any]:
        if isinstance(candidate.search_metadata, dict):
            payload = candidate.search_metadata.get("backend_anomaly")
            if isinstance(payload, dict):
                return dict(payload)

        patch_artifact = (
            candidate.patch_artifact if isinstance(candidate.patch_artifact, dict) else {}
        )
        payload = patch_artifact.get("_apex_backend_anomaly")
        if isinstance(payload, dict):
            return dict(payload)

        if patch_artifact.get("_apex_recovered_patch"):
            return {
                "kind": "cli_finalization_failure",
                "severity": "low",
                "reason": str(patch_artifact.get("_apex_recovery_reason") or "").strip(),
                "recovered_submission": True,
            }

        for entry in list(candidate.trajectory or []):
            if not isinstance(entry, dict):
                continue
            timeout_audit = entry.get("timeout_audit")
            if not isinstance(timeout_audit, dict):
                continue
            terminal_state = str(timeout_audit.get("terminal_state") or "").strip()
            if not terminal_state:
                continue
            severity = "moderate"
            if terminal_state == "recovered_after_timeout":
                severity = "low"
            elif terminal_state == "policy_violation":
                severity = "high"
                if bool((timeout_audit.get("policy_violation") or {}).get("likely_backend_helper")):
                    severity = "moderate"
            return {
                "kind": terminal_state,
                "severity": severity,
                "recovered_submission": bool(entry.get("recovered_submission")),
            }
        return {}

    def _cluster_backend_anomaly_penalty(self, cluster: PatchCluster) -> float:
        anomaly = self._candidate_backend_anomaly(cluster.representative)
        if not anomaly:
            return 0.0
        severity = str(anomaly.get("severity") or "").strip().lower()
        penalty = {
            "low": 0.06,
            "moderate": 0.12,
            "high": 0.18,
        }.get(severity, 0.10)
        if bool(anomaly.get("recovered_candidate")):
            penalty = max(penalty, 0.14)
        if bool(anomaly.get("recovered_submission")):
            penalty = max(0.06, penalty)
        return max(0.0, min(penalty, 0.25))

    def _attempt_patch_synthesis(
        self,
        clusters: list[PatchCluster],
        all_candidates: list[RolloutResult],
        test_command: Optional[str],
        *,
        issue_plan: Optional["IssuePlan"] = None,
    ) -> Optional[PatchCluster]:
        if (
            not self.config.selection.enable_patch_synthesis
            or len(clusters) < 2
            or not Path(self.repo_path).exists()
        ):
            return None

        ranked = sorted(
            clusters,
            key=lambda cluster: (
                cluster.accepted,
                cluster.verification_score,
                cluster.cross_validation_score,
                cluster.size,
                cluster.cluster_id,
            ),
            reverse=True,
        )[: max(2, self.config.selection.max_synthesis_candidates)]
        combinations = self._synthesis_combinations(ranked)
        # TIER 2 (T2.5): N-way greedy union. The legacy pairs/triples path
        # (above) is preserved; the greedy union is an ADDITIONAL ladder of
        # prefix unions built from a LARGER, coverage-ordered seed pool so a
        # 6-to-8-way union of disjoint module-group partials can form (the
        # legacy path caps coverage at 3 contributors). The verifier remains
        # authoritative: each union is verified by the same path and selected
        # only if it beats every component.
        if self.config.selection.enable_greedy_synthesis_union:
            for ladder_combo in self._greedy_synthesis_union(clusters):
                signatures = {
                    tuple(sorted(item.rollout_id for item in combo)) for combo in combinations
                }
                signature = tuple(sorted(item.rollout_id for item in ladder_combo))
                if signature not in signatures:
                    combinations.append(ladder_combo)
        if not combinations:
            return None

        reproduction_artifacts = [
            candidate.reproduction_artifact
            for candidate in all_candidates
            if candidate.reproduction_artifact
        ]
        best_cluster: Optional[PatchCluster] = None
        for combo_index, combo in enumerate(combinations):
            candidate = self._build_synthetic_candidate(combo, combo_index)
            if candidate is None:
                continue
            cluster = PatchCluster(
                cluster_id=max((item.cluster_id for item in clusters), default=-1)
                + combo_index
                + 1,
                patches=[candidate],
                signature=hashlib.sha256(candidate.patch.encode()).hexdigest()[:20]
                if candidate.patch
                else "",
                payload=self._cluster_payload(candidate),
            )
            verification, cross_validation_score = self._build_cluster_verification(
                cluster,
                test_command=test_command,
                matrix=None,
                index_by_rollout={},
                reproduction_artifacts=reproduction_artifacts,
                issue_plan=issue_plan,
            )
            cluster.verification = verification
            cluster.cross_validation_score = cross_validation_score
            self._apply_evidence_bound_review(candidate, verification, issue_plan=issue_plan)
            candidate.verification = verification.to_dict()
            self._refresh_cluster_selection_context(
                [cluster],
                test_command=test_command,
                issue_plan=issue_plan,
            )
            if best_cluster is None or cluster.combined_score > best_cluster.combined_score:
                best_cluster = cluster

        if best_cluster is None:
            return None

        logger.info(
            "Synthesized compatible candidate from rollouts %s (accepted=%s, score=%.2f)",
            ", ".join(str(item) for item in best_cluster.representative.source_rollout_ids),
            best_cluster.accepted,
            best_cluster.combined_score,
        )
        return best_cluster

    def _synthesis_combinations(self, clusters: list[PatchCluster]) -> list[list[RolloutResult]]:
        representatives = [
            cluster.representative
            for cluster in clusters
            if cluster.representative.worktree_path
            and Path(cluster.representative.worktree_path).exists()
            and self._candidate_changed_files(cluster.representative)
        ]
        combinations: list[list[RolloutResult]] = []
        seen: set[tuple[int, ...]] = set()
        max_combinations = max(1, self.config.selection.max_synthesis_combinations)

        for first_index, first in enumerate(representatives):
            for second in representatives[first_index + 1 :]:
                pair = [first, second]
                if not self._candidate_group_is_compatible(pair):
                    continue
                signature = tuple(sorted(item.rollout_id for item in pair))
                if signature not in seen:
                    combinations.append(pair)
                    seen.add(signature)
                if len(combinations) >= max_combinations:
                    return combinations

                for third in representatives:
                    if third in pair:
                        continue
                    triple = pair + [third]
                    signature = tuple(sorted(item.rollout_id for item in triple))
                    if signature in seen or not self._candidate_group_is_compatible(triple):
                        continue
                    combinations.append(triple)
                    seen.add(signature)
                    if len(combinations) >= max_combinations:
                        return combinations
        return combinations

    def _candidate_group_is_compatible(self, candidates: list[RolloutResult]) -> bool:
        if len(candidates) < 2:
            return False

        file_states: dict[str, tuple[bool, str]] = {}
        individual_counts: list[int] = []
        total_files = 0
        for candidate in candidates:
            worktree_path = candidate.worktree_path
            if not worktree_path:
                return False
            worktree = Path(worktree_path)
            candidate_files = self._candidate_changed_files(candidate)
            individual_counts.append(len(set(candidate_files)))
            for rel_path in candidate_files:
                total_files += 1
                state = self._candidate_file_state(worktree, rel_path)
                existing = file_states.get(rel_path)
                if existing is not None and existing != state:
                    return False
                file_states[rel_path] = state
        if not (len(file_states) >= 2 and len(file_states) > max(individual_counts or [0])):
            return False
        if not self._candidate_group_has_disjoint_repair_evidence(candidates):
            return False
        # Final guard: AST cross-file conflict detection. Two patches that
        # touch *different* files can still break each other if one removes
        # a name the other imports/calls. Bail at >4 members because the
        # check is O(n^2) and synthesis combos are bounded anyway.
        if len(candidates) > 4:
            return False
        conflict_reason = self._synthesis_group_has_cross_file_conflict(
            candidates,
            Path(self.repo_path),
        )
        if conflict_reason is not None:
            logger.info(
                "Synthesis group rejected (cross-file conflict): %s",
                conflict_reason,
            )
            return False
        return True

    def _candidate_new_passing_test_ids(
        self,
        candidate: RolloutResult,
    ) -> set[str]:
        """Return the set of passing test ids the candidate's QV observed (T2.5).

        Used to order the greedy union by coverage-contribution (new passing
        expected-tests added). Falls back to an empty set when the QV payload
        has no per-id passing list.
        """
        quick = (
            candidate.quick_verification if isinstance(candidate.quick_verification, dict) else {}
        )
        passing = {
            str(value)
            for value in list(quick.get("passed_tests") or [])
            + list(quick.get("matched_expected_test_ids") or [])
            if str(value).strip()
        }
        return passing

    def _candidate_qv_is_indeterminate(self, candidate: RolloutResult) -> bool:
        """Whether a candidate's QV gives NO usable per-id coverage signal (T2.5).

        A harness/launch failure (E2BIG) is explicitly indeterminate, and so is a
        payload with no per-id passing/failing lists AND no executed counts. Such
        a member contributes 0 to the greedy union's coverage delta only because
        its signal is missing, not because it did no work — so it must NOT be
        dropped purely on contribution<=0 (it still has to pass the file-disjoint
        + cross-file AST checks). Membership grants no acceptance: every union is
        independently re-verified downstream.
        """
        quick = (
            candidate.quick_verification if isinstance(candidate.quick_verification, dict) else {}
        )
        if not quick:
            return True
        if bool(quick.get("harness_indeterminate")):
            return True
        has_id_lists = bool(
            list(quick.get("passed_tests") or [])
            or list(quick.get("failed_tests") or [])
            or list(quick.get("matched_expected_test_ids") or [])
        )
        if has_id_lists:
            return False
        executed = 0
        for key in ("passed", "failed", "errors"):
            value = quick.get(key)
            if isinstance(value, int) and value > 0:
                executed += value
        return executed == 0

    def _greedy_synthesis_union(
        self,
        clusters: list[PatchCluster],
    ) -> list[list[RolloutResult]]:
        """Build a coverage-ordered, file-disjoint N-way union ladder (T2.5).

        Seeds a pool of ALL cluster representatives (ranked, capped at
        ``max_synthesis_pool`` — NOT truncated to the legacy top-6), orders by
        coverage-contribution (new passing expected-tests vs the accumulated
        union), and greedily adds a member iff it is file-disjoint from the
        accumulated set AND passes the incremental cross-file AST conflict check
        AND adds >= 1 new passing test. Emits a ladder of prefix unions
        (3-way, 5-way, ... up to ``max_synthesis_union_members``) so the
        verifier can keep the best-verifying prefix. Returns combos (lists of
        2+ representatives); the caller verifies each.
        """
        max_members = max(2, int(self.config.selection.max_synthesis_union_members or 12))
        pool_cap = max(max_members, int(self.config.selection.max_synthesis_pool or 24))
        representatives = [
            cluster.representative
            for cluster in sorted(
                clusters,
                key=lambda cluster: (
                    cluster.accepted,
                    cluster.verification_score,
                    cluster.cross_validation_score,
                    cluster.size,
                    cluster.cluster_id,
                ),
                reverse=True,
            )
            if cluster.representative.worktree_path
            and Path(cluster.representative.worktree_path).exists()
            and self._candidate_changed_files(cluster.representative)
        ][:pool_cap]
        if len(representatives) < 2:
            return []

        accumulated: list[RolloutResult] = []
        accumulated_files: set[str] = set()
        accumulated_passing: set[str] = set()
        remaining = list(representatives)
        ladder: list[list[RolloutResult]] = []

        while remaining and len(accumulated) < max_members:
            best_candidate: Optional[RolloutResult] = None
            best_contribution = 0
            best_new_files = 0
            best_is_determinate = 0
            for candidate in remaining:
                candidate_files = set(self._candidate_changed_files(candidate))
                if not candidate_files:
                    continue
                if candidate_files & accumulated_files:
                    # Not file-disjoint from the accumulated union.
                    continue
                new_passing = self._candidate_new_passing_test_ids(candidate) - accumulated_passing
                contribution = len(new_passing)
                indeterminate = self._candidate_qv_is_indeterminate(candidate)
                # Seed the union with the strongest first member even when the
                # QV payload lacks per-id passing lists (contribution == 0):
                # require a positive contribution only once the union is
                # non-empty AND the member has a usable (determinate) QV signal.
                # A member whose QV is INDETERMINATE (e.g. a decomposition module
                # group whose subset QV could not score) contributes 0 only
                # because its signal is missing — keep it as a file-disjoint
                # member rather than starving the union to a single rung.
                if accumulated and contribution <= 0 and not indeterminate:
                    continue
                if accumulated:
                    # Incremental AST conflict check: new member vs accumulated.
                    if not self._candidate_group_is_compatible_incremental(
                        accumulated + [candidate]
                    ):
                        continue
                # Prefer determinate coverage contribution; an indeterminate
                # member ranks below an equal-contribution determinate one.
                is_determinate = 0 if indeterminate else 1
                if best_candidate is None or (
                    contribution,
                    is_determinate,
                    len(candidate_files),
                ) > (best_contribution, best_is_determinate, best_new_files):
                    best_candidate = candidate
                    best_contribution = contribution
                    best_new_files = len(candidate_files)
                    best_is_determinate = is_determinate
            if best_candidate is None:
                break
            accumulated.append(best_candidate)
            accumulated_files |= set(self._candidate_changed_files(best_candidate))
            accumulated_passing |= self._candidate_new_passing_test_ids(best_candidate)
            remaining = [item for item in remaining if item is not best_candidate]
            # Emit a ladder rung at 2, 3, 5, 7, ... members so the verifier picks
            # the best-verifying prefix without us pre-committing to the full N.
            size = len(accumulated)
            if size >= 2 and (size <= 3 or size % 2 == 1):
                ladder.append(list(accumulated))
        # Always include the deepest (full) union as the final rung so the
        # maximal disjoint union is verified even when its size is even.
        if len(accumulated) >= 2 and (not ladder or ladder[-1] != accumulated):
            ladder.append(list(accumulated))
        return ladder

    def _candidate_group_is_compatible_incremental(
        self,
        candidates: list[RolloutResult],
    ) -> bool:
        """File-disjoint + cross-file AST compatibility WITHOUT the >4 cap (T2.5).

        Mirrors :meth:`_candidate_group_is_compatible` but relaxes the legacy
        ``>4`` member cap on the incremental greedy-union path only. The AST
        conflict check is the same authoritative cross-file analysis; it is run
        once per added member so the total cost stays O(n^2)-once across the
        whole ladder. The legacy pairs/triples path keeps its conservative cap.
        """
        if len(candidates) < 2:
            return False
        changed_sets = [set(self._candidate_changed_files(candidate)) for candidate in candidates]
        for index, current in enumerate(changed_sets):
            if not current:
                return False
            for other in changed_sets[index + 1 :]:
                if current & other:
                    return False
        conflict_reason = self._synthesis_group_has_cross_file_conflict(
            candidates,
            Path(self.repo_path),
        )
        if conflict_reason is not None:
            logger.info(
                "Greedy union member rejected (cross-file conflict): %s",
                conflict_reason,
            )
            return False
        return True

    def _candidate_group_has_disjoint_repair_evidence(
        self,
        candidates: list[RolloutResult],
    ) -> bool:
        changed_sets = [
            set(self._candidate_changed_files(candidate))
            for candidate in candidates
            if self._candidate_changed_files(candidate)
        ]
        if len(changed_sets) < len(candidates):
            return False
        for index, current in enumerate(changed_sets):
            for other in changed_sets[index + 1 :]:
                if current & other:
                    return False
        failure_sets: list[set[str]] = []
        for candidate in candidates:
            quick = (
                candidate.quick_verification
                if isinstance(candidate.quick_verification, dict)
                else {}
            )
            failures = {
                str(value)
                for value in list(quick.get("failed_tests") or [])
                + list(quick.get("error_tests") or [])
                + list(quick.get("missing_expected_test_ids") or [])
                if str(value).strip()
            }
            if failures:
                failure_sets.append(failures)
        if len(failure_sets) >= 2:
            return len({tuple(sorted(items)) for items in failure_sets}) > 1
        return True

    def _synthesis_baseline_for_file(
        self,
        repo_root: Path,
        rel_path: str,
    ) -> Optional[str]:
        """Return the baseline (HEAD) text of a tracked file, cached.

        Falls back to the on-disk content when ``git`` cannot be invoked
        (e.g., the repo is not a git checkout in tests). Returns ``None``
        if neither source is available.
        """
        cache_key = (str(repo_root), rel_path)
        if cache_key in self._baseline_text_cache:
            return self._baseline_text_cache[cache_key]
        baseline: Optional[str] = None
        completed = _run_selection_git(
            ["show", f"HEAD:{rel_path}"],
            cwd=repo_root,
            timeout=10,
        )
        if completed.returncode == 0:
            baseline = completed.stdout
        if baseline is None:
            on_disk = repo_root / rel_path
            if on_disk.exists() and on_disk.is_file():
                try:
                    baseline = on_disk.read_text(errors="replace")
                except OSError:
                    baseline = None
        self._baseline_text_cache[cache_key] = baseline
        return baseline

    def _synthesis_group_has_cross_file_conflict(
        self,
        candidates: list[RolloutResult],
        repo_root: Path,
    ) -> Optional[str]:
        """Detect an AST-level cross-file conflict between candidate edits.

        Conflict definition: candidate A *removes* a top-level name N from
        file F, and candidate B (touching a different file G) imports N
        from F or otherwise references it as a free Load. Returns a
        human-readable reason on conflict, else ``None``.

        Best-effort: if any AST parse fails, we conservatively return a
        reason — the caller treats that as "reject".
        """
        per_candidate: list[dict[str, Any]] = []
        for candidate in candidates:
            try:
                summary = self._summarize_candidate_files(candidate, repo_root)
            except _SynthesisAstError as exc:
                return f"AST parse failed: {exc}"
            per_candidate.append(summary)
        for index_a, summary_a in enumerate(per_candidate):
            for index_b, summary_b in enumerate(per_candidate):
                if index_a == index_b:
                    continue
                a_files = set(summary_a["files"].keys())
                b_files = set(summary_b["files"].keys())
                # Only consider B touching files A does NOT touch — the
                # same-file case is already handled by the file-hash check.
                only_b = b_files - a_files
                if not only_b:
                    continue
                a_removed_per_file = summary_a["removed_per_file"]
                if not any(a_removed_per_file.values()):
                    continue
                # For each B-only file, see if any used name (including
                # ``ImportFrom`` targets) matches a removed name in any
                # of A's touched files.
                for b_file in only_b:
                    b_info = summary_b["files"][b_file]
                    used_names: set[str] = set(b_info["used_names"])
                    for module, names in b_info["import_from"]:
                        used_names.update(names)
                    for a_file, removed in a_removed_per_file.items():
                        if not removed:
                            continue
                        # Same-package heuristic: the import-from check is
                        # restricted to imports whose module *could* refer
                        # to A's file (matches the file's module/name).
                        a_module_token = Path(a_file).stem
                        relevant_imports = {
                            name
                            for module, names in b_info["import_from"]
                            for name in names
                            if module == a_module_token or module.endswith("." + a_module_token)
                        }
                        free_uses = used_names & set(removed)
                        # If the name appears as a free Name(Load), or via
                        # an import-from from A's module, it's a conflict.
                        if relevant_imports & set(removed):
                            offender = next(iter(relevant_imports & set(removed)))
                            return (
                                f"{a_file} removes {offender!r} but {b_file} "
                                f"imports it from {a_module_token}"
                            )
                        if free_uses:
                            offender = next(iter(free_uses))
                            return f"{a_file} removes {offender!r} but {b_file} references it"
        return None

    def _summarize_candidate_files(
        self,
        candidate: RolloutResult,
        repo_root: Path,
    ) -> dict[str, Any]:
        worktree = Path(candidate.worktree_path or "")
        result: dict[str, Any] = {
            "files": {},
            "removed_per_file": {},
        }
        for rel_path in self._candidate_changed_files(candidate):
            if not rel_path.endswith(".py"):
                continue
            patched = worktree / rel_path
            if not patched.exists() or not patched.is_file():
                continue
            try:
                patched_text = patched.read_text(errors="replace")
                patched_summary = _summarize_python_module(patched_text)
            except (OSError, SyntaxError) as exc:
                raise _SynthesisAstError(f"{rel_path}: {exc}") from exc
            baseline_text = self._synthesis_baseline_for_file(repo_root, rel_path)
            removed: set[str] = set()
            if baseline_text is not None:
                try:
                    baseline_summary = _summarize_python_module(baseline_text)
                except SyntaxError as exc:
                    raise _SynthesisAstError(f"baseline {rel_path}: {exc}") from exc
                removed = baseline_summary["defined_names"] - patched_summary["defined_names"]
            result["files"][rel_path] = patched_summary
            result["removed_per_file"][rel_path] = removed
        return result

    def _candidate_file_state(
        self,
        worktree: Path,
        rel_path: str,
    ) -> tuple[bool, str]:
        target = worktree / rel_path
        if not target.exists():
            return (False, "")
        return (True, hashlib.sha256(target.read_bytes()).hexdigest())

    def _build_synthetic_candidate(
        self,
        candidates: list[RolloutResult],
        combo_index: int,
    ) -> Optional[RolloutResult]:
        workspace = self._prepare_synthesis_workspace(combo_index)
        if workspace is None:
            return None

        changed_files = sorted(
            {
                rel_path
                for candidate in candidates
                for rel_path in self._candidate_changed_files(candidate)
            }
        )
        if not changed_files:
            return None

        source_by_path: dict[str, RolloutResult] = {}
        for candidate in candidates:
            for rel_path in self._candidate_changed_files(candidate):
                source_by_path.setdefault(rel_path, candidate)

        for rel_path in changed_files:
            source = source_by_path[rel_path]
            source_path = Path(source.worktree_path or "") / rel_path
            destination = workspace / rel_path
            if source_path.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(source_path, destination)
            else:
                if destination.is_dir():
                    shutil.rmtree(destination, ignore_errors=True)
                else:
                    destination.unlink(missing_ok=True)

        add_result = _run_selection_git(
            ["add", "-A"],
            cwd=workspace,
            timeout=_SELECTION_GIT_WORKTREE_REMOVE_TIMEOUT_SECONDS,
        )
        if add_result.returncode != 0:
            return None
        diff_result = _run_selection_git(
            ["diff", "--binary", "--relative", "HEAD", "--", "."],
            cwd=workspace,
            timeout=_SELECTION_GIT_WORKTREE_REMOVE_TIMEOUT_SECONDS,
        )
        diff = diff_result.stdout if diff_result.returncode == 0 else ""
        if not diff.strip():
            return None

        source_ids = sorted({candidate.rollout_id for candidate in candidates})
        return RolloutResult(
            rollout_id=-(combo_index + 1),
            success=True,
            patch=diff,
            explanation=(
                "Synthesized compatible edits from rollouts "
                + ", ".join(str(item) for item in source_ids)
            ),
            changed_files=changed_files,
            worktree_path=str(workspace),
            llm_model="apex_synthesizer",
            plan_title="Synthetic merge",
            is_synthetic=True,
            source_rollout_ids=source_ids,
        )

    def _prepare_synthesis_workspace(self, combo_index: int) -> Optional[Path]:
        return self._prepare_ephemeral_workspace(prefix=f"apex-synth-{combo_index}-")

    def _prepare_ephemeral_workspace(
        self,
        *,
        prefix: str,
    ) -> Optional[Path]:
        repo_path = Path(self.repo_path)
        if not repo_path.exists() or not repo_path.is_dir():
            return None

        temp_dir: Optional[str] = None
        workspace_root_text = str(getattr(self.config, "workspace_dir", "") or "").strip()
        if workspace_root_text:
            workspace_root = Path(workspace_root_text)
            try:
                workspace_root.mkdir(parents=True, exist_ok=True)
                temp_dir = str(workspace_root)
            except OSError as exc:
                logger.debug(
                    "Failed to create configured selection workspace %s: %s",
                    workspace_root,
                    exc,
                )
        temp_root = Path(tempfile.mkdtemp(prefix=prefix, dir=temp_dir))
        workspace = temp_root / "workspace"
        snapshot_ignore = shutil.ignore_patterns(
            ".git",
            ".hg",
            ".jj",
            ".sl",
            ".svn",
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            ".mypy_cache",
        )
        mode = "snapshot"
        if self._repo_can_use_git_snapshot_clone(repo_path):
            cloned = clone_git_repo_with_overlay(
                repo_path,
                workspace,
                ignore=snapshot_ignore,
            )
            if cloned:
                mode = "git_clone"
        if not workspace.exists():
            copy_tree(
                repo_path,
                workspace,
                ignore=snapshot_ignore,
            )
            _run_selection_git(["init"], cwd=workspace, timeout=60)
            _run_selection_git(
                ["config", "user.email", "apex@example.com"],
                cwd=workspace,
                timeout=_SELECTION_GIT_QUICK_TIMEOUT_SECONDS,
            )
            _run_selection_git(
                ["config", "user.name", "APEX"],
                cwd=workspace,
                timeout=_SELECTION_GIT_QUICK_TIMEOUT_SECONDS,
            )
            _run_selection_git(
                ["add", "-A"],
                cwd=workspace,
                timeout=_SELECTION_GIT_WORKTREE_REMOVE_TIMEOUT_SECONDS,
            )
            _run_selection_git(
                ["commit", "--no-gpg-sign", "-m", "APEX synthesis baseline"],
                cwd=workspace,
                timeout=_SELECTION_GIT_WORKTREE_REMOVE_TIMEOUT_SECONDS,
            )
        self._ephemeral_worktrees.append(workspace)
        self._ephemeral_workspace_modes[workspace.resolve()] = mode
        return workspace

    def _sanitize_candidate_visible_test_edits(
        self,
        candidate: RolloutResult,
        *,
        issue_plan: Optional["IssuePlan"],
    ) -> Optional[RolloutResult]:
        analyses = self._candidate_visible_test_edit_dispositions(
            candidate,
            issue_plan=issue_plan,
        )
        unexpected_test_changes = sorted(
            path
            for path, analysis in analyses.items()
            if analysis.action in {"restore", "sanitize"}
        )
        if not unexpected_test_changes:
            return candidate
        if not candidate.worktree_path:
            return None

        workspace = self._prepare_ephemeral_workspace(
            prefix=f"apex-policy-{candidate.rollout_id}-",
        )
        if workspace is None:
            return None

        source_root = Path(candidate.worktree_path)
        restored_protected_test_files: list[str] = []
        sanitized_protected_test_files: list[str] = []
        for rel_path in self._candidate_changed_files(candidate):
            analysis = analyses.get(rel_path)
            if analysis is not None and analysis.action == "restore":
                restored_protected_test_files.append(rel_path)
                continue
            destination = workspace / rel_path
            if analysis is not None and analysis.action == "sanitize":
                if not analysis.sanitized_text:
                    restored_protected_test_files.append(rel_path)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(analysis.sanitized_text)
                sanitized_protected_test_files.append(rel_path)
                continue
            source_path = source_root / rel_path
            if source_path.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(source_path, destination)
            else:
                if destination.is_dir():
                    shutil.rmtree(destination, ignore_errors=True)
                else:
                    destination.unlink(missing_ok=True)

        add_result = _run_selection_git(
            ["add", "-A"],
            cwd=workspace,
            timeout=_SELECTION_GIT_WORKTREE_REMOVE_TIMEOUT_SECONDS,
        )
        if add_result.returncode != 0:
            return None
        diff_result = _run_selection_git(
            ["diff", "--binary", "--relative", "HEAD", "--", "."],
            cwd=workspace,
            timeout=_SELECTION_GIT_WORKTREE_REMOVE_TIMEOUT_SECONDS,
        )
        diff = diff_result.stdout if diff_result.returncode == 0 else ""
        if not diff.strip():
            return None

        materialized_changed_files = expand_changed_paths(
            workspace,
            list_git_changed_files(workspace),
        ) or [
            path
            for path in self._candidate_changed_files(candidate)
            if path not in set(restored_protected_test_files)
        ]
        if not materialized_changed_files:
            return None
        patch_artifact = (
            dict(candidate.patch_artifact) if isinstance(candidate.patch_artifact, dict) else {}
        )
        patch_artifact["changed_files"] = list(materialized_changed_files)
        selection_diagnostics = (
            dict(candidate.selection_diagnostics)
            if isinstance(candidate.selection_diagnostics, dict)
            else {}
        )
        selection_diagnostics["policy_sanitization"] = {
            "restored_protected_test_files": list(restored_protected_test_files),
            "sanitized_protected_test_files": list(sanitized_protected_test_files),
            "retained_changed_files": list(materialized_changed_files),
        }
        note_parts: list[str] = []
        if restored_protected_test_files:
            note_parts.append(
                "Protected visible test edits restored to baseline before final selection: "
                + ", ".join(restored_protected_test_files)
            )
        if sanitized_protected_test_files:
            note_parts.append(
                "Incomplete visible test edits were sanitized to placeholder-only completions before final selection: "
                + ", ".join(sanitized_protected_test_files)
            )
        note = "\n".join(note_parts).strip()
        explanation = (
            f"{candidate.explanation}\n\n{note}".strip() if candidate.explanation else note
        )
        return replace(
            candidate,
            patch=diff,
            explanation=explanation,
            changed_files=list(materialized_changed_files),
            worktree_path=str(workspace),
            baseline_commit=None,
            patch_artifact=patch_artifact,
            selection_diagnostics=selection_diagnostics,
        )

    def _cleanup_synthetic_worktrees(self, keep_path: Optional[str]) -> None:
        keep = Path(keep_path).resolve() if keep_path else None
        retained: list[Path] = []
        for workspace in self._ephemeral_worktrees:
            if keep is not None and workspace.resolve() == keep:
                retained.append(workspace)
                continue
            mode = self._ephemeral_workspace_modes.pop(workspace.resolve(), "snapshot")
            if mode == "git_clone" and workspace.exists():
                shutil.rmtree(workspace.parent, ignore_errors=True)
                continue
            shutil.rmtree(workspace.parent, ignore_errors=True)
        self._ephemeral_worktrees = retained
        self._ephemeral_workspace_modes = {
            workspace.resolve(): self._ephemeral_workspace_modes[workspace.resolve()]
            for workspace in retained
            if workspace.resolve() in self._ephemeral_workspace_modes
        }

    def _cluster_candidates(self, candidates: list[RolloutResult]) -> list[PatchCluster]:
        # Phase 2.6: deterministic clustering. Sort candidates by a stable
        # key (rollout_id, then a content-derived secondary) before doing
        # any clustering work so cluster IDs don't depend on the
        # iteration order the caller happened to hand us.
        sorted_candidates = sorted(
            candidates,
            key=lambda candidate: (
                int(getattr(candidate, "rollout_id", 0) or 0),
                hashlib.sha256(
                    (
                        str(candidate.patch or "")
                        + "\n"
                        + "\n".join(sorted(candidate.changed_files or []))
                    ).encode()
                ).hexdigest(),
            ),
        )

        # Step 1: bucket by *exact* fingerprint. Candidates whose
        # post-patch worktrees produce the same canonical AST/byte
        # signature must share a cluster regardless of similarity scoring.
        # Phase 2.3 assertion: by this point, every candidate has cleared
        # ``_prune_candidates`` and therefore has a substantive set of
        # changed files. ``_compute_candidate_fingerprint`` returning
        # None should be unreachable for pruned candidates; if it does
        # happen we still admit the candidate to similarity clustering
        # rather than crashing the whole selection round, but log an
        # error so the regression is visible.
        fingerprint_groups: dict[str, list[RolloutResult]] = {}
        unfingerprinted: list[RolloutResult] = []
        for candidate in sorted_candidates:
            fingerprint = self._compute_candidate_fingerprint(candidate)
            if fingerprint is None:
                logger.error(
                    "Candidate rollout %s has no fingerprint after pruning; "
                    "this should not happen post Phase 2.3 substantiality check.",
                    candidate.rollout_id,
                )
                unfingerprinted.append(candidate)
                continue
            fingerprint_groups.setdefault(fingerprint, []).append(candidate)

        # Step 2: agglomerative single-linkage clustering over the
        # fingerprint-equivalence buckets *plus* any unfingerprinted
        # stragglers. Each bucket is represented by its earliest-rollout
        # candidate (centroid). Buckets with similarity >= threshold are
        # merged into the same cluster.
        bucket_payload: list[tuple[str, list[RolloutResult]]] = []
        for fingerprint, members in fingerprint_groups.items():
            members_sorted = sorted(
                members,
                key=lambda candidate: int(getattr(candidate, "rollout_id", 0) or 0),
            )
            bucket_payload.append((fingerprint, members_sorted))
        for candidate in unfingerprinted:
            # Each unfingerprinted candidate forms its own initial bucket;
            # its payload is the structural similarity payload.
            bucket_payload.append(
                (
                    self._cluster_payload(candidate),
                    [candidate],
                )
            )

        # Stable bucket ordering: smallest member rollout_id first, then
        # the (already-deterministic) signature key as tiebreak. Cluster
        # IDs are assigned post-clustering by sorted-centroid order so
        # downstream consumers see a deterministic numbering.
        bucket_payload.sort(
            key=lambda entry: (
                int(getattr(entry[1][0], "rollout_id", 0) or 0),
                entry[0],
            )
        )

        threshold = self.config.selection.ast_similarity_threshold
        bucket_count = len(bucket_payload)
        # Compute pairwise similarity between bucket representatives.
        # Symmetric matrix; we only fill the upper triangle.
        similarities: dict[tuple[int, int], float] = {}
        for i in range(bucket_count):
            for j in range(i + 1, bucket_count):
                _, members_i = bucket_payload[i]
                _, members_j = bucket_payload[j]
                rep_i = members_i[0]
                rep_j = members_j[0]
                payload_i = self._cluster_payload(rep_i)
                similarities[(i, j)] = self._cluster_similarity_pair(rep_i, payload_i, rep_j)

        # Single-linkage agglomerative clustering: merge the highest-
        # similarity pair as long as it exceeds threshold; reduce the
        # similarity matrix using single-linkage (max similarity) and
        # repeat. Tiebreaks: smallest centroid rollout_id wins.
        cluster_assignment = list(range(bucket_count))

        def _find(idx: int) -> int:
            root = idx
            while cluster_assignment[root] != root:
                root = cluster_assignment[root]
            while cluster_assignment[idx] != root:
                next_idx = cluster_assignment[idx]
                cluster_assignment[idx] = root
                idx = next_idx
            return root

        # Sort all eligible pairs by descending similarity then by
        # ascending centroid rollout_id pair for deterministic tie
        # resolution.
        eligible_pairs = sorted(
            (
                (
                    -similarity,
                    bucket_payload[i][1][0].rollout_id,
                    bucket_payload[j][1][0].rollout_id,
                    i,
                    j,
                )
                for (i, j), similarity in similarities.items()
                if similarity >= threshold
            )
        )
        for _, _, _, i, j in eligible_pairs:
            root_i = _find(i)
            root_j = _find(j)
            if root_i == root_j:
                continue
            # Merge under the smaller-rollout-id centroid.
            centroid_i = bucket_payload[root_i][1][0].rollout_id
            centroid_j = bucket_payload[root_j][1][0].rollout_id
            if centroid_i <= centroid_j:
                cluster_assignment[root_j] = root_i
            else:
                cluster_assignment[root_i] = root_j

        # Group buckets by their root and assemble PatchClusters.
        groups: dict[int, list[int]] = {}
        for idx in range(bucket_count):
            root = _find(idx)
            groups.setdefault(root, []).append(idx)

        # Sort clusters by their centroid rollout_id (smallest first).
        ordered_roots = sorted(
            groups.keys(),
            key=lambda root: min(
                int(bucket_payload[bucket_idx][1][0].rollout_id or 0) for bucket_idx in groups[root]
            ),
        )

        clusters: list[PatchCluster] = []
        for new_cluster_id, root in enumerate(ordered_roots):
            bucket_indices = sorted(
                groups[root],
                key=lambda bucket_idx: int(bucket_payload[bucket_idx][1][0].rollout_id or 0),
            )
            patches: list[RolloutResult] = []
            payload_segments: list[str] = []
            for bucket_idx in bucket_indices:
                signature_key, members = bucket_payload[bucket_idx]
                patches.extend(members)
                payload_segments.append(signature_key)
            payload = "\n".join(payload_segments)
            signature = hashlib.sha256(payload.encode()).hexdigest()[:20]
            clusters.append(
                PatchCluster(
                    cluster_id=new_cluster_id,
                    signature=signature,
                    payload=payload,
                    patches=patches,
                )
            )
        return clusters

    def _cluster_similarity_pair(
        self,
        candidate: RolloutResult,
        candidate_payload: str,
        other: RolloutResult,
    ) -> float:
        """Symmetric similarity between two candidates.

        Decisive-Edge B.3: now driven by
        :func:`apex.selection.semantic_clustering.semantic_distance`.
        The exact-AST fingerprint pass in ``_cluster_candidates`` runs
        first (so semantically identical patches always co-cluster);
        this pass operates on the surviving fingerprint buckets, where
        we want semantic — not textual — proximity. The legacy
        Jaccard / SequenceMatcher mix is preserved as a final fallback
        when both candidates produce empty signatures (e.g. the diff
        was a binary or a non-Python config file we couldn't parse).
        """
        sig_candidate = self._semantic_signature(candidate)
        sig_other = self._semantic_signature(other)
        if _signature_has_signal(sig_candidate) and _signature_has_signal(sig_other):
            return semantic_similarity(sig_candidate, sig_other)
        # Final fallback: legacy text-similarity (Jaccard over file set
        # + normalized lines + SequenceMatcher on the payload string).
        # Only fires when *neither* candidate has any semantic signal.
        return self._legacy_text_similarity(candidate, candidate_payload, other)

    def _legacy_text_similarity(
        self,
        candidate: RolloutResult,
        candidate_payload: str,
        other: RolloutResult,
    ) -> float:
        """Legacy Jaccard + SequenceMatcher similarity preserved as a
        final fallback for Decisive-Edge B.3."""
        candidate_files = set(self._candidate_changed_files(candidate))
        other_files = set(self._candidate_changed_files(other))
        if not candidate_files and not other_files:
            file_similarity = 1.0
        else:
            union = candidate_files | other_files
            file_similarity = len(candidate_files & other_files) / max(len(union), 1)

        candidate_lines = set(self._normalized_patch_lines(candidate))
        other_lines = set(self._normalized_patch_lines(other))
        if not candidate_lines and not other_lines:
            line_similarity = 1.0
        else:
            union = candidate_lines | other_lines
            line_similarity = len(candidate_lines & other_lines) / max(len(union), 1)

        other_payload = self._cluster_payload(other)
        payload_similarity = SequenceMatcher(None, candidate_payload, other_payload).ratio()
        return (0.45 * file_similarity) + (0.40 * line_similarity) + (0.15 * payload_similarity)

    def _semantic_signature(
        self,
        candidate: RolloutResult,
    ) -> SemanticSignature:
        """Cache-and-return the semantic signature for one candidate.

        Cached on the candidate via a ``__apex_semantic_signature``
        attribute (``RolloutResult`` is a dataclass without a __slots__
        restriction so dynamic attributes are fine). The cache is
        process-local — no persistence — so a fresh selector run
        recomputes signatures from scratch.
        """
        cached = getattr(candidate, "__apex_semantic_signature", None)
        if cached is not None:
            return cached
        patch_text = candidate.patch or ""
        worktree_path = Path(candidate.worktree_path) if candidate.worktree_path else None
        try:
            signature = compute_semantic_signature(patch_text, repo_root=worktree_path)
        except Exception as exc:  # noqa: BLE001 - never fail clustering
            logger.warning(
                "compute_semantic_signature raised %s: %s; using empty signature.",
                type(exc).__name__,
                exc,
            )
            signature = SemanticSignature()
        try:
            object.__setattr__(candidate, "__apex_semantic_signature", signature)
        except (AttributeError, TypeError):
            pass
        return signature

    def _compute_candidate_fingerprint(self, candidate: RolloutResult) -> Optional[str]:
        if not candidate.worktree_path:
            return None
        worktree = Path(candidate.worktree_path)
        if not worktree.exists():
            return None

        fragments: list[str] = []
        changed_files = self._candidate_changed_files(candidate)
        if not changed_files:
            return None

        for rel_path in changed_files:
            file_path = worktree / rel_path
            if not file_path.exists():
                fragments.append(f"{rel_path}:<deleted>")
                continue
            if file_path.is_dir():
                continue

            content = file_path.read_text(errors="replace")
            if rel_path.endswith(".py"):
                try:
                    tree = ast.parse(content)
                except SyntaxError:
                    return None
                self._remove_docstrings(tree)
                canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
            else:
                canonical = self._normalize_non_python_content(rel_path, content)
            fragments.append(f"{rel_path}:{canonical}")

        if not fragments:
            return None
        return hashlib.sha256("\n".join(fragments).encode()).hexdigest()

    def _candidate_changed_files(self, candidate: RolloutResult) -> list[str]:
        raw_paths = list(candidate.changed_files)
        if (
            not raw_paths
            and candidate.worktree_path
            and hasattr(self.verifier, "_list_changed_files")
        ):
            try:
                raw_paths = self.verifier._list_changed_files(
                    Path(candidate.worktree_path),
                    baseline_ref=candidate.baseline_commit,
                )
            except TypeError as exc:
                if "baseline_ref" not in str(exc):
                    raise
                raw_paths = self.verifier._list_changed_files(Path(candidate.worktree_path))
        if not candidate.worktree_path:
            return sorted(dict.fromkeys(raw_paths))
        return expand_changed_paths(Path(candidate.worktree_path), raw_paths)

    def _remove_docstrings(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                body.pop(0)

    def _normalize_non_python_content(self, rel_path: str, content: str) -> str:
        lowered = rel_path.lower()
        normalized = content
        if lowered.endswith(
            (".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".h", ".hpp", ".php")
        ):
            normalized = re.sub(r"/\*.*?\*/", "", normalized, flags=re.DOTALL)
            normalized = re.sub(r"//.*", "", normalized)
        normalized = re.sub(r"\s+", "", normalized)
        return normalized

    def _cluster_payload(self, candidate: RolloutResult) -> str:
        lines = self._normalized_patch_lines(candidate)
        changed_files = self._candidate_changed_files(candidate)
        return "\n".join(changed_files + lines)

    def _normalized_patch_lines(self, candidate: RolloutResult) -> list[str]:
        normalized_lines = []
        for line in (candidate.patch or "").splitlines():
            if line.startswith(("diff ", "index ", "---", "+++", "@@")):
                continue
            if line.startswith(("+", "-")):
                normalized = line[1:].strip()
                if normalized and not normalized.startswith("#"):
                    normalized_lines.append(normalized)
        return sorted(normalized_lines)

    def _cluster_similarity(
        self,
        candidate: RolloutResult,
        payload: str,
        cluster: PatchCluster,
    ) -> float:
        """Decisive-Edge B.3: semantic distance against the cluster's
        representative, with the legacy Jaccard / SequenceMatcher mix
        preserved as a final fallback when no semantic signal is
        available on either side.

        ``payload`` (the legacy ``_cluster_payload`` string) is still
        accepted for back-compat with callers that compute it; it is
        only consulted on the fallback path.
        """
        representative = cluster.representative
        sig_candidate = self._semantic_signature(candidate)
        sig_cluster = self._semantic_signature(representative)
        if _signature_has_signal(sig_candidate) and _signature_has_signal(sig_cluster):
            return semantic_similarity(sig_candidate, sig_cluster)
        # Fall back to the legacy text similarity for cluster vs.
        # representative; uses the cluster payload string when
        # available so this branch matches historical scoring.
        candidate_files = set(self._candidate_changed_files(candidate))
        cluster_files = set(self._candidate_changed_files(representative))
        if not candidate_files and not cluster_files:
            file_similarity = 1.0
        else:
            union = candidate_files | cluster_files
            file_similarity = len(candidate_files & cluster_files) / max(len(union), 1)

        candidate_lines = set(self._normalized_patch_lines(candidate))
        cluster_lines = set(self._normalized_patch_lines(representative))
        if not candidate_lines and not cluster_lines:
            line_similarity = 1.0
        else:
            union = candidate_lines | cluster_lines
            line_similarity = len(candidate_lines & cluster_lines) / max(len(union), 1)

        payload_similarity = SequenceMatcher(None, payload, cluster.payload).ratio()
        return (0.45 * file_similarity) + (0.40 * line_similarity) + (0.15 * payload_similarity)

    def _verify_clusters(
        self,
        clusters: list[PatchCluster],
        all_candidates: list[RolloutResult],
        test_command: Optional[str],
        *,
        baseline_result: Optional[Any] = None,
        issue_plan: Optional["IssuePlan"] = None,
    ) -> None:
        index_by_rollout = {
            candidate.rollout_id: index for index, candidate in enumerate(all_candidates)
        }
        reproduction_artifacts = [
            candidate.reproduction_artifact
            for candidate in all_candidates
            if candidate.reproduction_artifact
        ]

        max_workers = self._verification_worker_limit(len(clusters))
        ordered_clusters = self._ordered_clusters_for_verification(clusters)
        matrix = None
        if self._should_short_circuit_verification_after_full_quick_pass(
            ordered_clusters,
            test_command=test_command,
        ):
            for index, cluster in enumerate(ordered_clusters, start=1):
                verification, cross_validation_score = self._build_cluster_verification(
                    cluster,
                    test_command=test_command,
                    matrix=matrix,
                    index_by_rollout=index_by_rollout,
                    reproduction_artifacts=reproduction_artifacts,
                    baseline_result=baseline_result,
                    issue_plan=issue_plan,
                )
                cluster.cross_validation_score = cross_validation_score
                cluster.verification = verification
                cluster.representative.verification = verification.to_dict()
                self._write_task_selection_state(
                    "verifying_clusters",
                    extra_fields={
                        "selection_cluster_count": len(clusters),
                        "selection_verified_cluster_count": index,
                    },
                )
                if self._cluster_has_decisive_acceptance(cluster):
                    self._annotate_cluster_verification_short_circuit(
                        cluster,
                        verified_count=index,
                        skipped_count=max(len(clusters) - index, 0),
                    )
                    return
            return

        if (
            self.config.selection.cross_validation_enabled
            and any(candidate.test_suite for candidate in all_candidates)
            and hasattr(self.verifier, "build_cross_validation_matrix")
        ):
            try:
                matrix = self.verifier.build_cross_validation_matrix(
                    all_candidates,
                    max_workers=self._cross_validation_worker_limit(len(all_candidates)),
                    test_command=test_command,
                )
            except TypeError as exc:
                if "max_workers" not in str(exc) and "test_command" not in str(exc):
                    raise
                try:
                    matrix = self.verifier.build_cross_validation_matrix(
                        all_candidates,
                        max_workers=self._cross_validation_worker_limit(len(all_candidates)),
                    )
                except TypeError as inner:
                    if "max_workers" not in str(inner):
                        raise
                    matrix = self.verifier.build_cross_validation_matrix(all_candidates)

        if max_workers == 1:
            for index, cluster in enumerate(ordered_clusters, start=1):
                verification, cross_validation_score = self._build_cluster_verification(
                    cluster,
                    test_command=test_command,
                    matrix=matrix,
                    index_by_rollout=index_by_rollout,
                    reproduction_artifacts=reproduction_artifacts,
                    baseline_result=baseline_result,
                    issue_plan=issue_plan,
                )
                cluster.cross_validation_score = cross_validation_score
                cluster.verification = verification
                cluster.representative.verification = verification.to_dict()
                self._write_task_selection_state(
                    "verifying_clusters",
                    extra_fields={
                        "selection_cluster_count": len(clusters),
                        "selection_verified_cluster_count": index,
                    },
                )
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Phase 2.6: launch verification work in parallel for
            # throughput, but reduce results in deterministic order
            # (sorted by cluster_id) so cluster.verification and the
            # subsequent ranking aren't sensitive to which thread happens
            # to finish first. Wall-clock-dependent reduction order was
            # producing different selection winners across reruns of the
            # same candidate set.
            futures = {
                cluster.cluster_id: executor.submit(
                    self._build_cluster_verification,
                    cluster,
                    test_command=test_command,
                    matrix=matrix,
                    index_by_rollout=index_by_rollout,
                    reproduction_artifacts=reproduction_artifacts,
                    baseline_result=baseline_result,
                    issue_plan=issue_plan,
                )
                for cluster in ordered_clusters
            }
            verified_count = 0
            for cluster in ordered_clusters:
                future = futures[cluster.cluster_id]
                verification, cross_validation_score = future.result()
                cluster.cross_validation_score = cross_validation_score
                cluster.verification = verification
                cluster.representative.verification = verification.to_dict()
                verified_count += 1
                self._write_task_selection_state(
                    "verifying_clusters",
                    extra_fields={
                        "selection_cluster_count": len(clusters),
                        "selection_verified_cluster_count": verified_count,
                    },
                )

    def _build_cluster_verification(
        self,
        cluster: PatchCluster,
        *,
        test_command: Optional[str],
        matrix: Any,
        index_by_rollout: dict[int, int],
        reproduction_artifacts: list[dict[str, Any]],
        baseline_result: Optional[Any] = None,
        issue_plan: Optional["IssuePlan"] = None,
    ) -> tuple[VerificationResult, float]:
        def _verify_member(candidate: RolloutResult) -> tuple[VerificationResult, float]:
            verification = self._verify_candidate_or_confirmed_handoff(
                candidate,
                test_command,
                baseline_result=baseline_result,
                issue_plan=issue_plan,
            )
            cross_validation_score = 0.0

            if candidate.rollout_id in self._prune_results:
                verification.prune_result = self._serialize_prune_result(
                    self._prune_results[candidate.rollout_id]
                )
            self._apply_candidate_stub_residue_gate(candidate, verification)
            self._reconcile_verification_with_candidate_handoff(candidate, verification)
            self._reconcile_candidate_validity_with_verification(candidate, verification)

            if matrix is not None and candidate.rollout_id in index_by_rollout:
                row = matrix[index_by_rollout[candidate.rollout_id]]
                values = (
                    [float(item) for item in row.tolist()]
                    if hasattr(row, "tolist")
                    else [float(item) for item in row]
                )
                scores = self._cross_validation_scores(
                    values,
                    index_by_rollout[candidate.rollout_id],
                )
                verification.cross_validation_scores = scores
                if scores:
                    cross_validation_score = sum(scores) / len(scores)
                    verification.overall_score = self.verifier._compute_score(verification)
                verification.accepted = self._verification_meets_acceptance_bar(
                    candidate,
                    verification,
                    test_command,
                    issue_plan=issue_plan,
                )
                self._apply_evidence_bound_review(
                    candidate,
                    verification,
                    issue_plan=issue_plan,
                )
                return verification, cross_validation_score

            other_artifacts = [
                artifact
                for artifact in reproduction_artifacts
                if artifact != candidate.reproduction_artifact
            ]
            if self.config.selection.cross_validation_enabled and other_artifacts:
                try:
                    verification.cross_validation_scores = self.verifier.cross_validate(
                        candidate.worktree_path or "",
                        other_artifacts,
                        test_command=test_command,
                    )
                except TypeError as exc:
                    if "test_command" not in str(exc):
                        raise
                    verification.cross_validation_scores = self.verifier.cross_validate(
                        candidate.worktree_path or "",
                        other_artifacts,
                    )
                if verification.cross_validation_scores:
                    cross_validation_score = sum(verification.cross_validation_scores) / len(
                        verification.cross_validation_scores
                    )
                    verification.overall_score = self.verifier._compute_score(verification)
            verification.accepted = self._verification_meets_acceptance_bar(
                candidate,
                verification,
                test_command,
                issue_plan=issue_plan,
            )
            self._apply_evidence_bound_review(candidate, verification, issue_plan=issue_plan)
            return verification, cross_validation_score

        outcomes: list[tuple[RolloutResult, VerificationResult, float]] = []
        for candidate in self._ordered_cluster_candidates_for_verification(cluster):
            verification, cross_validation_score = _verify_member(candidate)
            candidate.verification = verification.to_dict()
            outcomes.append((candidate, verification, cross_validation_score))
            if verification.accepted:
                break

        if not outcomes:
            representative = cluster.representative
            verification, cross_validation_score = _verify_member(representative)
            representative.verification = verification.to_dict()
            return verification, cross_validation_score

        def _verification_rank(
            outcome: tuple[RolloutResult, VerificationResult, float],
        ) -> tuple[Any, ...]:
            candidate, verification, cross_validation_score = outcome
            verification_payload = verification.to_dict()
            hard_validity_rejected = verification_has_explicit_validity_rejection(
                verification_payload
            )
            test_result = verification.test_result
            pass_rate = float(getattr(test_result, "pass_rate", 0.0) or 0.0)
            coverage_preserved = getattr(test_result, "expected_coverage_preserved", None)
            quick_signal = quick_verification_signal_score(
                candidate.quick_verification
                if isinstance(candidate.quick_verification, dict)
                else {}
            )
            residual_defect_rank = tuple(
                -count
                for count in self._verification_residual_defect_counts(
                    candidate,
                    verification,
                )
            )
            return (
                1 if verification.accepted else 0,
                0 if hard_validity_rejected else 1,
                1 if coverage_preserved is not False else 0,
                pass_rate,
                *residual_defect_rank,
                float(verification.overall_score or 0.0),
                float(cross_validation_score or 0.0),
                float(quick_signal or 0.0),
                -int(getattr(candidate, "rollout_id", 0) or 0),
            )

        chosen_candidate, chosen_verification, chosen_cross_validation_score = max(
            outcomes,
            key=_verification_rank,
        )
        if cluster.patches and cluster.patches[0] is not chosen_candidate:
            cluster.patches = [chosen_candidate] + [
                candidate for candidate in cluster.patches if candidate is not chosen_candidate
            ]
            diagnostics = (
                dict(chosen_candidate.selection_diagnostics)
                if isinstance(chosen_candidate.selection_diagnostics, dict)
                else {}
            )
            diagnostics["cluster_representative_upgrade"] = {
                "previous_rollout_id": outcomes[0][0].rollout_id,
                "selected_rollout_id": chosen_candidate.rollout_id,
                "reason": (
                    "cluster_member_satisfied_acceptance_bar"
                    if chosen_verification.accepted
                    else "cluster_member_minimized_validity_residuals"
                ),
            }
            chosen_candidate.selection_diagnostics = diagnostics
        return chosen_verification, chosen_cross_validation_score

    def _ordered_clusters_for_verification(
        self,
        clusters: list[PatchCluster],
    ) -> list[PatchCluster]:
        return sorted(
            clusters,
            key=self._cluster_verification_priority,
        )

    def _ordered_cluster_candidates_for_verification(
        self,
        cluster: PatchCluster,
    ) -> list[RolloutResult]:
        return sorted(
            list(cluster.patches or []),
            key=self._candidate_verification_priority,
        )

    def _cluster_verification_priority(self, cluster: PatchCluster) -> tuple[Any, ...]:
        candidates = list(cluster.patches or [])
        if not candidates:
            return (1, 0.0, 0, int(cluster.cluster_id))
        candidate_keys = [
            self._candidate_verification_priority(candidate) for candidate in candidates
        ]
        return (*min(candidate_keys), int(cluster.cluster_id))

    def _candidate_verification_priority(self, candidate: RolloutResult) -> tuple[Any, ...]:
        quick = (
            candidate.quick_verification if isinstance(candidate.quick_verification, dict) else {}
        )
        full_scope_pass = self._candidate_full_scope_quick_pass_preserves_expected(candidate)
        signal_score = quick_verification_signal_score(quick)
        changed_count = len(list(candidate.changed_files or []))
        return (
            0 if full_scope_pass else 1,
            -float(signal_score or 0.0),
            changed_count,
            int(getattr(candidate, "rollout_id", 0) or 0),
        )

    def _should_short_circuit_verification_after_full_quick_pass(
        self,
        clusters: list[PatchCluster],
        *,
        test_command: Optional[str],
    ) -> bool:
        if len(clusters) <= 1 or not str(test_command or "").strip():
            return False
        return any(
            self._candidate_full_scope_quick_pass_preserves_expected(candidate)
            for cluster in clusters
            for candidate in list(cluster.patches or [])
        )

    def _annotate_cluster_verification_short_circuit(
        self,
        cluster: PatchCluster,
        *,
        verified_count: int,
        skipped_count: int,
    ) -> None:
        diagnostics = (
            dict(cluster.representative.selection_diagnostics)
            if isinstance(cluster.representative.selection_diagnostics, dict)
            else {}
        )
        selector_diag = (
            dict(diagnostics.get("selector"))
            if isinstance(diagnostics.get("selector"), dict)
            else {}
        )
        selector_diag["verification_short_circuit"] = {
            "reason": "decisive_full_scope_quick_pass_verified",
            "verified_cluster_count": int(verified_count),
            "skipped_cluster_count": int(max(skipped_count, 0)),
        }
        diagnostics["selector"] = selector_diag
        cluster.representative.selection_diagnostics = diagnostics

    def _verification_residual_defect_counts(
        self,
        candidate: RolloutResult,
        verification: VerificationResult,
    ) -> tuple[int, int, int, int, int]:
        """Count remaining verifier defects for invalid cluster-member ranking.

        This is deliberately generic: it uses the verifier contract surfaces
        rather than any benchmark-specific file or language rule. The counts
        only break ties among otherwise comparable candidates, so a later
        follow-up round starts from the member with the smallest remaining
        repair surface.
        """
        syntax_defects = 0 if verification.syntax_valid else 1
        lint_defects = 0
        if not verification.lint_clean:
            lint_lines = [
                line for line in str(verification.lint_output or "").splitlines() if line.strip()
            ]
            lint_defects = max(1, len(lint_lines))

        diagnostics = (
            candidate.selection_diagnostics
            if isinstance(candidate.selection_diagnostics, dict)
            else {}
        )
        stub_payload = diagnostics.get("stub_residue")
        stub_defects = len(stub_payload) if isinstance(stub_payload, list) else 0

        validity_reasons = [
            str(reason)
            for reason in list(getattr(verification, "validity_reasons", []) or [])
            if str(reason).strip()
        ]
        quality_gate_defects = 0
        if verification.quality_gate_passed is False:
            quality_gate_defects = max(1, len(validity_reasons))

        prune_result = (
            verification.prune_result if isinstance(verification.prune_result, dict) else {}
        )
        prune_defects = 0
        if prune_result.get("is_valid") is False:
            regressed_tests = prune_result.get("regressed_tests")
            if isinstance(regressed_tests, list):
                prune_defects = max(1, len(regressed_tests))
            else:
                prune_defects = 1

        test_result = verification.test_result
        coverage_defects = (
            1 if getattr(test_result, "expected_coverage_preserved", None) is False else 0
        )
        return (
            syntax_defects,
            lint_defects,
            quality_gate_defects + stub_defects,
            prune_defects,
            coverage_defects,
        )

    def _candidate_has_confirmed_quick_prune_acceptance(
        self,
        candidate: RolloutResult,
    ) -> bool:
        """True when APEX already accepted a full-coverage quick handoff.

        This is a selector-level validity contract, not a benchmark shortcut:
        the candidate must have survived hard validity gates, preserved expected
        coverage, and either be marked by pruning as
        ``quick_verification_accepted`` or carry an eligible submission
        validity payload from rollout finalization. Final benchmark audits remain
        authoritative for published scoring.
        """

        if not self._candidate_full_scope_quick_pass_preserves_expected(candidate):
            return False
        prune_result = self._prune_results.get(candidate.rollout_id)
        if prune_result is not None:
            if hasattr(prune_result, "to_dict"):
                try:
                    prune_payload = prune_result.to_dict()
                except Exception:
                    prune_payload = {}
            elif isinstance(prune_result, dict):
                prune_payload = prune_result
            else:
                prune_payload = {
                    "is_valid": getattr(prune_result, "is_valid", None),
                    "reason": getattr(prune_result, "reason", ""),
                }
            if (
                prune_payload.get("is_valid") is True
                and str(prune_payload.get("reason") or "").strip() == "quick_verification_accepted"
            ):
                return True
        validity = self._candidate_validity_payload(candidate)
        if not validity:
            return False
        return bool(
            validity.get("eligible_for_submission") is True
            and validity.get("quick_verification_passed") is True
            and validity.get("expected_coverage_preserved") is True
            and int(validity.get("missing_expected_test_count") or 0) == 0
            and validity.get("backend_protocol_error") is not True
            and validity.get("coverage_collapse_terminal") is not True
            and validity.get("provenance_violation") is not True
            and validity.get("protected_tests_unchanged") is not False
            and validity.get("collection_critical_files_unchanged") is not False
            and validity.get("quality_gate_passed") is not False
        )

    def _verification_from_confirmed_quick_handoff(
        self,
        candidate: RolloutResult,
        test_command: Optional[str],
    ) -> VerificationResult:
        quick = (
            candidate.quick_verification if isinstance(candidate.quick_verification, dict) else {}
        )
        passed = int(quick.get("passed") or 0)
        failed = int(quick.get("failed") or 0)
        errors = int(quick.get("errors") or 0)
        test_result = TestResult(
            passed=max(passed, 1 if passed <= 0 and failed == 0 and errors == 0 else 0),
            failed=failed,
            errors=errors,
            reproduction_passes=True,
            regression_passes=bool(str(test_command or "").strip()),
            regression_inconclusive=False,
            reproduction_output=str(quick.get("output_excerpt") or ""),
            regression_output=(
                "Selector reused confirmed full-coverage quick-verification handoff; "
                "duplicate full-suite verifier run deferred to the final benchmark audit."
            ),
            expected_test_count=int(quick.get("expected_test_count") or 0),
            matched_expected_test_count=int(quick.get("matched_expected_test_count") or 0),
            missing_expected_test_count=int(quick.get("missing_expected_test_count") or 0),
            collected_test_count=int(quick.get("collected_test_count") or 0),
            expected_coverage_preserved=True,
        )
        missing_ids = quick.get("missing_expected_test_ids")
        if isinstance(missing_ids, list):
            test_result.missing_expected_test_ids = [
                str(test_id) for test_id in missing_ids if str(test_id or "").strip()
            ]
        self._copy_quick_verification_counts_to_test_result(test_result, quick)
        verification = VerificationResult(
            rollout_id=candidate.rollout_id,
            syntax_valid=True,
            lint_clean=True,
            lint_applied=False,
            accepted=False,
            changed_files=list(candidate.changed_files or []),
            test_result=test_result,
            quality_gate_passed=None,
            validity_reasons=["confirmed_quick_verification_handoff"],
        )
        compute_score = getattr(self.verifier, "_compute_score", None)
        if callable(compute_score):
            verification.overall_score = compute_score(verification)
        else:
            verification.overall_score = self._fallback_verification_score(verification)
        diagnostics = (
            dict(candidate.selection_diagnostics)
            if isinstance(candidate.selection_diagnostics, dict)
            else {}
        )
        selector_diag = (
            dict(diagnostics.get("selector"))
            if isinstance(diagnostics.get("selector"), dict)
            else {}
        )
        selector_diag["verification_reused_confirmed_quick_handoff"] = {
            "reason": "quick_verification_accepted",
            "quick_scope": quick.get("scope"),
            "expected_test_count": quick.get("expected_test_count"),
            "matched_expected_test_count": quick.get("matched_expected_test_count"),
            "missing_expected_test_count": quick.get("missing_expected_test_count"),
        }
        diagnostics["selector"] = selector_diag
        candidate.selection_diagnostics = diagnostics
        metadata = (
            dict(candidate.search_metadata) if isinstance(candidate.search_metadata, dict) else {}
        )
        metadata["selector_verification_source"] = "confirmed_quick_verification_handoff"
        candidate.search_metadata = metadata
        return verification

    def _verify_candidate_or_confirmed_handoff(
        self,
        candidate: RolloutResult,
        test_command: Optional[str],
        *,
        baseline_result: Optional[Any] = None,
        issue_plan: Optional["IssuePlan"] = None,
    ) -> VerificationResult:
        if self._candidate_has_confirmed_quick_prune_acceptance(candidate):
            return self._verification_from_confirmed_quick_handoff(candidate, test_command)
        return self._verify_candidate(
            candidate,
            test_command,
            baseline_result=baseline_result,
            issue_plan=issue_plan,
        )

    def _verification_worker_limit(self, cluster_count: int) -> int:
        if cluster_count <= 0:
            return 1
        configured = max(self.config.rollout.parallel_workers, 1)
        return max(1, min(cluster_count, configured))

    def _cross_validation_worker_limit(self, candidate_count: int) -> int:
        if candidate_count <= 0:
            return 1
        configured = max(self.config.rollout.parallel_workers, 1)
        return max(1, min(candidate_count, configured))

    def _cross_validation_scores(self, values: list[float], self_index: int) -> list[float]:
        """Exclude self-validation so a candidate is scored only by other rollouts.

        Returns an empty list when the candidate has no peers — callers must
        treat that as 'abstain' rather than synthesizing a 0.5 prior. The old
        ``[0.5]`` fallback artificially inflated singleton candidates above
        peers that had genuine 0.0 cross-validation evidence.
        """
        if not values:
            return []
        return [score for index, score in enumerate(values) if index != self_index]

    def _verify_candidate(
        self,
        candidate: RolloutResult,
        test_command: Optional[str],
        *,
        baseline_result: Optional[Any] = None,
        issue_plan: Optional["IssuePlan"] = None,
    ) -> VerificationResult:
        if self._component_disabled_for_candidate(candidate, "full_suite_gate"):
            return self._quick_only_ablation_verification(candidate)
        current_worktree_path = str(candidate.worktree_path or "").strip()
        if (
            (not current_worktree_path or not Path(current_worktree_path).is_dir())
            and is_git_repo(Path(self.repo_path))
            and not self._ensure_candidate_worktree_materialized(
                candidate,
                reason="candidate_verification",
            )
        ):
            return VerificationResult(
                rollout_id=candidate.rollout_id,
                syntax_valid=False,
                lint_clean=False,
                lint_applied=False,
                accepted=False,
                changed_files=list(candidate.changed_files or []),
                overall_score=0.0,
                quality_gate_passed=False,
                validity_reasons=["candidate_worktree_missing"],
            )
        artifact = candidate.reproduction_artifact or {}
        expected_test_ids = (
            canonical_expected_test_ids(issue_plan) if issue_plan is not None else []
        )
        expected_test_count = (
            canonical_expected_test_count(issue_plan) if issue_plan is not None else 0
        )
        test_inventory = canonical_test_inventory(issue_plan) if issue_plan is not None else None
        kwargs: dict[str, Any] = {
            "rollout_id": candidate.rollout_id,
            "worktree_path": candidate.worktree_path or "",
            "reproduction_command": artifact.get("command"),
            "reproduction_script": artifact.get("script_content"),
            "reproduction_script_path": artifact.get("script_path"),
            "test_command": test_command,
        }
        if test_inventory is not None and not test_inventory.is_empty():
            kwargs["test_inventory"] = test_inventory
        if expected_test_count > 0:
            kwargs["expected_test_count"] = expected_test_count
        if expected_test_ids:
            kwargs["expected_test_ids"] = expected_test_ids
        if candidate.baseline_commit is not None:
            kwargs["baseline_ref"] = candidate.baseline_commit
        if baseline_result is not None:
            kwargs["baseline_result"] = baseline_result

        try:
            return self.verifier.verify_patch(**kwargs)
        except TypeError as exc:
            return self._retry_verify_candidate(kwargs, exc)

    def _quick_only_ablation_verification(
        self,
        candidate: RolloutResult,
    ) -> VerificationResult:
        quick = (
            candidate.quick_verification if isinstance(candidate.quick_verification, dict) else {}
        )
        passed = int(quick.get("passed") or 0)
        failed = int(quick.get("failed") or 0)
        errors = int(quick.get("errors") or 0)
        signal = quick_verification_signal_score(quick)
        accepted = quick_verification_has_strong_signal(quick, require_full_scope=False)
        test_result = TestResult(
            passed=passed,
            failed=failed,
            errors=errors,
            reproduction_passes=accepted,
            regression_passes=accepted,
            regression_inconclusive=False,
            reproduction_output=str(quick.get("output_excerpt") or ""),
            regression_output=str(quick.get("output_excerpt") or ""),
            expected_test_count=int(quick.get("expected_test_count") or 0),
            matched_expected_test_count=int(quick.get("matched_expected_test_count") or 0),
            missing_expected_test_count=int(quick.get("missing_expected_test_count") or 0),
            collected_test_count=int(quick.get("collected_test_count") or 0),
            expected_coverage_preserved=quick.get("coverage_preserved")
            if isinstance(quick.get("coverage_preserved"), bool)
            else None,
        )
        missing_ids = quick.get("missing_expected_test_ids")
        if isinstance(missing_ids, list):
            test_result.missing_expected_test_ids = [
                str(test_id) for test_id in missing_ids if str(test_id or "").strip()
            ]
        for source_key, attr in {
            "test_inventory_framework": "test_inventory_framework",
            "test_inventory_language": "test_inventory_language",
            "test_inventory_source": "test_inventory_source",
            "test_inventory_collection_command": "test_inventory_collection_command",
        }.items():
            value = quick.get(source_key)
            if isinstance(value, str) and value.strip():
                setattr(test_result, attr, value)
        verification = VerificationResult(
            rollout_id=candidate.rollout_id,
            syntax_valid=True,
            lint_clean=True,
            lint_applied=False,
            accepted=bool(accepted),
            changed_files=list(candidate.changed_files or []),
            test_result=test_result,
            overall_score=float(signal if isinstance(signal, (int, float)) else 0.0),
            quality_gate_passed=None,
            validity_reasons=["component_ablation_disabled_full_suite_gate"],
        )
        diagnostics = (
            dict(candidate.selection_diagnostics)
            if isinstance(candidate.selection_diagnostics, dict)
            else {}
        )
        diagnostics["component_ablation_full_suite_gate"] = {
            "used_quick_verification_only": True,
            "quick_scope": quick.get("scope"),
            "accepted": bool(accepted),
        }
        candidate.selection_diagnostics = diagnostics
        return verification

    def _prune_candidate_by_regression(
        self,
        candidate: RolloutResult,
        baseline: Any,
        test_command: str,
    ) -> Any:
        if not self._ensure_candidate_worktree_materialized(
            candidate,
            reason="regression_pruning",
        ):
            return PruneResult(
                is_valid=False,
                regressed_tests=[],
                still_passing=[],
                reason="candidate_worktree_missing",
            )
        try:
            if candidate.baseline_commit is not None:
                return self.verifier.prune_by_regression(
                    candidate.worktree_path or "",
                    baseline,
                    test_command,
                    baseline_ref=candidate.baseline_commit,
                )
            return self.verifier.prune_by_regression(
                candidate.worktree_path or "",
                baseline,
                test_command,
            )
        except TypeError as exc:
            if "baseline_ref" not in str(exc):
                raise
            return self.verifier.prune_by_regression(
                candidate.worktree_path or "",
                baseline,
                test_command,
            )

    def _serialize_prune_result(self, prune_result: Any) -> dict[str, Any]:
        if hasattr(prune_result, "to_dict"):
            return prune_result.to_dict()
        if isinstance(prune_result, dict):
            return prune_result
        return {"value": prune_result}

    def _retry_verify_candidate(
        self,
        kwargs: dict[str, Any],
        exc: TypeError,
    ) -> VerificationResult:
        message = str(exc)
        if "test_inventory" in message:
            kwargs = dict(kwargs)
            kwargs.pop("test_inventory", None)
            try:
                return self.verifier.verify_patch(**kwargs)
            except TypeError as inner:
                return self._retry_verify_candidate(kwargs, inner)
        if "reproduction_script_path" in message:
            kwargs = dict(kwargs)
            kwargs.pop("reproduction_script_path", None)
            try:
                return self.verifier.verify_patch(**kwargs)
            except TypeError as inner:
                return self._retry_verify_candidate(kwargs, inner)
        if "baseline_ref" in message:
            kwargs = dict(kwargs)
            kwargs.pop("baseline_ref", None)
            try:
                return self.verifier.verify_patch(**kwargs)
            except TypeError as inner:
                return self._retry_verify_candidate(kwargs, inner)
        if "baseline_result" in message:
            kwargs = dict(kwargs)
            kwargs.pop("baseline_result", None)
            try:
                return self.verifier.verify_patch(**kwargs)
            except TypeError as inner:
                return self._retry_verify_candidate(kwargs, inner)
        if "expected_test_ids" in message:
            kwargs = dict(kwargs)
            kwargs.pop("expected_test_ids", None)
            try:
                return self.verifier.verify_patch(**kwargs)
            except TypeError as inner:
                return self._retry_verify_candidate(kwargs, inner)
        if "expected_test_count" in message:
            kwargs = dict(kwargs)
            kwargs.pop("expected_test_count", None)
            try:
                return self.verifier.verify_patch(**kwargs)
            except TypeError as inner:
                return self._retry_verify_candidate(kwargs, inner)
        raise exc

    def _candidate_public_symbol_losses(
        self,
        candidate: RolloutResult,
    ) -> list[Any]:
        """Cross-language symbol-loss detection.

        Today implemented for Python via AST diff against ``apex-base``.
        Returns the list of public symbols (def / class / module-level
        assignment) that existed in the baseline copy of an edited file
        but are missing from the candidate's version. Other languages
        return empty (would need tree-sitter for accuracy); the
        infrastructure-edit gate + stub scanner cover the same failure
        mode at coarser grain there.
        """
        if not candidate.worktree_path or not candidate.changed_files:
            return []
        workspace = Path(candidate.worktree_path)
        if not workspace.exists():
            return []
        try:
            from ..core.symbol_survival import detect_public_symbol_losses
        except ImportError:
            return []
        return detect_public_symbol_losses(workspace, candidate.changed_files)

    def _candidate_stub_residue_findings(
        self,
        candidate: RolloutResult,
    ) -> list[Any]:
        """Cross-language scan for unimplemented function bodies in
        non-test source files the candidate edited.
        """
        if not candidate.worktree_path or not candidate.changed_files:
            return []
        workspace = Path(candidate.worktree_path)
        if not workspace.exists():
            return []
        try:
            from ..core.stub_scanner import scan_files_for_stubs
            from ..core.test_runners import detect_adapter
        except ImportError:
            return []
        adapter = detect_adapter(workspace)
        patterns = adapter.stub_patterns() if adapter is not None else []
        return scan_files_for_stubs(
            workspace,
            candidate.changed_files,
            adapter_stub_patterns=patterns,
        )

    @staticmethod
    def _verification_has_proven_full_expected_coverage(
        verification: VerificationResult,
    ) -> bool:
        """True only when the verification PROVES full expected coverage ran.

        Requires the expected test set to be KNOWN (not None), strictly positive,
        fully matched (matched >= expected), with zero missing and no failures or
        errors. Unknown counts (e.g. a capped quick-verification that did not
        collect the full expected set) return False — absence of coverage data is
        not proof of coverage. Used to gate any downgrade of ground-truth stub
        residue so an incomplete candidate cannot be nominated on subset evidence.
        """
        test_result = getattr(verification, "test_result", None)
        if test_result is None:
            return False
        if getattr(test_result, "expected_coverage_preserved", None) is False:
            return False
        if int(getattr(test_result, "failed", 0) or 0) > 0:
            return False
        if int(getattr(test_result, "errors", 0) or 0) > 0:
            return False
        expected = getattr(test_result, "expected_test_count", None)
        matched = getattr(test_result, "matched_expected_test_count", None)
        missing = getattr(test_result, "missing_expected_test_count", None)
        if expected is None or matched is None or missing is None:
            return False
        if int(expected or 0) <= 0:
            return False
        if int(missing or 0) > 0:
            return False
        if int(matched or 0) < int(expected or 0):
            return False
        return True

    @staticmethod
    def _verification_proves_clean_scored_expected_suite(
        verification: VerificationResult,
    ) -> bool:
        """Return true when verifier evidence supersedes narrower rollout QV."""

        if not PatchSelector._verification_has_proven_full_expected_coverage(verification):
            return False
        test_result = getattr(verification, "test_result", None)
        if test_result is None:
            return False
        if not bool(getattr(test_result, "regression_passes", False)):
            return False
        if bool(getattr(test_result, "regression_inconclusive", False)):
            return False
        try:
            pass_rate = float(getattr(test_result, "pass_rate", 0.0) or 0.0)
        except (TypeError, ValueError):
            return False
        return pass_rate >= 0.999

    @staticmethod
    def _quick_verification_from_clean_selector_verification(
        verification: VerificationResult,
    ) -> dict[str, Any]:
        if not PatchSelector._verification_proves_clean_scored_expected_suite(verification):
            return {}
        test_result = verification.test_result
        if test_result is None:
            return {}
        expected = int(getattr(test_result, "expected_test_count", 0) or 0)
        matched = int(getattr(test_result, "matched_expected_test_count", 0) or 0)
        passed = int(getattr(test_result, "passed", 0) or 0)
        collected = int(getattr(test_result, "collected_test_count", 0) or 0)
        payload: dict[str, Any] = {
            "scope": "full_test_command",
            "returncode": 0,
            "timed_out": False,
            "full_scope_timed_out": False,
            "passed": passed,
            "failed": 0,
            "errors": 0,
            "pass_rate": 1.0,
            "expected_test_count": expected,
            "matched_expected_test_count": matched,
            "missing_expected_test_count": 0,
            "coverage_preserved": True,
            "expected_coverage_ratio": 1.0,
            "collected_test_count": max(collected, matched, passed, expected),
            "verification_source": "selector_verifier",
        }
        for source_key, attr in {
            "test_inventory_framework": "test_inventory_framework",
            "test_inventory_language": "test_inventory_language",
            "test_inventory_source": "test_inventory_source",
            "test_inventory_collection_command": "test_inventory_collection_command",
        }.items():
            value = getattr(test_result, attr, "")
            if isinstance(value, str) and value.strip():
                payload[source_key] = value
        signal_score = quick_verification_signal_score(payload)
        if isinstance(signal_score, (int, float)):
            payload["signal_score"] = max(0.0, min(float(signal_score), 1.0))
        return payload

    @staticmethod
    def _merge_selector_verification_quick_payload(
        existing: dict[str, Any],
        recovered: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(recovered)
        for key in (
            "selected_tests",
            "passed_tests",
            "failed_tests",
            "failure_clusters",
            "output_excerpt",
            "failure_classification",
        ):
            if key not in merged and key in existing:
                merged[key] = existing[key]
        return merged

    def _reconcile_candidate_validity_with_verification(
        self,
        candidate: RolloutResult,
        verification: VerificationResult,
    ) -> None:
        """Let authoritative verifier coverage repair stale reduced-scope validity."""

        quick_payload = self._quick_verification_from_clean_selector_verification(verification)
        if not quick_payload:
            return

        existing_quick = (
            candidate.quick_verification if isinstance(candidate.quick_verification, dict) else {}
        )
        handoff_artifacts_changed = False
        if self._quick_verification_artifact_is_stronger(existing_quick, quick_payload):
            candidate.quick_verification = self._merge_selector_verification_quick_payload(
                existing_quick,
                quick_payload,
            )
            handoff_artifacts_changed = True

        validity = getattr(candidate, "validity", None)
        if validity is None:
            worktree_path = str(candidate.worktree_path or "").strip()
            validity = CandidateValidity(
                has_patch=bool(isinstance(candidate.patch, str) and candidate.patch.strip()),
                worktree_materialized=bool(worktree_path and Path(worktree_path).is_dir()),
                expected_coverage_preserved=True,
                missing_expected_test_count=0,
                protected_tests_unchanged=True,
                collection_critical_files_unchanged=True,
                quick_verification_passed=True,
                quality_gate_passed=None,
                backend_protocol_error=False,
                coverage_collapse_terminal=False,
                provenance_violation=False,
                reasons=[],
            )
            candidate.validity = validity
            handoff_artifacts_changed = True

        if (
            bool(validity.backend_protocol_error)
            or bool(getattr(validity, "provenance_violation", False))
            or validity.quality_gate_passed is False
            or not bool(validity.protected_tests_unchanged)
            or not bool(validity.collection_critical_files_unchanged)
        ):
            if handoff_artifacts_changed:
                self._persist_reconciled_candidate_handoff_artifacts(candidate)
            return

        original = validity.as_dict()
        validity.expected_coverage_preserved = True
        validity.missing_expected_test_count = 0
        validity.quick_verification_passed = True
        validity.coverage_collapse_terminal = False
        stale_reasons = {
            "coverage_collapse_terminal",
            "expected_coverage_collapsed",
            "expected_coverage_gap",
            "quick_verification_failed",
        }
        validity.reasons = [
            str(reason)
            for reason in list(validity.reasons or [])
            if str(reason) not in stale_reasons
            and not str(reason).startswith("missing_expected_tests:")
        ]

        if validity.as_dict() == original:
            if handoff_artifacts_changed:
                self._persist_reconciled_candidate_handoff_artifacts(candidate)
            return
        handoff_artifacts_changed = True
        diagnostics = (
            dict(candidate.selection_diagnostics)
            if isinstance(candidate.selection_diagnostics, dict)
            else {}
        )
        selector_diag = (
            dict(diagnostics.get("selector"))
            if isinstance(diagnostics.get("selector"), dict)
            else {}
        )
        selector_diag["validity_reconciled_from_selector_verification"] = True
        selector_diag["validity_reconciled_original"] = original
        diagnostics["selector"] = selector_diag
        candidate.selection_diagnostics = diagnostics
        metadata = (
            dict(candidate.search_metadata) if isinstance(candidate.search_metadata, dict) else {}
        )
        metadata["candidate_validity_reconciled_from_selector_verification"] = True
        candidate.search_metadata = metadata
        self._persist_reconciled_candidate_handoff_artifacts(candidate)

    def _apply_candidate_stub_residue_gate(
        self,
        candidate: RolloutResult,
        verification: VerificationResult,
    ) -> None:
        findings = self._candidate_stub_residue_findings(candidate)
        if not findings:
            return
        payload = [
            {
                "path": getattr(finding, "path", ""),
                "symbol": getattr(finding, "symbol", ""),
                "reason": getattr(finding, "reason", ""),
            }
            for finding in findings[:25]
        ]
        diagnostics = (
            dict(candidate.selection_diagnostics)
            if isinstance(candidate.selection_diagnostics, dict)
            else {}
        )
        diagnostics["stub_residue"] = payload
        candidate.selection_diagnostics = diagnostics
        # Stub residue (an edited symbol whose body still raises
        # NotImplementedError / is an Ellipsis placeholder) is GROUND-TRUTH
        # evidence that the implementation is incomplete. It may be downgraded to
        # an advisory-only ranking signal ONLY when one of the two downgrade
        # paths proves clean execution over the KNOWN full expected set — either
        # the candidate's own full-scope quick pass (candidate.quick_verification)
        # or independent verifier evidence (verification.test_result). Both paths
        # now reject UNKNOWN / partial coverage (e.g. a capped quick-verification
        # subset that never ran the stub's tests): absence of coverage data is not
        # proof, and treating it as such nominated an incomplete candidate that
        # then scored 0 on the official audit.
        if self._candidate_full_scope_quick_pass_preserves_expected(
            candidate
        ) or self._stub_residue_has_clean_objective_evidence(verification):
            diagnostics["stub_residue_advisory"] = {
                "reason": (
                    "full-scope verification passed the scored objective; "
                    "residual stubs remain ranking and follow-up evidence only"
                )
            }
            candidate.selection_diagnostics = diagnostics
            return
        validity = getattr(candidate, "validity", None)
        if validity is not None:
            validity.quality_gate_passed = False
            if "stub_residue" not in validity.reasons:
                validity.reasons.append("stub_residue")
        verification.quality_gate_passed = False
        if "stub_residue" not in verification.validity_reasons:
            verification.validity_reasons.append("stub_residue")
        verification.accepted = False

    def _stub_residue_has_clean_objective_evidence(
        self,
        verification: VerificationResult,
    ) -> bool:
        """Return true when independent verification PROVES the scored objective.

        Stub residue is a useful quality and follow-up signal, but it is not
        itself evidence that the selected patch fails the task. Keep it hard until
        independent verifier-side evidence shows clean execution of the scored
        objective over the KNOWN full expected set. Unknown coverage (e.g. a
        capped quick-verification subset that never ran the stub's tests) is NOT
        proof and must not downgrade ground-truth incompleteness — previously an
        unknown-coverage subset with failed=0 nominated an incomplete candidate
        that scored 0 on the official audit.
        """
        if not self._verification_has_proven_full_expected_coverage(verification):
            return False
        if getattr(verification.test_result, "regression_passes", None) is False:
            return False
        return True

    def _candidate_infrastructure_edit_reason(
        self,
        candidate: RolloutResult,
        *,
        issue_plan: Optional["IssuePlan"],
        verification: Optional[VerificationResult] = None,
    ) -> Optional[str]:
        """Reject patches that touch test-infrastructure files (conftest.py,
        jest.config.*, Cargo.toml, pom.xml, etc.) unless those exact paths
        appear in the issue's ``incomplete_test_files`` allowlist.

        Discovers the framework via the language-neutral test-runner
        registry; works for any adapter (pytest, jest, vitest, mocha,
        ``go test``, ``cargo test``, JUnit, ...).

        Special case: ``conftest.py`` edits are allowed when the candidate
        has a strict-pass verification (``test_result.pass_rate >= 1.0`` and
        ``missing_expected_test_count == 0``). Conftest fixtures are
        legitimately part of repair work and rejecting on this alone wastes
        clean rollouts (see Tornado / Phase 2 10.H).
        """
        if not candidate.worktree_path:
            return None
        workspace = Path(candidate.worktree_path)
        if not workspace.exists():
            return None
        try:
            from ..core.test_runners import detect_adapter
        except ImportError:
            return None
        adapter = detect_adapter(workspace)
        if adapter is None:
            return None
        infrastructure = adapter.infrastructure_paths(workspace)
        if not infrastructure:
            return None
        _, allowed_test_files = _completion_test_edit_context(issue_plan)
        changed = set(self._candidate_changed_files(candidate))
        violations = sorted(
            path for path in changed if path in infrastructure and path not in allowed_test_files
        )
        if not violations:
            return None
        # Conftest-only special case: when the only infrastructure files
        # touched are conftest.py variants, allow them when the candidate
        # already strict-passes the regression suite (Phase 2 10.H).
        only_conftest = all(Path(path).name == "conftest.py" for path in violations)
        if only_conftest and self._candidate_strict_passes(
            verification,
            require_expected_coverage=self._gold_suite_visible_mode(issue_plan),
        ):
            return None
        return f"test-infrastructure files edited ({adapter.name}): " + ", ".join(violations[:6])

    @staticmethod
    def _candidate_strict_passes(
        verification: Optional[VerificationResult],
        *,
        require_expected_coverage: bool = False,
    ) -> bool:
        """True when the verifier reports a clean strict pass.

        A strict pass means: regression command exit 0 (i.e. ``regression_passes``
        is True), pass_rate >= 1.0, no missing expected tests, and (when
        expected coverage was tracked) coverage was preserved.
        """
        if verification is None:
            return False
        test_result = getattr(verification, "test_result", None)
        if test_result is None:
            return False
        if not bool(getattr(test_result, "regression_passes", False)):
            return False
        if int(getattr(test_result, "failed", 0) or 0) != 0:
            return False
        if int(getattr(test_result, "errors", 0) or 0) != 0:
            return False
        try:
            pass_rate = float(getattr(test_result, "pass_rate", 0.0) or 0.0)
        except (TypeError, ValueError):
            return False
        if pass_rate < 1.0:
            return False
        if int(getattr(test_result, "missing_expected_test_count", 0) or 0) != 0:
            return False
        expected_coverage_preserved = getattr(
            test_result,
            "expected_coverage_preserved",
            None,
        )
        if expected_coverage_preserved is False:
            return False
        if (
            require_expected_coverage
            and not PatchSelector._verification_has_proven_full_expected_coverage(verification)
        ):
            return False
        return True

    @staticmethod
    def _candidate_validity_payload(candidate: RolloutResult) -> dict[str, Any]:
        validity = getattr(candidate, "validity", None)
        if validity is None:
            return {}
        if hasattr(validity, "as_dict"):
            try:
                payload = validity.as_dict()
            except Exception:  # noqa: BLE001 - validity diagnostics are advisory here
                return {}
            return payload if isinstance(payload, dict) else {}
        return validity if isinstance(validity, dict) else {}

    @staticmethod
    def _candidate_full_scope_quick_pass_preserves_expected(
        candidate: RolloutResult,
    ) -> bool:
        quick_verification = (
            candidate.quick_verification if isinstance(candidate.quick_verification, dict) else {}
        )
        if str(quick_verification.get("scope") or "").strip() != "full_test_command":
            return False
        local_full_scope_pass = quick_verification_has_local_full_scope_pass(quick_verification)
        validity = PatchSelector._candidate_validity_payload(candidate)
        if validity and validity.get("expected_coverage_preserved") is False:
            return False
        validity_confirmed_pass = bool(
            validity.get("quick_verification_passed") is True
            and validity.get("eligible_for_submission") is True
            and validity.get("expected_coverage_preserved") is not False
            and int(validity.get("missing_expected_test_count") or 0) == 0
        )
        scored_expected_suite_pass = quick_verification_has_scored_expected_suite_pass(
            quick_verification
        )
        if not local_full_scope_pass and not scored_expected_suite_pass:
            return False
        if quick_verification.get("coverage_preserved") is False:
            return False

        expected_count = quick_verification.get("expected_test_count")
        passed = quick_verification.get("passed")
        failed = quick_verification.get("failed")
        errors = quick_verification.get("errors")
        if not all(isinstance(value, int) and value >= 0 for value in (passed, failed, errors)):
            has_case_outcomes = bool(
                quick_verification.get("passed_tests") or quick_verification.get("failed_tests")
            )
            if not validity_confirmed_pass and not has_case_outcomes:
                return False
        elif int(passed) <= 0 or int(failed) != 0 or int(errors) != 0:
            return False

        missing_expected = quick_verification.get("missing_expected_test_count")
        if isinstance(missing_expected, int) and missing_expected > 0:
            return False

        if (
            isinstance(expected_count, int)
            and not isinstance(expected_count, bool)
            and expected_count > 0
        ):
            matched_count = quick_verification.get("matched_expected_test_count")
            if not (
                isinstance(matched_count, int)
                and not isinstance(matched_count, bool)
                and matched_count >= expected_count
            ):
                return False

        expected_coverage_ratio = quick_verification_expected_coverage_ratio(quick_verification)
        if isinstance(expected_coverage_ratio, (int, float)):
            return float(expected_coverage_ratio) >= 0.999

        if validity_confirmed_pass:
            return True
        if (
            isinstance(expected_count, int)
            and not isinstance(expected_count, bool)
            and expected_count > 0
        ):
            return quick_verification.get("coverage_preserved") is True
        return quick_verification.get("coverage_preserved") is True

    @staticmethod
    def _copy_quick_verification_counts_to_test_result(
        test_result: Any,
        quick_verification: dict[str, Any],
    ) -> None:
        int_fields = {
            "passed": "passed",
            "failed": "failed",
            "errors": "errors",
            "expected_test_count": "expected_test_count",
            "matched_expected_test_count": "matched_expected_test_count",
            "missing_expected_test_count": "missing_expected_test_count",
            "collected_test_count": "collected_test_count",
        }
        for source_key, attr in int_fields.items():
            value = quick_verification.get(source_key)
            if isinstance(value, int) and value >= 0:
                setattr(test_result, attr, value)
        missing_ids = quick_verification.get("missing_expected_test_ids")
        if isinstance(missing_ids, list):
            test_result.missing_expected_test_ids = [
                str(test_id) for test_id in missing_ids if str(test_id or "").strip()
            ]
        for source_key, attr in {
            "test_inventory_framework": "test_inventory_framework",
            "test_inventory_language": "test_inventory_language",
            "test_inventory_source": "test_inventory_source",
            "test_inventory_collection_command": "test_inventory_collection_command",
        }.items():
            value = quick_verification.get(source_key)
            if isinstance(value, str) and value.strip():
                setattr(test_result, attr, value)

    def _reconcile_verification_with_candidate_handoff(
        self,
        candidate: RolloutResult,
        verification: VerificationResult,
    ) -> None:
        if not self._candidate_full_scope_quick_pass_preserves_expected(candidate):
            return
        test_result = getattr(verification, "test_result", None)
        if test_result is None:
            return

        quick_verification = (
            candidate.quick_verification if isinstance(candidate.quick_verification, dict) else {}
        )
        original = {
            "passed": int(getattr(test_result, "passed", 0) or 0),
            "failed": int(getattr(test_result, "failed", 0) or 0),
            "errors": int(getattr(test_result, "errors", 0) or 0),
            "expected_coverage_preserved": getattr(
                test_result,
                "expected_coverage_preserved",
                None,
            ),
            "expected_test_count": int(getattr(test_result, "expected_test_count", 0) or 0),
            "matched_expected_test_count": int(
                getattr(test_result, "matched_expected_test_count", 0) or 0
            ),
            "missing_expected_test_count": int(
                getattr(test_result, "missing_expected_test_count", 0) or 0
            ),
        }

        reconciled = False
        if getattr(test_result, "expected_coverage_preserved", None) is False:
            test_result.expected_coverage_preserved = True
            reconciled = True

        if bool(getattr(test_result, "regression_passes", False)) and (
            int(getattr(test_result, "failed", 0) or 0) > 0
            or int(getattr(test_result, "errors", 0) or 0) > 0
        ):
            self._copy_quick_verification_counts_to_test_result(
                test_result,
                quick_verification,
            )
            reconciled = True
        else:
            taxonomy = classify_candidate_verification(candidate, verification)
            if taxonomy.kind in {
                VerificationFailureKind.COLLECTION_FAILURE,
                VerificationFailureKind.HARNESS_CONFIG_FAILURE,
                VerificationFailureKind.TIMEOUT_INCONCLUSIVE,
                VerificationFailureKind.ENVIRONMENT_FAILURE,
            } and quick_verification_has_local_full_scope_pass(quick_verification):
                self._copy_quick_verification_counts_to_test_result(
                    test_result,
                    quick_verification,
                )
                test_result.expected_coverage_preserved = True
                test_result.regression_passes = True
                test_result.regression_inconclusive = False
                reconciled = True

        if not reconciled:
            for source_key, attr in {
                "expected_test_count": "expected_test_count",
                "matched_expected_test_count": "matched_expected_test_count",
                "missing_expected_test_count": "missing_expected_test_count",
                "collected_test_count": "collected_test_count",
            }.items():
                value = quick_verification.get(source_key)
                if isinstance(value, int) and value >= 0:
                    current = int(getattr(test_result, attr, 0) or 0)
                    if attr == "missing_expected_test_count":
                        if current != value:
                            setattr(test_result, attr, value)
                            reconciled = True
                    elif value > current:
                        setattr(test_result, attr, value)
                        reconciled = True

        if not reconciled:
            return
        compute_score = getattr(self.verifier, "_compute_score", None)
        if callable(compute_score):
            verification.overall_score = compute_score(verification)
        else:
            verification.overall_score = max(
                float(verification.overall_score or 0.0),
                self._fallback_verification_score(verification),
            )
        diagnostics = (
            dict(candidate.selection_diagnostics)
            if isinstance(candidate.selection_diagnostics, dict)
            else {}
        )
        selector_diag = (
            dict(diagnostics.get("selector"))
            if isinstance(diagnostics.get("selector"), dict)
            else {}
        )
        selector_diag["verification_reconciled_from_full_scope_handoff"] = True
        selector_diag["verification_reconciled_original"] = original
        selector_diag["verification_reconciled_overall_score"] = float(
            verification.overall_score or 0.0
        )
        diagnostics["selector"] = selector_diag
        candidate.selection_diagnostics = diagnostics
        metadata = (
            dict(candidate.search_metadata) if isinstance(candidate.search_metadata, dict) else {}
        )
        metadata["candidate_handoff_verification_reconciled"] = True
        candidate.search_metadata = metadata

    @staticmethod
    def _fallback_verification_score(verification: VerificationResult) -> float:
        if not verification.syntax_valid:
            return 0.0
        score = 0.0
        if verification.lint_clean and verification.lint_applied:
            score += 0.1
        test_result = verification.test_result
        if test_result is not None:
            if test_result.reproduction_passes:
                score += 0.35
            if test_result.regression_passes:
                score += 0.35
            elif test_result.regression_inconclusive:
                score += 0.15
            score += 0.1 * test_result.pass_rate
        if verification.cross_validation_scores:
            score += 0.1 * (
                sum(verification.cross_validation_scores)
                / len(verification.cross_validation_scores)
            )
        return max(0.0, min(float(score), 1.0))

    def _candidate_test_edit_policy_reason(
        self,
        candidate: RolloutResult,
        *,
        issue_plan: Optional["IssuePlan"],
        verification: Optional[VerificationResult] = None,
    ) -> Optional[str]:
        # Infrastructure-file gate fires regardless of completion-like
        # context: editing conftest.py / jest.config.* / Cargo.toml is
        # always suspicious unless the issue explicitly authorizes it
        # via incomplete_test_files.
        infra_reason = self._candidate_infrastructure_edit_reason(
            candidate, issue_plan=issue_plan, verification=verification
        )
        if infra_reason:
            return infra_reason
        completion_like, _ = _completion_test_edit_context(issue_plan)
        gold_suite_visible = self._gold_suite_visible_mode(issue_plan)
        if not completion_like and not gold_suite_visible:
            return None

        test_changes = {
            path
            for path in self._candidate_changed_files(candidate)
            if SelectionCritic._is_pytest_collected_test_path(path)
        }
        if not test_changes:
            return None

        unexpected_test_changes = self._candidate_unexpected_visible_test_changes(
            candidate,
            issue_plan=issue_plan,
        )
        if unexpected_test_changes:
            return "unexpected visible test edits: " + ", ".join(unexpected_test_changes)

        source_changes = {
            path
            for path in self._candidate_changed_files(candidate)
            if not SelectionCritic._is_pytest_collected_test_path(path)
        }
        if not source_changes:
            return "visible scaffold test edits require accompanying source changes"

        return None

    def _candidate_unexpected_visible_test_changes(
        self,
        candidate: RolloutResult,
        *,
        issue_plan: Optional["IssuePlan"],
    ) -> list[str]:
        completion_like, allowed_test_files = _completion_test_edit_context(issue_plan)
        if not completion_like and not self._gold_suite_visible_mode(issue_plan):
            return []

        dispositions = self._candidate_visible_test_edit_dispositions(
            candidate,
            issue_plan=issue_plan,
        )
        unexpected: list[str] = []
        for path, analysis in dispositions.items():
            if analysis.action == "allow":
                continue
            if path not in allowed_test_files or analysis.action in {"restore", "sanitize"}:
                unexpected.append(path)
        return sorted(unexpected)

    def _candidate_visible_test_edit_dispositions(
        self,
        candidate: RolloutResult,
        *,
        issue_plan: Optional["IssuePlan"],
    ) -> dict[str, VisibleTestEditDisposition]:
        completion_like, allowed_test_files = _completion_test_edit_context(issue_plan)
        if not completion_like and not self._gold_suite_visible_mode(issue_plan):
            return {}

        test_changes = sorted(
            {
                path
                for path in self._candidate_changed_files(candidate)
                if SelectionCritic._is_pytest_collected_test_path(path)
            }
        )
        if not test_changes:
            return {}

        if not candidate.worktree_path:
            return {
                path: VisibleTestEditDisposition(
                    action=("allow" if path in allowed_test_files else "restore"),
                    reason="No candidate worktree available for protected visible-test analysis.",
                )
                for path in test_changes
            }

        repo_root = Path(self.repo_path)
        candidate_root = Path(candidate.worktree_path)
        if not repo_root.exists() or not candidate_root.exists():
            return {
                path: VisibleTestEditDisposition(
                    action=("allow" if path in allowed_test_files else "restore"),
                    reason="Visible-test baseline or candidate workspace is unavailable.",
                )
                for path in test_changes
            }

        dispositions: dict[str, VisibleTestEditDisposition] = {}
        for path in test_changes:
            allow_placeholder_completion = path in allowed_test_files
            baseline_path = repo_root / path
            candidate_path = candidate_root / path
            try:
                baseline_text = baseline_path.read_text(errors="replace")
                candidate_text = candidate_path.read_text(errors="replace")
            except OSError:
                dispositions[path] = VisibleTestEditDisposition(
                    action="restore",
                    reason=f"{path} could not be read for protected visible-test analysis.",
                )
                continue
            dispositions[path] = analyze_visible_test_edit(
                rel_path=path,
                baseline_text=baseline_text,
                candidate_text=candidate_text,
                allow_placeholder_completion=allow_placeholder_completion,
            )
        return dispositions

    def _prefer_accepted_clusters(
        self,
        clusters: list[PatchCluster],
        *,
        issue_plan: Optional["IssuePlan"] = None,
    ) -> list[PatchCluster]:
        def expected_coverage_not_collapsed(cluster: PatchCluster) -> bool:
            preserved = getattr(
                getattr(cluster.verification, "test_result", None),
                "expected_coverage_preserved",
                None,
            )
            return (
                preserved is not False
                or self._candidate_full_scope_quick_pass_preserves_expected(cluster.representative)
            )

        policy_valid = [
            cluster
            for cluster in clusters
            if not self._candidate_test_edit_policy_reason(
                cluster.representative,
                issue_plan=issue_plan,
                verification=cluster.verification,
            )
            and expected_coverage_not_collapsed(cluster)
        ]
        if not policy_valid:
            # Phase 2 10.I: before returning empty (which would propagate to
            # ``orchestrator_failed`` selection), force-select a strict-pass
            # cluster only when no hard validity boundary rejected it. The
            # rescue is for soft policy collapse, not for patches that changed
            # protected tests or test-universe infrastructure.
            strict_pass_clusters = [
                cluster
                for cluster in clusters
                if PatchSelector._candidate_strict_passes(
                    cluster.verification,
                    require_expected_coverage=self._gold_suite_visible_mode(issue_plan),
                )
                and not self._candidate_test_edit_policy_reason(
                    cluster.representative,
                    issue_plan=issue_plan,
                    verification=cluster.verification,
                )
                and expected_coverage_not_collapsed(cluster)
            ]
            if strict_pass_clusters:
                logger.warning(
                    "force_selected_strict_pass: policy collapsed to empty "
                    "but %s cluster(s) strict-pass the verifier without hard "
                    "validity rejection; overriding soft policy filter to ship "
                    "them rather than orchestrator_failed.",
                    len(strict_pass_clusters),
                )
                for cluster in strict_pass_clusters:
                    diagnostics = (
                        dict(cluster.representative.selection_diagnostics)
                        if isinstance(cluster.representative.selection_diagnostics, dict)
                        else {}
                    )
                    selector_diag = (
                        dict(diagnostics.get("selector"))
                        if isinstance(diagnostics.get("selector"), dict)
                        else {}
                    )
                    selector_diag["force_selected_strict_pass"] = True
                    selector_diag.setdefault(
                        "force_selected_strict_pass_reason",
                        "policy_collapsed_to_empty_with_strict_pass_candidate",
                    )
                    diagnostics["selector"] = selector_diag
                    cluster.representative.selection_diagnostics = diagnostics
                accepted = [cluster for cluster in strict_pass_clusters if cluster.accepted]
                return accepted or strict_pass_clusters
            return []
        accepted = [cluster for cluster in policy_valid if cluster.accepted]
        return accepted or policy_valid

    def _verification_meets_acceptance_bar(
        self,
        candidate: RolloutResult,
        verification: VerificationResult,
        test_command: Optional[str],
        *,
        issue_plan: Optional["IssuePlan"] = None,
    ) -> bool:
        if not verification.syntax_valid or not verification.lint_clean:
            return False
        if verification.quality_gate_passed is False:
            return False

        if (
            isinstance(verification.prune_result, dict)
            and verification.prune_result.get("is_valid") is False
        ):
            return False

        if self._candidate_test_edit_policy_reason(
            candidate,
            issue_plan=issue_plan,
            verification=verification,
        ):
            return False

        test_result = verification.test_result
        # ``quick_verification`` is produced by APEX's rollout harness in the
        # target workspace. A clean full-scope pass with expected coverage is
        # stronger than stale residual reproduction checks, but it still cannot
        # override hard validity gates handled above.
        full_scope_quick_pass = self._candidate_full_scope_quick_pass_preserves_expected(candidate)
        if (
            self._gold_suite_visible_mode(issue_plan)
            and canonical_expected_test_count(issue_plan) > 0
            and not full_scope_quick_pass
            and not self._verification_has_proven_full_expected_coverage(verification)
        ):
            return False
        strong_full_suite_signal = quick_verification_has_strong_signal(
            candidate.quick_verification if isinstance(candidate.quick_verification, dict) else {},
            require_full_scope=True,
        )
        artifact = candidate.reproduction_artifact or {}
        if any(artifact.get(key) for key in ("command", "script_content", "script_path")):
            # Prefer verifier-side reproduction evidence when it is fresh, but
            # do not let a stale/narrow residual reproduction result invalidate
            # a rollout-harness full expected-suite pass.
            if test_result is None:
                return False
            if not test_result.reproduction_passes and not (
                self._reproduction_failure_is_inconclusive(test_result)
                and self._test_result_has_clean_authoritative_regression(test_result)
            ):
                if not full_scope_quick_pass:
                    return False

        if test_command:
            # Independent regression evidence is mandatory.
            if test_result is None:
                return False
            expected_coverage_preserved = getattr(
                test_result,
                "expected_coverage_preserved",
                None,
            )
            if expected_coverage_preserved is False:
                return False
            regression_failure_is_inconclusive = (
                test_result.regression_inconclusive
                and self._regression_failure_is_inconclusive(test_result)
            )
            taxonomy = classify_candidate_verification(candidate, verification)
            verifier_failure_is_non_code_inconclusive = taxonomy.kind in {
                VerificationFailureKind.COLLECTION_FAILURE,
                VerificationFailureKind.HARNESS_CONFIG_FAILURE,
                VerificationFailureKind.TIMEOUT_INCONCLUSIVE,
                VerificationFailureKind.ENVIRONMENT_FAILURE,
            }
            authoritative_quick_overrides_verifier_failure = bool(
                full_scope_quick_pass and verifier_failure_is_non_code_inconclusive
            )
            # A regression timeout is INCONCLUSIVE — we have no evidence
            # of regressions, but no evidence of safety either. Accept
            # only if the rollout's own full-scope quick verification
            # was strong-signal (independent corroboration that the
            # full suite would have passed) AND the targeted-failing-
            # tests scope also passed cleanly. Without both, fall
            # through to the strict regression_passes requirement.
            strong_targeted_signal = quick_verification_has_strong_signal(
                candidate.quick_verification
                if isinstance(candidate.quick_verification, dict)
                else {},
                require_full_scope=False,
            )
            if expected_coverage_preserved is True and (
                int(getattr(test_result, "failed", 0) or 0) > 0
                or int(getattr(test_result, "errors", 0) or 0) > 0
            ):
                if not (
                    (
                        regression_failure_is_inconclusive
                        and strong_full_suite_signal
                        and strong_targeted_signal
                    )
                    or authoritative_quick_overrides_verifier_failure
                ):
                    return False
            if not test_result.regression_passes:
                if (
                    not (
                        regression_failure_is_inconclusive
                        and strong_full_suite_signal
                        and strong_targeted_signal
                    )
                    and not full_scope_quick_pass
                ):
                    return False
            # The pass-rate threshold can be relaxed when the rollout's own
            # full-scope quick verification corroborates the verifier's
            # ``regression_passes`` result, because pytest summary parsing
            # is occasionally fragile (e.g., on collection errors). The
            # corroborating in-rollout claim plus the independent
            # verifier-side ``regression_passes`` is the ``and`` we trust,
            # not either signal alone.
            if not (strong_full_suite_signal or authoritative_quick_overrides_verifier_failure):
                if test_result.pass_rate < self.config.selection.min_test_pass_rate:
                    # When the regression suite was inconclusive, the
                    # verifier-side pass_rate is also unreliable — fall
                    # back on the strong targeted-tests signal as the
                    # acceptance evidence.
                    if not (
                        test_result.regression_inconclusive
                        and strong_full_suite_signal
                        and strong_targeted_signal
                    ):
                        return False

        return True

    @staticmethod
    def _reproduction_failure_is_inconclusive(test_result: object) -> bool:
        output = str(getattr(test_result, "reproduction_output", "") or "").lower()
        return any(
            marker in output
            for marker in (
                "timed out",
                "timeout",
                "time out",
                "deadline exceeded",
                "inconclusive",
            )
        )

    @staticmethod
    def _regression_failure_is_inconclusive(test_result: object) -> bool:
        output = str(getattr(test_result, "regression_output", "") or "").lower()
        return any(
            marker in output
            for marker in (
                "timed out",
                "timeout",
                "time out",
                "deadline exceeded",
                "inconclusive",
            )
        )

    @staticmethod
    def _test_result_has_clean_authoritative_regression(test_result: object) -> bool:
        if not bool(getattr(test_result, "regression_passes", False)):
            return False
        if int(getattr(test_result, "failed", 0) or 0) > 0:
            return False
        if int(getattr(test_result, "errors", 0) or 0) > 0:
            return False
        if getattr(test_result, "expected_coverage_preserved", None) is False:
            return False
        expected_count = int(getattr(test_result, "expected_test_count", 0) or 0)
        matched_count = int(getattr(test_result, "matched_expected_test_count", 0) or 0)
        missing_count = int(getattr(test_result, "missing_expected_test_count", 0) or 0)
        if expected_count > 0 and (missing_count > 0 or matched_count < expected_count):
            return False
        return True
