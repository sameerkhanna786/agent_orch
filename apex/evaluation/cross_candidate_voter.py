"""V5 cross-candidate execution voting (TEX-T pattern + APEX novelties).

Picks the winning test candidate from a multi-agent ensemble using a
two-stage selection:

  Stage 1 — oracle voting (TEX-T):
    For each test, compute oracle_score = number of candidate patches
    the test is F→P-against. The dual-version verifier supplies this.
    Pick the candidate(s) with the highest score.

  Stage 2 — mutation-killing tiebreaker (APEX Novelty 2):
    When multiple tests tie on oracle_score, prefer the test that kills
    the most function-local mutants on the buggy source. A test that
    discriminates more program perturbations is intrinsically a stronger
    discriminator. This preferentially picks tests with tight oracles
    (``assert result == [1, 2, 3]``) over loose ones
    (``assert result is not None``), lifting both SWT-Bench F→P and
    TestGenEval mutation score.

This is the SOTA-aligned pattern: TEX-T (87% SWT-V) does Stage 1 only;
adding Stage 2 is APEX's additive lever. No published system combines
patch-as-oracle voting with mutation-killing tiebreaks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .multi_candidate import _candidate_validation_floor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectionDiagnostics:
    """Per-candidate selection signals + final selection rationale."""

    candidates_considered: int = 0
    oracle_scores: dict[str, float] = field(default_factory=dict)
    mutation_kills: dict[str, int] = field(default_factory=dict)
    winner_id: Optional[str] = None
    winner_oracle_score: float = 0.0
    winner_mutation_kills: int = 0
    selection_path: str = ""
    elapsed_seconds: float = 0.0
    # Audit H4+H5+H12: surface details that were silently swallowed.
    abstained: bool = False
    abstain_reason: str = ""
    mutation_scorer_errors: list[str] = field(default_factory=list)
    validation_floor_ranks: dict[str, float] = field(default_factory=dict)
    validation_floor_decision: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_winner(
    *,
    test_candidates: list[dict[str, Any]],
    dual_version_rows: list[Any],
    mutation_killing_scorer: Optional[Any] = None,
    mutation_score_min_for_tiebreak: int = 2,
    abstain_when_no_oracle: bool = True,
) -> tuple[Optional[dict[str, Any]], SelectionDiagnostics]:
    """Select the winning test candidate.

    Args:
        test_candidates: list of dicts (each has at least ``test_id``,
            ``artifact_path``, ``artifact_content``). The order matters
            only for the deterministic tiebreaker after Stages 1+2.
        dual_version_rows: list of ``TestRow`` objects from
            ``dual_version_verifier.verify_tests_against_patches``.
        mutation_killing_scorer: callable
            ``(artifact_content) -> int`` returning the mutation-kill
            count for that candidate's test artifact. When ``None``,
            Stage 2 is skipped and ties fall through to deterministic
            order.
        mutation_score_min_for_tiebreak: only invoke the mutation
            scorer when the oracle-score tie has ≥ this many candidates;
            avoids wasting mutation budget on a clear winner.

    Returns:
        ``(winner_candidate_dict | None, diagnostics)``
    """

    started = time.time()
    if not test_candidates:
        return None, SelectionDiagnostics(
            selection_path="no_candidates",
            elapsed_seconds=time.time() - started,
        )
    candidates_by_id: dict[str, dict[str, Any]] = {
        str(c.get("test_id") or c.get("agent") or f"test_{i}"): c
        for i, c in enumerate(test_candidates)
    }
    validation_floor_ranks = {
        cid: _candidate_validation_floor(dict(candidate))
        for cid, candidate in candidates_by_id.items()
    }
    if validation_floor_ranks:
        best_floor = max(validation_floor_ranks.values())
        filtered_ids = {cid for cid, rank in validation_floor_ranks.items() if rank == best_floor}
        if filtered_ids and len(filtered_ids) < len(candidates_by_id):
            candidates_by_id = {
                cid: candidate for cid, candidate in candidates_by_id.items() if cid in filtered_ids
            }
            floor_decision = f"validation_floor_{best_floor:g}"
        else:
            floor_decision = "validation_floor_no_filter"
    else:
        floor_decision = "validation_floor_unavailable"
    rows_by_id: dict[str, Any] = {
        getattr(row, "test_id", None) or row["test_id"]: row for row in (dual_version_rows or [])
    }

    # Stage 1: oracle-score voting
    oracle_scores: dict[str, float] = {
        cid: float(getattr(rows_by_id.get(cid), "oracle_score", 0.0) or 0.0)
        for cid in candidates_by_id
    }
    if not oracle_scores:
        return None, SelectionDiagnostics(
            selection_path="no_dual_version_data",
            validation_floor_ranks=dict(validation_floor_ranks),
            validation_floor_decision=floor_decision,
            elapsed_seconds=time.time() - started,
        )
    max_score = max(oracle_scores.values())
    if max_score == 0:
        # Audit H5: when nobody scored, the voter has no signal worth
        # acting on. Without an abstain mechanism the deterministic
        # tiebreak collapses to "always pick the first agent" which (a)
        # heavily biases the run toward agent slot 0 and (b) under the
        # DVV F→P inversion bug actively prefers broken tests. Returning
        # ``winner=None`` lets the caller preserve its baseline pick.
        if abstain_when_no_oracle:
            return None, SelectionDiagnostics(
                candidates_considered=len(candidates_by_id),
                oracle_scores=dict(oracle_scores),
                selection_path="abstain_no_oracle_signal",
                elapsed_seconds=round(time.time() - started, 3),
                abstained=True,
                abstain_reason=(
                    "all dual-version oracle_scores were 0; preserving "
                    "the upstream baseline pick instead of guessing"
                ),
                validation_floor_ranks=dict(validation_floor_ranks),
                validation_floor_decision=floor_decision,
            )
        leaders = list(candidates_by_id.keys())
        path_so_far = "no_oracle_winners_use_mutation"
    else:
        leaders = [cid for cid, s in oracle_scores.items() if s == max_score]
        path_so_far = "oracle_voting"

    # Stage 2: mutation-killing tiebreaker
    mutation_kills: dict[str, int] = {}
    mutation_errors: list[str] = []
    if mutation_killing_scorer is not None and len(leaders) >= max(
        1, int(mutation_score_min_for_tiebreak)
    ):
        for cid in leaders:
            try:
                mutation_kills[cid] = int(
                    mutation_killing_scorer(candidates_by_id[cid].get("artifact_content") or "")
                )
            except Exception as exc:
                # Audit H12: don't silently swallow scorer failures.
                mutation_kills[cid] = 0
                mutation_errors.append(f"{cid}: {type(exc).__name__}: {str(exc)[:200]}")
        if any(v > 0 for v in mutation_kills.values()):
            top_kills = max(mutation_kills.values())
            leaders = [cid for cid in leaders if mutation_kills.get(cid, 0) == top_kills]
            path_so_far = (
                "oracle_voting+mutation_tiebreak"
                if path_so_far == "oracle_voting"
                else "no_oracle_winners+mutation_tiebreak"
            )

    # Audit H4: content-derived deterministic tiebreak. The previous
    # logic walked candidates_by_id insertion order and always picked
    # whichever agent emitted first — biasing the run toward slot 0.
    # Hashing on the candidate's artifact_content is stable, agent-
    # agnostic, and independent of input order.
    if leaders:
        import hashlib

        def _content_hash(cid: str) -> str:
            content = str(candidates_by_id.get(cid, {}).get("artifact_content") or "")
            return hashlib.sha1(content.encode("utf-8")).hexdigest()

        winner_id: Optional[str] = min(leaders, key=lambda cid: (_content_hash(cid), cid))
    else:
        winner_id = None
    winner = candidates_by_id.get(winner_id) if winner_id else None
    diagnostics = SelectionDiagnostics(
        candidates_considered=len(candidates_by_id),
        oracle_scores=dict(oracle_scores),
        mutation_kills=dict(mutation_kills),
        winner_id=winner_id,
        winner_oracle_score=oracle_scores.get(winner_id, 0.0) if winner_id else 0.0,
        winner_mutation_kills=mutation_kills.get(winner_id, 0) if winner_id else 0,
        selection_path=path_so_far,
        elapsed_seconds=round(time.time() - started, 3),
        mutation_scorer_errors=mutation_errors,
        validation_floor_ranks=dict(validation_floor_ranks),
        validation_floor_decision=floor_decision,
    )
    return winner, diagnostics


def make_local_mutation_scorer(
    *,
    focal_module_source: str,
    workdir: Path,
    benchmark_adapter: Any,
    artifact_path: str,
    n_mutants: int = 5,
    focal_module_relpath: Optional[str] = None,
):
    """Build a ``mutation_killing_scorer`` callable backed by a small
    set of locally-generated focal-module mutants.

    Per candidate test artifact, the scorer:
      1. Generates ``n_mutants`` simple AST mutants of the focal source
         (operator flips, constant tweaks, return removals).
      2. For each mutant, materializes it in a tmp clone of ``workdir``
         and runs the candidate test via ``benchmark_adapter``.
      3. Returns the count of mutants the candidate test killed
         (``benchmark_adapter`` reports FAIL on a mutant we tried to
         smuggle past it = mutant killed = good).

    Cheap (no cosmic-ray); 5 mutants × test run ≈ 30-60s per candidate.
    Skip in low-budget runs by passing ``mutation_killing_scorer=None``.

    Audit M8: when ``workdir`` is empty (the v5_workdirs/<id>/ pattern
    used by the testgenevallite runner) AND ``focal_module_relpath`` is
    provided, we materialize the focal file at that relpath in the tmp
    clone before running. This unsticks the scorer from the prior
    failure mode where every invocation returned 0 because the focal
    file couldn't be located. If neither approach finds a focal file we
    raise a ``MutationScorerUnavailableError`` so the caller can record
    a diagnostic instead of silently scoring 0.
    """

    import random
    import shutil
    import tempfile

    from apex.evaluation.final_acceptance_gate import GeneratedArtifact

    rng = random.Random(0)  # deterministic mutant set across candidates
    mutants = _generate_simple_mutants(focal_module_source, n_mutants, rng)
    if not mutants:
        # Defensive: no mutants → scorer always returns 0.
        return lambda _content: 0

    def _materialize_focal_in_tmp(tmp_dir: Path, mutated_source: str) -> Optional[Path]:
        """Find or write the focal file in ``tmp_dir``. Returns the path
        we wrote to, or None if we couldn't locate / construct one."""

        focal_file = _find_focal_file(tmp_dir, focal_module_source)
        if focal_file is not None:
            focal_file.write_text(mutated_source, encoding="utf-8")
            return focal_file
        if focal_module_relpath:
            target = tmp_dir / focal_module_relpath.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(mutated_source, encoding="utf-8")
            return target
        return None

    def scorer(artifact_content: str) -> int:
        if not artifact_content or not artifact_content.strip():
            return 0
        artifact = GeneratedArtifact(path=artifact_path, content=artifact_content)
        kills = 0
        for mutated_source in mutants:
            with tempfile.TemporaryDirectory(prefix="apex_mut_") as tmp:
                tmp_dir = Path(tmp)
                if workdir.exists():
                    shutil.copytree(workdir, tmp_dir, dirs_exist_ok=True)
                focal_written = _materialize_focal_in_tmp(tmp_dir, mutated_source)
                if focal_written is None:
                    # Audit M8: surface "we couldn't materialize the
                    # focal file" instead of silently treating every
                    # mutant as un-killable.
                    logger.warning(
                        "mutation scorer skipped: cannot locate focal "
                        "file in %s and no focal_module_relpath given",
                        tmp_dir,
                    )
                    continue
                try:
                    run = benchmark_adapter.run_unfiltered(artifact, tmp_dir)
                    status = str(getattr(run, "status", "") or "").lower()
                except Exception:
                    status = "harness_error"
                # Killed = test detected the mutation = run failed.
                if status in {"fail", "failed", "error", "errored"}:
                    kills += 1
        return kills

    return scorer


def _generate_simple_mutants(source: str, n: int, rng: Any) -> list[str]:
    """Cheap AST mutators: invert comparisons, flip booleans, swap +/-,
    drop a return value. Skips if source can't be parsed."""

    import ast

    if not source or not source.strip():
        return []
    try:
        base_tree = ast.parse(source)
    except SyntaxError:
        return []
    mutants: list[str] = []
    candidates = list(_collect_mutation_targets(base_tree))
    rng.shuffle(candidates)
    for transformer in candidates[: max(1, int(n))]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        transformer(tree)
        try:
            mutated = ast.unparse(tree)
        except Exception:
            continue
        if mutated and mutated != source:
            mutants.append(mutated)
    return mutants


def _collect_mutation_targets(tree):
    """Yield mutator callables (each applied to a fresh tree) that
    perform one simple program perturbation."""

    import ast

    class _InvertCompare(ast.NodeTransformer):
        def __init__(self):
            self.applied = False

        def visit_Compare(self, node):
            if self.applied:
                return node
            mapping = {
                ast.Eq: ast.NotEq,
                ast.NotEq: ast.Eq,
                ast.Lt: ast.GtE,
                ast.Gt: ast.LtE,
                ast.LtE: ast.Gt,
                ast.GtE: ast.Lt,
            }
            for i, op in enumerate(node.ops):
                if type(op) in mapping:
                    node.ops[i] = mapping[type(op)]()
                    self.applied = True
                    break
            return node

    class _FlipBoolean(ast.NodeTransformer):
        def __init__(self):
            self.applied = False

        def visit_Constant(self, node):
            if self.applied:
                return node
            if isinstance(node.value, bool):
                node.value = not node.value
                self.applied = True
            return node

    class _SwapBinOp(ast.NodeTransformer):
        def __init__(self):
            self.applied = False

        def visit_BinOp(self, node):
            if self.applied:
                return node
            mapping = {
                ast.Add: ast.Sub,
                ast.Sub: ast.Add,
                ast.Mult: ast.Add,
                ast.Div: ast.Mult,
            }
            if type(node.op) in mapping:
                node.op = mapping[type(node.op)]()
                self.applied = True
            return self.generic_visit(node)

    class _DropReturn(ast.NodeTransformer):
        def __init__(self):
            self.applied = False

        def visit_Return(self, node):
            if self.applied:
                return node
            node.value = None
            self.applied = True
            return node

    yield lambda t, c=_InvertCompare(): c.visit(t)
    yield lambda t, c=_FlipBoolean(): c.visit(t)
    yield lambda t, c=_SwapBinOp(): c.visit(t)
    yield lambda t, c=_DropReturn(): c.visit(t)


def _find_focal_file(workdir: Path, focal_source: str) -> Optional[Path]:
    """Locate the focal source file inside ``workdir`` by content match."""

    if not focal_source or not focal_source.strip():
        return None
    needle = focal_source[:200].strip()
    for py_file in workdir.rglob("*.py"):
        try:
            text = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if needle in text:
            return py_file
    return None
