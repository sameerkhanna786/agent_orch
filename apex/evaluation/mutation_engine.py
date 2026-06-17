"""AST-based mutation engine for testgen quality evaluation.

Implements Stage 4 of ``test_generation_design.md`` (Mutation-Driven
Adversarial Discrimination), which previously existed only as a schema
field the test_writer agent self-reported. This module instead generates
plausible-but-incorrect mutants of the target source code, runs candidate
tests against each mutant, and reports a real mutation score.

A test "kills" a mutant if (a) it passed on the unmutated baseline and
(b) it fails or errors after the mutation is applied. This baseline-aware
classification prevents environment / flakiness failures from being
counted as kills.

Public API:
    generate_mutants(source_path, ...) -> list[Mutant]
        AST-walk a Python source file, produce a deterministic, capped
        list of mutants using PIT/Major/mutmut-style operators.

    evaluate_mutation_score(fixed_dir, mutants, test_paths, ...)
        For each mutant: write the mutated source in-place, run pytest
        on the candidate tests, classify the outcome, restore the
        original file. Return a MutationReport.

Design notes:
    * The engine deliberately mutates files in-place inside a single
      sandbox (`fixed_dir`) and reverts via cached file content rather
      than re-cloning per mutant. Cloning was measured at ~600ms per
      mutant on ansible/ansible; in-place mutation is ~5ms.
    * A pytest baseline run is performed once at the start so flakiness
      and pre-existing collection errors do not get attributed to the
      mutation. Only tests that PASSED on baseline can kill a mutant.
    * Each mutant runs under a per-mutant wall-clock budget (default 60s)
      so a mutation that triggers an infinite loop cannot stall the run.
"""

from __future__ import annotations

import ast
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)


# Decisive-Edge D.3: tag each mutant with its source family ("syntactic"
# from this module's AST-token operators, "semantic" from the higher-level
# operators in :mod:`apex.evaluation.semantic_mutations`). The mutation
# report carries this field per-mutant so reviewers can split kill rate
# by family.
MutationSource = Literal["syntactic", "semantic", "both"]


_SUPPORTED_LANGUAGES = frozenset({"python", "py", "python3"})
_DEFAULT_PER_MUTANT_TIMEOUT_SECONDS = 60.0
_DEFAULT_BASELINE_TIMEOUT_SECONDS = 300.0

# Phase 4.5: in-loop helper default. The legacy 15s was tight enough that
# slow imports (django, pandas, ansible test discovery) routinely timed out
# and got mis-scored as "survived". Bumped to 30s. Operators who want the
# old behavior can pass ``per_mutant_timeout_seconds=15.0`` explicitly or
# set ``OrchestrationConfig.mutation_per_mutant_timeout_seconds=15.0``.
_DEFAULT_IN_LOOP_PER_MUTANT_TIMEOUT_SECONDS = 30.0
_DEFAULT_IN_LOOP_BASELINE_TIMEOUT_SECONDS = 30.0

# Phase 4.5: when more than this fraction of mutants timed out we flag the
# whole suite with ``quality_concern_high_timeout_rate`` so reviewers know
# the score is not trustworthy.
_HIGH_TIMEOUT_RATE_THRESHOLD = 0.20


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Mutant:
    """A single source-level mutation."""

    operator: str
    source_path: str  # repo-relative
    line: int  # 1-indexed line of the mutated node
    col: int  # 1-indexed column
    original_snippet: str
    mutated_snippet: str
    mutated_source: str  # full mutated file content (ready to write)
    # Decisive-Edge D.3: ``"syntactic"`` (from this module's AST-token
    # operators) or ``"semantic"`` (from the higher-level operators in
    # :mod:`apex.evaluation.semantic_mutations`). Defaults to
    # ``"syntactic"`` for back-compat with callers that build Mutants
    # directly without specifying the kind.
    mutation_kind: str = "syntactic"


@dataclass
class MutantOutcome:
    mutant: Mutant
    status: str  # "killed" | "survived" | "error" | "timeout" | "no_baseline_pass"
    duration_seconds: float = 0.0
    killing_tests: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class MutationReport:
    source_paths: list[str]
    baseline_pass_count: int
    baseline_status: str  # "ok" | "no_tests" | "error" | "timeout"
    baseline_error: Optional[str]
    total_mutants: int
    killed: int
    survived: int
    errored: int
    timed_out: int
    no_baseline_pass: int
    mutation_score: float  # killed / max(total - errored - timed_out - no_baseline_pass, 1)
    per_mutant: list[MutantOutcome] = field(default_factory=list)
    duration_seconds: float = 0.0
    baseline_total_count: int = 0
    effective_mutation_evaluable: int = 0
    mutation_score_denominator: str = "unknown"
    baseline_failure_summary: list[dict[str, Any]] = field(default_factory=list)
    # Phase 4.5: cohort-poisoning fix. ``mutation_score`` is now the mean
    # over per-test kill rates of the BASELINE-PASSING tests (failing
    # tests contribute 0 to the mean but no longer zero the whole suite).
    # ``suite_mutation_health`` describes the regime:
    #   * "pass":         every baseline test passed; score is the suite
    #                     mean of per-test kill rates.
    #   * "partial":      some baseline tests failed; score is the mean
    #                     over passing tests (failing tests excluded).
    #   * "no_baseline":  zero baseline tests passed; score is 0.0
    #                     (preserved legacy behavior for this case).
    suite_mutation_health: str = "pass"
    n_passing_tests: int = 0
    n_failing_tests: int = 0
    quality_concerns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_paths": list(self.source_paths),
            "baseline_pass_count": self.baseline_pass_count,
            "baseline_total_count": self.baseline_total_count,
            "baseline_status": self.baseline_status,
            "metric_status": _mutation_metric_status(self),
            "baseline_error": self.baseline_error,
            "total_mutants": self.total_mutants,
            "killed": self.killed,
            "survived": self.survived,
            "errored": self.errored,
            "timed_out": self.timed_out,
            "no_baseline_pass": self.no_baseline_pass,
            "mutation_score": round(self.mutation_score, 4),
            "effective_mutation_evaluable": self.effective_mutation_evaluable,
            "mutation_score_denominator": self.mutation_score_denominator,
            "suite_mutation_health": self.suite_mutation_health,
            "n_passing_tests": self.n_passing_tests,
            "n_failing_tests": self.n_failing_tests,
            "quality_concerns": list(self.quality_concerns),
            "baseline_failure_summary": [dict(item) for item in self.baseline_failure_summary],
            "duration_seconds": round(self.duration_seconds, 3),
            "per_mutant": [
                {
                    "operator": o.mutant.operator,
                    "source_path": o.mutant.source_path,
                    "line": o.mutant.line,
                    "col": o.mutant.col,
                    "original_snippet": o.mutant.original_snippet,
                    "mutated_snippet": o.mutant.mutated_snippet,
                    # Decisive-Edge D.3: surface the mutant's family so
                    # reports can break down kill rate by source.
                    "mutation_kind": getattr(o.mutant, "mutation_kind", "syntactic"),
                    "status": o.status,
                    "duration_seconds": round(o.duration_seconds, 3),
                    "killing_tests": list(o.killing_tests)[:10],
                    "error": o.error,
                }
                for o in self.per_mutant
            ],
        }


def _mutation_metric_status(report: MutationReport) -> str:
    status = str(report.baseline_status or "").lower()
    denominator = str(report.mutation_score_denominator or "").lower()
    if report.effective_mutation_evaluable > 0 or denominator in {
        "all_baseline_passing_tests",
        "baseline_passing_subset",
    }:
        return "available"
    if status == "timeout":
        return "timeout"
    if status in {"no_mutants", "no_mutants_generated", "no_target_source_paths"}:
        return "unavailable"
    if status == "unsupported" or denominator.startswith("unsupported"):
        return "unsupported"
    if status in {"exception", "error"}:
        return "infra_error"
    if denominator == "none_no_baseline_passing_tests":
        return "unavailable"
    return "unknown"


# ---------------------------------------------------------------------------
# Mutation generation
# ---------------------------------------------------------------------------


_BOUNDARY_OPERATORS = {
    ast.Lt: ("<", "<=", ast.LtE),
    ast.LtE: ("<=", "<", ast.Lt),
    ast.Gt: (">", ">=", ast.GtE),
    ast.GtE: (">=", ">", ast.Gt),
    ast.Eq: ("==", "!=", ast.NotEq),
    ast.NotEq: ("!=", "==", ast.Eq),
}

_ARITH_OPERATORS = {
    ast.Add: ("+", "-", ast.Sub),
    ast.Sub: ("-", "+", ast.Add),
    ast.Mult: ("*", "/", ast.Div),
    ast.Div: ("/", "*", ast.Mult),
}


def generate_mutants(
    *,
    source_path: str | Path,
    source_text: Optional[str] = None,
    language: str = "python",
    max_mutants: int = 32,
    seed: int = 0,
    mutation_source: MutationSource = "both",
) -> list[Mutant]:
    """Walk an AST and emit a deterministic, capped list of mutants.

    The mutation operators are lifted from PIT / Major / mutmut and ranked
    informally by published killability. We apply each operator at every
    eligible node, then if the resulting candidate count exceeds
    ``max_mutants`` we sample uniformly using the supplied seed.

    Returns an empty list if the source cannot be parsed (silently — the
    caller's mutation-score will be 0.0 with ``total_mutants=0``).

    Phase I.5: non-Python languages dispatch to the tree-sitter
    backend (``mutation_engine_treesitter.generate_mutants_treesitter``)
    when tree-sitter is installed; Python continues to use the AST
    path for richer node coverage.

    Decisive-Edge D.3: ``mutation_source`` selects which mutation
    families to draw from.

      * ``"syntactic"``: legacy behavior — only AST-token operators
        from this module.
      * ``"semantic"``: only the higher-level operators from
        :mod:`apex.evaluation.semantic_mutations`.
      * ``"both"`` (default): union of the two, deduplicated by
        ``(line, col, operator, mutated_snippet)``. The cap
        (``max_mutants``) is applied after the union.

    Non-Python paths ignore ``mutation_source`` — the tree-sitter
    backend has no semantic operators today and the request silently
    falls back to the syntactic family.
    """
    normalized_language = (language or "").lower()
    if normalized_language not in _SUPPORTED_LANGUAGES:
        # Phase I.5: dispatch to tree-sitter for JS/TS/Go/Rust/Java/etc.
        # No semantic operators on non-Python yet — semantic-only or
        # both-mode requests degrade to syntactic to preserve callers.
        from .mutation_engine_treesitter import generate_mutants_treesitter

        return generate_mutants_treesitter(
            source_path=source_path,
            source_text=source_text,
            language=normalized_language,
            max_mutants=max_mutants,
            seed=seed,
        )
    src_path = Path(source_path)
    if source_text is None:
        try:
            source_text = src_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return []

    candidates: list[Mutant] = []
    repo_relative_path = str(source_path)

    if mutation_source in {"syntactic", "both"}:
        for node in ast.walk(tree):
            candidates.extend(_mutate_compare(node, src_text=source_text, path=repo_relative_path))
            candidates.extend(_mutate_binop(node, src_text=source_text, path=repo_relative_path))
            candidates.extend(_mutate_constant(node, src_text=source_text, path=repo_relative_path))
            candidates.extend(
                _mutate_predicate(node, src_text=source_text, path=repo_relative_path)
            )
            candidates.extend(_mutate_return(node, src_text=source_text, path=repo_relative_path))

    if mutation_source in {"semantic", "both"}:
        candidates.extend(
            _semantic_mutants_for_source(
                source_text=source_text,
                source_path=repo_relative_path,
            )
        )

    # Deduplicate by (operator, line, col, mutated_snippet) — the AST walk
    # can produce equivalent edits when, e.g., a Compare is nested inside a
    # boolean predicate that we also mutate as a whole. Including operator
    # in the key keeps "syntactic boundary swap" and "semantic comparison
    # swap" as distinct mutants when their text-rendering happens to match.
    seen: set[tuple[str, int, int, str]] = set()
    unique: list[Mutant] = []
    for m in candidates:
        key = (m.operator, m.line, m.col, m.mutated_snippet)
        if key in seen:
            continue
        seen.add(key)
        unique.append(m)

    if len(unique) <= max_mutants:
        return unique
    rng = random.Random(seed)
    sampled = rng.sample(unique, max_mutants)
    # Sort sampled by (line, col, operator) so test ordering and reports are
    # stable across runs even when sampling.
    sampled.sort(key=lambda m: (m.line, m.col, m.operator))
    return sampled


def _semantic_mutants_for_source(
    *,
    source_text: str,
    source_path: str,
) -> list[Mutant]:
    """Bridge from the semantic registry into ``Mutant`` records.

    Local import keeps the semantic_mutations module optional for tools
    that import this module purely for the syntactic path (e.g. legacy
    test fixtures that patch ``generate_mutants`` directly).
    """
    try:
        from .semantic_mutations import DEFAULT_REGISTRY, MUTATION_KIND
    except ImportError:  # pragma: no cover — defensive
        return []
    out: list[Mutant] = []
    for mutation in DEFAULT_REGISTRY.apply_to_source(source_text):
        # Render a short snippet from the description so the diagnostic
        # reports stay readable. The full mutated source is on the
        # Mutant for the runner.
        out.append(
            Mutant(
                operator=f"semantic_{mutation.name}",
                source_path=source_path,
                line=mutation.line,
                col=mutation.col,
                original_snippet="",
                mutated_snippet=mutation.description,
                mutated_source=mutation.mutated_source,
                mutation_kind=MUTATION_KIND,
            )
        )
    return out


def _replace_token_in_source(
    *,
    source_text: str,
    line: int,
    col: int,
    end_line: Optional[int],
    end_col: Optional[int],
    replacement: str,
) -> Optional[str]:
    """Replace a span [(line,col)..(end_line,end_col)) in source_text.

    Returns None if the position is out of range. Lines and columns are
    1-indexed for line, 0-indexed for col (matching ast node attributes).
    """
    lines = source_text.splitlines(keepends=True)
    if line < 1 or line > len(lines):
        return None
    if end_line is None:
        end_line = line
    if end_col is None:
        end_col = col + 1
    line_start_offsets = [0]
    for ln in lines:
        line_start_offsets.append(line_start_offsets[-1] + len(ln))
    start_offset = line_start_offsets[line - 1] + col
    end_offset = line_start_offsets[end_line - 1] + end_col
    if start_offset > end_offset or end_offset > len(source_text):
        return None
    return source_text[:start_offset] + replacement + source_text[end_offset:]


def _ops_for_node(node: ast.AST, attr: str) -> list[ast.AST]:
    value = getattr(node, attr, None)
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _mutate_compare(node: ast.AST, *, src_text: str, path: str) -> list[Mutant]:
    if not isinstance(node, ast.Compare) or not node.ops:
        return []
    out: list[Mutant] = []
    # We mutate only the FIRST comparator op to keep the mutant set tractable.
    op = node.ops[0]
    op_type = type(op)
    if op_type not in _BOUNDARY_OPERATORS:
        return out
    original_str, replacement_str, _ = _BOUNDARY_OPERATORS[op_type]
    # ast does not give us an op token range; reconstruct one by reading the
    # source slice between the first comparator's left expression and the
    # second comparator. We use ast.get_source_segment for the whole Compare
    # and substring-replace the *first* occurrence of original_str — that's
    # correct because the comparator op separates the left operand from the
    # first comparator.
    segment = ast.get_source_segment(src_text, node)
    if segment is None or original_str not in segment:
        return out
    mutated_segment = segment.replace(original_str, replacement_str, 1)
    if mutated_segment == segment:
        return out
    new_source = _splice_segment(
        source_text=src_text,
        node=node,
        new_segment=mutated_segment,
    )
    if new_source is None:
        return out
    out.append(
        Mutant(
            operator=f"boundary_{original_str}_to_{replacement_str}",
            source_path=path,
            line=getattr(node, "lineno", 0),
            col=(getattr(node, "col_offset", 0) or 0) + 1,
            original_snippet=segment,
            mutated_snippet=mutated_segment,
            mutated_source=new_source,
        )
    )
    return out


def _mutate_binop(node: ast.AST, *, src_text: str, path: str) -> list[Mutant]:
    if not isinstance(node, ast.BinOp):
        return []
    op_type = type(node.op)
    if op_type not in _ARITH_OPERATORS:
        return []
    original_str, replacement_str, _ = _ARITH_OPERATORS[op_type]
    segment = ast.get_source_segment(src_text, node)
    if segment is None:
        return []
    # Mutate only the *outer* operator. The simplest reliable way is to
    # render the BinOp's operands and concatenate with the new operator.
    left_seg = ast.get_source_segment(src_text, node.left)
    right_seg = ast.get_source_segment(src_text, node.right)
    if left_seg is None or right_seg is None:
        return []
    mutated_segment = f"({left_seg}) {replacement_str} ({right_seg})"
    new_source = _splice_segment(source_text=src_text, node=node, new_segment=mutated_segment)
    if new_source is None:
        return []
    return [
        Mutant(
            operator=f"arith_{original_str}_to_{replacement_str}",
            source_path=path,
            line=getattr(node, "lineno", 0),
            col=(getattr(node, "col_offset", 0) or 0) + 1,
            original_snippet=segment,
            mutated_snippet=mutated_segment,
            mutated_source=new_source,
        )
    ]


def _mutate_constant(node: ast.AST, *, src_text: str, path: str) -> list[Mutant]:
    if not isinstance(node, ast.Constant):
        return []
    value = node.value
    segment = ast.get_source_segment(src_text, node)
    if segment is None:
        return []
    mutations: list[tuple[str, str]] = []
    if value is True:
        mutations.append(("True_to_False", "False"))
    elif value is False:
        mutations.append(("False_to_True", "True"))
    elif isinstance(value, int) and not isinstance(value, bool):
        if value == 0:
            mutations.append(("int_0_to_1", "1"))
        elif value == 1:
            mutations.append(("int_1_to_0", "0"))
        else:
            mutations.append(("int_off_by_one", str(value + 1)))
    elif isinstance(value, str):
        # Skip docstrings (Expr-statement-of-Constant-string at module/func
        # top is a docstring in CPython conventions) — we approximate via
        # ``end_lineno - lineno > 0`` length and pure-whitespace check.
        if "\n" in value or not value.strip():
            return []
        mutations.append(("str_to_xyzzy", repr("xyzzy")))
    if not mutations:
        return []
    out: list[Mutant] = []
    for op_label, replacement in mutations:
        new_source = _splice_segment(source_text=src_text, node=node, new_segment=replacement)
        if new_source is None:
            continue
        out.append(
            Mutant(
                operator=f"constant_{op_label}",
                source_path=path,
                line=getattr(node, "lineno", 0),
                col=(getattr(node, "col_offset", 0) or 0) + 1,
                original_snippet=segment,
                mutated_snippet=replacement,
                mutated_source=new_source,
            )
        )
    return out


def _mutate_predicate(node: ast.AST, *, src_text: str, path: str) -> list[Mutant]:
    """Wrap the test-predicate of an if/while/assert with `not (...)`."""
    if not isinstance(node, (ast.If, ast.While, ast.Assert)):
        return []
    test = getattr(node, "test", None)
    if test is None:
        return []
    test_seg = ast.get_source_segment(src_text, test)
    if test_seg is None:
        return []
    mutated_segment = f"not ({test_seg})"
    new_source = _splice_segment(source_text=src_text, node=test, new_segment=mutated_segment)
    if new_source is None:
        return []
    kind = type(node).__name__.lower()
    return [
        Mutant(
            operator=f"negate_{kind}_predicate",
            source_path=path,
            line=getattr(test, "lineno", 0),
            col=(getattr(test, "col_offset", 0) or 0) + 1,
            original_snippet=test_seg,
            mutated_snippet=mutated_segment,
            mutated_source=new_source,
        )
    ]


def _mutate_return(node: ast.AST, *, src_text: str, path: str) -> list[Mutant]:
    if not isinstance(node, ast.Return):
        return []
    value = node.value
    if value is None:
        return []
    seg = ast.get_source_segment(src_text, value)
    if seg is None:
        return []
    # ``return None`` is the canonical "did the agent test the return value"
    # mutation. ``return not X`` is also useful for boolean-returning
    # functions but we keep V1 simple with one mutation per return.
    new_source = _splice_segment(source_text=src_text, node=value, new_segment="None")
    if new_source is None:
        return []
    return [
        Mutant(
            operator="return_value_to_none",
            source_path=path,
            line=getattr(value, "lineno", 0),
            col=(getattr(value, "col_offset", 0) or 0) + 1,
            original_snippet=seg,
            mutated_snippet="None",
            mutated_source=new_source,
        )
    ]


def _splice_segment(
    *,
    source_text: str,
    node: ast.AST,
    new_segment: str,
) -> Optional[str]:
    """Replace the slice of source_text covered by `node`'s position range
    with `new_segment`. Returns None on missing position info."""
    if (
        not hasattr(node, "lineno")
        or not hasattr(node, "col_offset")
        or not hasattr(node, "end_lineno")
        or not hasattr(node, "end_col_offset")
    ):
        return None
    end_lineno = getattr(node, "end_lineno", None)
    end_col = getattr(node, "end_col_offset", None)
    if end_lineno is None or end_col is None:
        return None
    return _replace_token_in_source(
        source_text=source_text,
        line=node.lineno,
        col=node.col_offset,
        end_line=end_lineno,
        end_col=end_col,
        replacement=new_segment,
    )


# ---------------------------------------------------------------------------
# Mutation evaluation
# ---------------------------------------------------------------------------


def _summarize_baseline_failures(run: Any, *, max_entries: int = 20) -> list[dict[str, Any]]:
    """Summarize baseline non-passing tests for diagnostics.

    The F2P runner already normalizes test nodeids and statuses, but it does
    not carry per-node failure excerpts. We still surface the failing nodeids
    and a shared tail excerpt so the next feedback loop can identify which
    generated tests poisoned mutation eligibility.
    """

    per_test_status = dict(getattr(run, "per_test_status", {}) or {})
    entries: list[dict[str, Any]] = []
    for nodeid, status in per_test_status.items():
        status_text = str(status or "unknown").lower()
        if status_text == "pass":
            continue
        entries.append(
            {
                "nodeid": str(nodeid),
                "status": status_text,
            }
        )
        if len(entries) >= max_entries:
            break
    if entries:
        excerpt = "\n".join(
            str(getattr(run, attr, "") or "")
            for attr in ("stdout_tail", "stderr_tail", "error")
            if str(getattr(run, attr, "") or "").strip()
        ).strip()
        if excerpt:
            for entry in entries:
                entry["output_excerpt"] = excerpt[-1200:]
        return entries
    status = str(getattr(run, "status", "") or "").strip()
    error = str(getattr(run, "error", "") or "").strip()
    if status or error:
        entry: dict[str, Any] = {"nodeid": "", "status": status or "unknown"}
        if error:
            entry["output_excerpt"] = error[-1200:]
        return [entry]
    return []


def evaluate_mutation_score(
    *,
    fixed_dir: str | Path,
    mutants: list[Mutant],
    test_paths: list[str],
    language: str = "python",
    python_executable: Optional[str] = None,
    per_mutant_timeout_seconds: float = _DEFAULT_PER_MUTANT_TIMEOUT_SECONDS,
    baseline_timeout_seconds: float = _DEFAULT_BASELINE_TIMEOUT_SECONDS,
) -> MutationReport:
    """Run candidate tests against each mutant and classify outcomes.

    A mutant is *killed* iff at least one test that PASSED on the unmutated
    baseline FAILS or ERRORS after the mutation is applied. Mutants whose
    pytest run errored out (e.g. SyntaxError introduced by the mutation,
    interpreter crash) are reported as ``status=error`` and excluded from
    the denominator.
    """
    started_at = time.time()
    fixed_path = Path(fixed_dir)
    source_paths = sorted({m.source_path for m in mutants})

    if not mutants:
        return MutationReport(
            source_paths=source_paths,
            baseline_pass_count=0,
            baseline_status="no_mutants",
            baseline_error=None,
            total_mutants=0,
            killed=0,
            survived=0,
            errored=0,
            timed_out=0,
            no_baseline_pass=0,
            mutation_score=0.0,
            duration_seconds=time.time() - started_at,
            baseline_total_count=0,
            effective_mutation_evaluable=0,
            mutation_score_denominator="no_mutants",
        )

    # Lazy: reuse the f2p_oracle runner stack so Python and non-Python
    # adapters share nodeid normalization, timeout handling, and reports.
    from apex.evaluation.f2p_oracle import (
        _resolve_test_runner_adapter,
        _run_tests_on_paths,
    )

    interpreter = python_executable or sys.executable or "python3"
    adapter = _resolve_test_runner_adapter(
        fixed_dir=fixed_path,
        language=(language or "python").lower(),
    )

    # Stage 1: baseline run on the unmutated fixed sandbox.
    baseline = _run_tests_on_paths(
        adapter=adapter,
        sandbox_dir=fixed_path,
        test_paths=test_paths,
        timeout_seconds=baseline_timeout_seconds,
        python_executable=interpreter,
    )
    baseline_passes: set[str] = {
        nodeid for nodeid, status in baseline.per_test_status.items() if status == "pass"
    }
    baseline_total_count = len(dict(baseline.per_test_status or {}))
    baseline_failure_summary = _summarize_baseline_failures(baseline)
    baseline_status = baseline.status
    baseline_error = baseline.error
    baseline_failing_tests = [
        nodeid
        for nodeid, status in (baseline.per_test_status or {}).items()
        if str(status or "").lower() != "pass"
    ]
    n_failing_tests = len(baseline_failing_tests)
    if not baseline_passes:
        # No baseline-passing test means *every* mutant is unkillable by
        # construction — record once and short-circuit per-mutant runs.
        # Phase 4.5: keep this branch's existing semantic
        # (mutation_score=0.0) but mark suite health as "no_baseline" so
        # downstream selectors can distinguish it from a real "tested
        # but couldn't kill anything" zero.
        return MutationReport(
            source_paths=source_paths,
            baseline_pass_count=0,
            baseline_status=baseline_status,
            baseline_error=baseline_error or "no_baseline_passing_tests",
            total_mutants=len(mutants),
            killed=0,
            survived=0,
            errored=0,
            timed_out=0,
            no_baseline_pass=len(mutants),
            mutation_score=0.0,
            per_mutant=[MutantOutcome(mutant=m, status="no_baseline_pass") for m in mutants],
            duration_seconds=time.time() - started_at,
            baseline_total_count=baseline_total_count,
            effective_mutation_evaluable=0,
            mutation_score_denominator="none_no_baseline_passing_tests",
            baseline_failure_summary=baseline_failure_summary,
            suite_mutation_health="no_baseline",
            n_passing_tests=0,
            n_failing_tests=n_failing_tests,
        )

    # Stage 2: per-mutant — write mutated source, run tests, restore.
    # Cache original file contents so we can revert deterministically even
    # if multiple mutants target the same file.
    originals: dict[str, str] = {}
    for sp in source_paths:
        target = fixed_path / sp
        try:
            originals[sp] = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            originals[sp] = ""

    outcomes: list[MutantOutcome] = []
    killed = survived = errored = timed_out = 0
    # Phase 4.5: track per-test kill counts so we can compute the
    # mean over baseline-passing tests of per-test kill rates (rather
    # than the legacy "any test killed it" suite-level binary).
    per_test_kill_counts: dict[str, int] = {nodeid: 0 for nodeid in baseline_passes}
    for mutant in mutants:
        target = fixed_path / mutant.source_path
        mutant_started = time.time()
        try:
            target.write_text(mutant.mutated_source, encoding="utf-8")
        except OSError as exc:
            outcomes.append(
                MutantOutcome(
                    mutant=mutant,
                    status="error",
                    duration_seconds=time.time() - mutant_started,
                    error=f"write_failed: {exc}",
                )
            )
            errored += 1
            continue
        try:
            run = _run_tests_on_paths(
                adapter=adapter,
                sandbox_dir=fixed_path,
                test_paths=test_paths,
                timeout_seconds=per_mutant_timeout_seconds,
                python_executable=interpreter,
            )
        finally:
            # Always revert before continuing — a leaked mutation would
            # corrupt the next mutant's baseline.
            try:
                target.write_text(originals.get(mutant.source_path, ""), encoding="utf-8")
            except OSError:
                logger.warning(
                    "Failed to restore %s after mutant %s — subsequent mutants may be corrupt.",
                    mutant.source_path,
                    mutant.operator,
                )

        if run.status == "timeout":
            outcomes.append(
                MutantOutcome(
                    mutant=mutant,
                    status="timeout",
                    duration_seconds=time.time() - mutant_started,
                    error=run.error,
                )
            )
            timed_out += 1
            continue
        if run.status == "exception":
            outcomes.append(
                MutantOutcome(
                    mutant=mutant,
                    status="error",
                    duration_seconds=time.time() - mutant_started,
                    error=run.error,
                )
            )
            errored += 1
            continue
        if run.status != "ok":
            outcomes.append(
                MutantOutcome(
                    mutant=mutant,
                    status="error",
                    duration_seconds=time.time() - mutant_started,
                    error=run.error or f"mutant_run_status={run.status}",
                )
            )
            errored += 1
            continue

        killing = sorted(
            nodeid
            for nodeid in baseline_passes
            if run.per_test_status.get(nodeid) in {"fail", None}
            and run.per_test_status.get(nodeid) != "pass"
        )
        # The "or None" arm of the comprehension above lets a *missing*
        # nodeid count as a kill — this catches mutants that broke test
        # collection itself (the test no longer ran but did pass before).
        if killing:
            killed += 1
            status = "killed"
            for nodeid in killing:
                per_test_kill_counts[nodeid] = per_test_kill_counts.get(nodeid, 0) + 1
        else:
            survived += 1
            status = "survived"
        outcomes.append(
            MutantOutcome(
                mutant=mutant,
                status=status,
                duration_seconds=time.time() - mutant_started,
                killing_tests=killing,
            )
        )

    classified_total = killed + survived
    # Phase 4.5: per-test kill rate denominator. Each baseline-passing test
    # could have killed up to ``classified_total`` mutants (the ones that
    # actually ran and didn't error/timeout). We compute the per-test
    # kill rate against this denominator and average across baseline-passing
    # tests. This decouples the suite mutation score from any single
    # poison-pill test and gives a more stable signal for the selector.
    if classified_total > 0 and baseline_passes:
        per_test_rates = [
            per_test_kill_counts.get(nodeid, 0) / classified_total for nodeid in baseline_passes
        ]
        mutation_score = sum(per_test_rates) / len(per_test_rates)
    else:
        mutation_score = 0.0

    quality_concerns: list[str] = []
    # Phase 4.5: if more than the threshold fraction of mutants timed out,
    # flag the suite. Timeouts are excluded from ``classified_total`` —
    # but a high timeout rate often signals a flaky import path or a
    # resource-starved sandbox, both of which make the score untrustworthy.
    timeout_pool = killed + survived + timed_out
    if timeout_pool > 0:
        timeout_rate = timed_out / timeout_pool
        if timeout_rate > _HIGH_TIMEOUT_RATE_THRESHOLD:
            quality_concerns.append("quality_concern_high_timeout_rate")

    # Phase 4.5: classify suite mutation health using the actual
    # baseline-failing-test count (not the legacy ``baseline_failure_summary``
    # heuristic, which fires a non-empty stub even when baseline is fully
    # green via its catch-all status fallback).
    if n_failing_tests > 0:
        denominator = "baseline_passing_subset"
        suite_mutation_health = "partial"
    elif baseline_total_count > 0:
        denominator = "all_baseline_passing_tests"
        suite_mutation_health = "pass"
    else:
        denominator = "unknown_baseline_test_count"
        suite_mutation_health = "pass"

    return MutationReport(
        source_paths=source_paths,
        baseline_pass_count=len(baseline_passes),
        baseline_status=baseline_status,
        baseline_error=baseline_error,
        total_mutants=len(mutants),
        killed=killed,
        survived=survived,
        errored=errored,
        timed_out=timed_out,
        no_baseline_pass=0,
        mutation_score=mutation_score,
        per_mutant=outcomes,
        duration_seconds=time.time() - started_at,
        baseline_total_count=baseline_total_count,
        effective_mutation_evaluable=classified_total,
        mutation_score_denominator=denominator,
        baseline_failure_summary=baseline_failure_summary,
        suite_mutation_health=suite_mutation_health,
        n_passing_tests=len(baseline_passes),
        n_failing_tests=n_failing_tests,
        quality_concerns=quality_concerns,
    )


# ---------------------------------------------------------------------------
# Convenience: extract source paths to mutate from a unified gold-patch diff.
# ---------------------------------------------------------------------------


def evaluate_mutation_sensitivity_in_loop(
    *,
    worktree_path: str | Path,
    test_paths: list[str],
    target_source_paths: list[str],
    language: str = "python",
    python_executable: Optional[str] = None,
    max_mutants_per_file: int = 2,
    max_files: int = 1,
    per_mutant_timeout_seconds: float = _DEFAULT_IN_LOOP_PER_MUTANT_TIMEOUT_SECONDS,
    baseline_timeout_seconds: float = _DEFAULT_IN_LOOP_BASELINE_TIMEOUT_SECONDS,
    seed: int = 0,
    mutation_source: MutationSource = "both",
) -> MutationReport:
    """In-loop mutation-sensitivity helper for the test_writer iteration loop.

    Today's :func:`evaluate_mutation_score` is designed for the post-rollout
    case where the sandbox has the gold patch applied (so the agent's tests
    PASS at baseline and a "kill" means the mutation broke the contract).
    Inside the test_writer iteration loop there is NO gold patch — the
    worktree is the broken state and the agent's tests are written to FAIL
    on it. Calling the existing engine in that setting short-circuits with
    ``baseline_status='no_baseline_passing_tests'`` and returns
    ``mutation_score=0.0`` regardless of how good the agent's tests are.

    This helper reuses the same baseline-aware classification but operates
    on the worktree directly. The reported ``killed`` count is a
    *sensitivity* signal:

      * baseline-passing-and-then-failing-under-mutation = sensitive
        (the test discriminates near-broken variants)
      * baseline-passing-and-still-passing-under-mutation = insensitive
        (the test tolerates the broken behavior — likely too loose)

    Tests that already FAILED on baseline are not classified — the agent's
    F2P-shaped tests should fail on broken by design and the mutation
    signal can't add information there.

    Tight default caps (``max_mutants_per_file=2``, ``max_files=1``,
    ``per_mutant_timeout_seconds=30``) keep the in-loop overhead under
    ~2 minutes per iteration on typical Python repos. Phase 4.5 bumped
    the per-mutant timeout from 15s → 30s because the legacy 15s caused
    slow imports (django, pandas, ansible test discovery) to time out
    and get mis-scored as "survived" — a false negative for the test
    suite. Operators who need the legacy budget can pass
    ``per_mutant_timeout_seconds=15.0`` explicitly. The Phase B
    ``evaluate_mutation_score`` defaults are 2-4x larger because they
    run once at selection time, not on every iteration.

    Generalizes outside benchmarks: ``target_source_paths`` is supplied by
    the caller (focus files from the issue plan, files imported by the
    agent's tests, or the gold patch's modified files when known). No
    benchmark task object required.
    """
    started_at = time.time()
    worktree = Path(worktree_path)
    selected_files = [str(p) for p in (target_source_paths or []) if p][:max_files]
    if not selected_files:
        return MutationReport(
            source_paths=[],
            baseline_pass_count=0,
            baseline_status="no_target_source_paths",
            baseline_error=None,
            total_mutants=0,
            killed=0,
            survived=0,
            errored=0,
            timed_out=0,
            no_baseline_pass=0,
            mutation_score=0.0,
            duration_seconds=time.time() - started_at,
            baseline_total_count=0,
            effective_mutation_evaluable=0,
            mutation_score_denominator="no_target_source_paths",
        )

    mutants: list[Mutant] = []
    for rel_path in selected_files:
        absolute = worktree / rel_path
        if not absolute.exists():
            continue
        file_mutants = generate_mutants(
            source_path=absolute,
            language=language,
            max_mutants=max_mutants_per_file,
            seed=seed,
            mutation_source=mutation_source,
        )
        # Re-key source_path to the worktree-relative form
        # evaluate_mutation_score expects.
        for m in file_mutants:
            m.source_path = rel_path
        mutants.extend(file_mutants)

    if not mutants:
        return MutationReport(
            source_paths=selected_files,
            baseline_pass_count=0,
            baseline_status="no_mutants_generated",
            baseline_error=None,
            total_mutants=0,
            killed=0,
            survived=0,
            errored=0,
            timed_out=0,
            no_baseline_pass=0,
            mutation_score=0.0,
            duration_seconds=time.time() - started_at,
            baseline_total_count=0,
            effective_mutation_evaluable=0,
            mutation_score_denominator="no_mutants_generated",
        )

    return evaluate_mutation_score(
        fixed_dir=worktree,
        mutants=mutants,
        test_paths=test_paths,
        language=language,
        python_executable=python_executable,
        per_mutant_timeout_seconds=per_mutant_timeout_seconds,
        baseline_timeout_seconds=baseline_timeout_seconds,
    )


_MUTATION_SOURCE_EXTENSIONS_BY_LANGUAGE: dict[str, tuple[str, ...]] = {
    "python": (".py",),
    "py": (".py",),
    "python3": (".py",),
    "javascript": (".js", ".jsx", ".mjs", ".cjs"),
    "js": (".js", ".jsx", ".mjs", ".cjs"),
    "jsx": (".js", ".jsx", ".mjs", ".cjs"),
    "typescript": (".ts", ".tsx", ".mts", ".cts"),
    "ts": (".ts", ".tsx", ".mts", ".cts"),
    "tsx": (".ts", ".tsx", ".mts", ".cts"),
    "go": (".go",),
    "golang": (".go",),
    "rust": (".rs",),
    "rs": (".rs",),
    "java": (".java",),
    "kotlin": (".kt", ".kts"),
    "kt": (".kt", ".kts"),
    "swift": (".swift",),
    "csharp": (".cs",),
    "cs": (".cs",),
    "c#": (".cs",),
    "php": (".php",),
    "ruby": (".rb",),
    "rb": (".rb",),
    "c": (".c", ".h"),
    "cpp": (".cc", ".cpp", ".cxx", ".hpp", ".h"),
    "c++": (".cc", ".cpp", ".cxx", ".hpp", ".h"),
    "cc": (".cc", ".cpp", ".cxx", ".hpp", ".h"),
}


def mutation_source_extensions_for_language(language: str) -> tuple[str, ...]:
    return _MUTATION_SOURCE_EXTENSIONS_BY_LANGUAGE.get(
        (language or "python").lower(),
        (".py",),
    )


def source_path_is_supported_for_mutation(path: str, *, language: str) -> bool:
    normalized = str(path or "").strip().lower()
    if not normalized:
        return False
    return normalized.endswith(mutation_source_extensions_for_language(language))


def source_paths_from_patch(patch_text: str, *, language: str = "python") -> list[str]:
    """Extract the set of files modified by a unified diff.

    Used by the testgen-eval integration: we want to mutate the files the
    gold patch touched, not random source in the repo, because those are
    the files the candidate tests are supposed to constrain.
    """
    paths: list[str] = []
    seen: set[str] = set()
    for line in (patch_text or "").splitlines():
        if not line.startswith("+++ "):
            continue
        target = line[4:].strip()
        # Strip the conventional "b/" prefix added by `git diff`.
        if target.startswith("b/"):
            target = target[2:]
        if target in {"/dev/null", ""} or target in seen:
            continue
        if not source_path_is_supported_for_mutation(target, language=language):
            continue
        seen.add(target)
        paths.append(target)
    return paths
