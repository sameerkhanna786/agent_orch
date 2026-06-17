"""Hierarchical gap-fill: add one test per uncovered focal symbol.

This is the lightweight half of V4 W7. Instead of restructuring generation
into a full per-test loop (which would require a new model-prompt schema),
we:

  1. Enumerate the focal symbols the API probe reports.
  2. Identify the subset that is *not yet referenced* in any test in the
     current artifact (a fast textual heuristic, language-agnostic).
  3. For each uncovered symbol, ask the LLM for ONE additional test.
  4. Validate each candidate test via :func:`apex.evaluation.atomic_acceptance.append_test_atomically`.
  5. Keep the candidate only when atomic acceptance passes.

This preserves the existing one-shot generation prompt (no architectural
breakage) and adds a budgeted, deterministic validator-gated tail. Disabled
by default; enable with ``APEX_HIERARCHICAL_GAP_FILL=1``.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GapFillOutcome:
    status: str
    appended_count: int = 0
    rejected_count: int = 0
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_uncovered_focal_symbols(
    artifacts: list[dict[str, Any]],
    api_probe: Any,
    *,
    skip_private: bool = True,
) -> list[str]:
    """Return focal-symbol names not referenced in any test in the artifacts."""

    referenced: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        text = str(artifact.get("content") or "")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            referenced.update(_lexical_name_refs(text))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                referenced.add(node.id)
            elif isinstance(node, ast.Attribute):
                referenced.add(node.attr)
    uncovered: list[str] = []
    for symbol in list(getattr(api_probe, "symbols", []) or []):
        name = str(getattr(symbol, "name", "") or "").strip()
        if not name:
            continue
        if skip_private and name.startswith("_"):
            continue
        if name in referenced:
            continue
        kind = str(getattr(symbol, "kind", "") or "")
        # Re-exports are surface API only — generating a test for them is
        # almost always low value.
        if kind == "reexport":
            continue
        uncovered.append(name)
    return uncovered


def apply_gap_fill(
    artifacts: list[dict[str, Any]],
    *,
    api_probe: Any,
    workdir: Path,
    benchmark_adapter: Any,
    request_one_test: Callable[[str, str], str],
    target_path: str,
    max_tests: int = 5,
    request_parallelism: int = 4,
) -> GapFillOutcome:
    """Walk the uncovered symbols and atomically append one test per symbol.

    Two-phase execution:

      Phase 1 (parallel): fan out ``request_one_test`` per uncovered symbol
      across ``request_parallelism`` threads. The slow LLM call is the
      bottleneck; each candidate sees the SAME ``base_text`` (the artifact
      before any gap-fill addition), so they're independent.

      Phase 2 (serial): walk candidates in symbol order, atomically appending
      each via ``append_test_atomically``. Atomic acceptance MUST run
      against the current base_text, which mutates as we accept tests, so
      this phase stays serial. The docker validation here is the cost we
      can't parallelize without losing the per-test isolation contract.

    Args:
        artifacts: current artifact list (each dict has at least ``path`` and
            ``content``).
        api_probe: APEX API probe result (has ``.symbols``).
        workdir: project workdir for the atomic verification subprocess.
        benchmark_adapter: adapter whose ``run_unfiltered`` decides whether
            an appended candidate test passes.
        request_one_test: callable ``(symbol_name, current_artifact_text) -> str``
            that returns Python source for a single test targeting the
            symbol. The caller owns the LLM/CLI client.
        target_path: path of the artifact to grow (must exist in artifacts).
        max_tests: how many uncovered symbols to attempt.
        request_parallelism: how many LLM requests to fan out concurrently
            in Phase 1. Defaults to 4 to roughly match the outer task
            parallelism without saturating the codex CLI process pool.
    """

    from concurrent.futures import ThreadPoolExecutor

    from .atomic_acceptance import append_test_atomically

    uncovered = find_uncovered_focal_symbols(artifacts, api_probe)[:max_tests]
    if not uncovered:
        return GapFillOutcome(status="no_uncovered_symbols", artifacts=artifacts)
    target_index = next(
        (
            i
            for i, art in enumerate(artifacts)
            if isinstance(art, dict) and str(art.get("path") or "") == target_path
        ),
        None,
    )
    if target_index is None:
        return GapFillOutcome(status="target_artifact_missing", artifacts=artifacts)
    target = dict(artifacts[target_index])
    base_text = str(target.get("content") or "")

    # Phase 1: parallel LLM fan-out per uncovered symbol.
    parallelism = max(1, min(int(request_parallelism or 1), len(uncovered)))
    candidates_by_symbol: dict[str, str | Exception] = {}
    initial_base_text = base_text  # snapshot for parallel calls

    def _call(sym: str) -> tuple[str, str | Exception]:
        try:
            return sym, request_one_test(sym, initial_base_text)
        except Exception as exc:  # pragma: no cover - LLM/CLI errors are diagnostic
            return sym, exc

    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        for sym, value in pool.map(_call, uncovered):
            candidates_by_symbol[sym] = value

    appended = 0
    rejected = 0
    diagnostics: list[dict[str, Any]] = []
    # Phase 2: serial atomic acceptance preserving per-test isolation.
    for symbol in uncovered:
        candidate_or_exc = candidates_by_symbol.get(symbol)
        if isinstance(candidate_or_exc, Exception):
            diagnostics.append(
                {
                    "symbol": symbol,
                    "status": "request_failed",
                    "error": f"{type(candidate_or_exc).__name__}: {candidate_or_exc}",
                }
            )
            rejected += 1
            continue
        candidate = candidate_or_exc or ""
        if not candidate or not candidate.strip():
            diagnostics.append({"symbol": symbol, "status": "empty_candidate"})
            rejected += 1
            continue
        result = append_test_atomically(
            base_text,
            candidate,
            benchmark_adapter=benchmark_adapter,
            workdir=workdir,
            path=target_path,
        )
        diagnostics.append(
            {
                "symbol": symbol,
                "status": result.status,
                "diagnostic": result.diagnostic,
            }
        )
        if result.accepted:
            base_text = result.artifact_text
            appended += 1
        else:
            rejected += 1
    target["content"] = base_text
    new_artifacts = list(artifacts)
    new_artifacts[target_index] = target
    return GapFillOutcome(
        status="ok" if appended or rejected else "no_attempts",
        appended_count=appended,
        rejected_count=rejected,
        artifacts=new_artifacts,
        diagnostics=diagnostics,
    )


def _lexical_name_refs(text: str) -> set[str]:
    """Best-effort fallback when AST parsing fails: scan for identifiers."""

    import re

    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text or ""))
