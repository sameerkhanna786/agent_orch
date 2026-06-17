#!/usr/bin/env python3
"""Phase A.5 — Harvest per-candidate ranking outcomes from a sweep dir.

The testgen ranking calibrator
(:mod:`apex.scripts.calibrate_testgen_ranking`) consumes a directory of
historical runs in the layout::

    runs_dir/<run_id>/apex_result.json

where each ``apex_result.json`` carries ``candidate_selection`` populated
by :func:`apex.evaluation.multi_candidate.summarize_candidate_selection`
(or its mirror inside ``diagnostics``).  The smoke sweep, however, writes
runs into a deeper benchmark-scoped layout::

    sweep_root/<benchmark>/<task_id>/apex_result.json
    sweep_root/<benchmark>/<task_id>/apex_output/apex_result.json   # in-container variants

This script normalises both layouts into the flat one expected by the
calibrator. It does not move or rewrite the source files; instead it
emits one JSONL row per (task_id, candidate_id) on stdout describing the
candidate's signal vector + the post-hoc winning candidate, suitable for
direct ingestion by downstream calibrators or for upload to a tracking
database.

Optionally (``--mirror-runs-dir <path>``) the script can also build a
flat ``<path>/<task_id>/apex_result.json`` mirror tree so the existing
``calibrate_testgen_ranking.py --runs-dir`` consumer keeps working
unchanged. The mirror writes a slimmed payload that contains only the
``candidate_selection`` block plus the run-level signals the calibrator
reads — enough to avoid re-implementing the loader.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

# Mirrors apex.scripts.calibrate_testgen_ranking._WEIGHT_KEYS for stable
# JSONL columns even when a candidate is missing some signals.
_RANKING_SIGNAL_KEYS: tuple[str, ...] = (
    "pass_at_1",
    "mutation_score",
    "coverage_delta",
    "oracle_grounding",
    "assertion_effect",
    "dual_state_score",
    "meaningful_test_count_log",
)


def _iter_apex_results(root: Path) -> Iterable[Path]:
    if not root.exists():
        return iter(())
    if root.is_file() and root.name == "apex_result.json":
        return iter([root])
    return root.rglob("apex_result.json")


def _candidate_block(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the candidate_selection sub-dict, looking in the canonical and
    diagnostic mirror locations the runners use today."""
    block = payload.get("candidate_selection")
    if isinstance(block, dict) and block:
        return block
    diagnostics = payload.get("diagnostics") or {}
    if isinstance(diagnostics, dict):
        block = diagnostics.get("candidate_selection")
        if isinstance(block, dict) and block:
            return block
    return {}


def _signals_from_candidate(candidate: dict[str, Any]) -> dict[str, float]:
    """Mirror of :func:`apex.scripts.calibrate_testgen_ranking._signals_from_candidate`.

    Inlined here so this script stays import-light (no apex package
    import) and can be invoked from a stripped-down environment.
    """
    import math

    return {
        "pass_at_1": float(candidate.get("unfiltered_pass_at_1") or 0.0),
        "mutation_score": float(candidate.get("mutation_score") or 0.0),
        "coverage_delta": float(candidate.get("coverage_delta") or 0.0),
        "oracle_grounding": float(candidate.get("oracle_grounding_score") or 0.0),
        "assertion_effect": float(candidate.get("assertion_effect_score") or 0.0),
        "dual_state_score": float(candidate.get("dual_state_score") or 0.0),
        "meaningful_test_count_log": math.log1p(
            max(0, int(candidate.get("meaningful_test_count") or 0))
        ),
    }


def _post_hoc_best(candidates: list[dict[str, Any]]) -> Optional[str]:
    if not candidates:
        return None

    def key(c: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(c.get("unfiltered_pass_at_1") or 0.0),
            float(c.get("mutation_score") or 0.0),
            float(c.get("oracle_grounding_score") or 0.0),
        )

    return str(max(candidates, key=key).get("candidate_id") or "")


def _infer_task_id(result_path: Path) -> str:
    skip_names = {"apex_output", "outputs", "output", "run", "."}
    parent = result_path.parent
    while parent.name in skip_names and parent.parent != parent:
        parent = parent.parent
    return parent.name


def _infer_benchmark(result_path: Path) -> str:
    skip_names = {"apex_output", "outputs", "output", "run", "."}
    parent = result_path.parent
    while parent.name in skip_names and parent.parent != parent:
        parent = parent.parent
    benchmark_dir = parent.parent
    name = (benchmark_dir.name or "").strip().lower()
    aliases = {
        "commit0_lite": "commit0",
        "commit0": "commit0",
        "swtbench_lite": "swt_bench",
        "swtbench": "swt_bench",
        "swt_bench_lite": "swt_bench",
        "testgeneval_lite": "testgeneval",
        "testgeneval": "testgeneval",
    }
    return aliases.get(name, name)


def _row_for_candidate(
    *,
    task_id: str,
    benchmark: str,
    selected_id: str,
    post_hoc_best: str,
    candidate: dict[str, Any],
    result_path: Path,
) -> dict[str, Any]:
    cid = str(candidate.get("candidate_id") or "")
    composite = candidate.get("composite_score")
    return {
        "task_id": task_id,
        "benchmark": benchmark,
        "candidate_id": cid,
        "selected": cid == selected_id and bool(selected_id),
        "is_post_hoc_best": cid == post_hoc_best and bool(post_hoc_best),
        "composite_score": float(composite) if isinstance(composite, (int, float)) else None,
        "signals": _signals_from_candidate(candidate),
        "result_path": str(result_path),
    }


def harvest_ranking_outcomes(
    root: Path,
    *,
    stream: Optional[Any] = None,
    err_stream: Optional[Any] = None,
    mirror_runs_dir: Optional[Path] = None,
) -> dict[str, int]:
    out_stream = stream if stream is not None else sys.stdout
    err = err_stream if err_stream is not None else sys.stderr

    summary = {
        "result_files": 0,
        "tasks_with_candidates": 0,
        "candidates_emitted": 0,
        "skipped_files": 0,
        "tasks_without_candidates": 0,
    }

    paths = sorted({p.resolve() for p in _iter_apex_results(root)})
    if not paths:
        err.write(f"WARNING: no apex_result.json under {root}\n")
        return summary

    for result_path in paths:
        summary["result_files"] += 1
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            err.write(f"WARNING: cannot parse {result_path}: {exc}\n")
            summary["skipped_files"] += 1
            continue
        if not isinstance(payload, dict):
            summary["skipped_files"] += 1
            continue
        block = _candidate_block(payload)
        candidates = block.get("candidates") if isinstance(block, dict) else None
        if not isinstance(candidates, list) or not candidates:
            summary["tasks_without_candidates"] += 1
            continue

        task_id = _infer_task_id(result_path)
        benchmark = _infer_benchmark(result_path)
        selected_id = str(block.get("selected_candidate") or "")
        post_hoc_best = _post_hoc_best([c for c in candidates if isinstance(c, dict)]) or ""

        emitted_for_task = 0
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            row = _row_for_candidate(
                task_id=task_id,
                benchmark=benchmark,
                selected_id=selected_id,
                post_hoc_best=post_hoc_best,
                candidate=candidate,
                result_path=result_path,
            )
            out_stream.write(json.dumps(row, sort_keys=True))
            out_stream.write("\n")
            summary["candidates_emitted"] += 1
            emitted_for_task += 1

        if emitted_for_task > 0:
            summary["tasks_with_candidates"] += 1

        # Optional mirror so the existing calibrate_testgen_ranking.py
        # --runs-dir consumer keeps working unchanged.
        if mirror_runs_dir is not None:
            target_dir = mirror_runs_dir / task_id
            target_dir.mkdir(parents=True, exist_ok=True)
            slim_payload = {
                "candidate_selection": block,
                "_source": str(result_path),
                "_benchmark": benchmark,
            }
            (target_dir / "apex_result.json").write_text(
                json.dumps(slim_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    return summary


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Benchmark sweep root (parent of <benchmark>/<task_id>/apex_result.json).",
    )
    parser.add_argument(
        "--mirror-runs-dir",
        type=Path,
        default=None,
        help="Optional flat directory to populate as <runs-dir>/<task_id>/apex_result.json. "
        "Use this to feed `calibrate_testgen_ranking.py --runs-dir <path>` directly.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary of the harvest.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    summary = harvest_ranking_outcomes(
        args.root,
        mirror_runs_dir=args.mirror_runs_dir,
    )
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    sys.stderr.write(json.dumps({"harvest_summary": summary}, sort_keys=True) + "\n")
    if summary["candidates_emitted"] == 0 and summary["result_files"] > 0:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
