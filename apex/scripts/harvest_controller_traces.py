#!/usr/bin/env python3
"""Phase A.5 — Harvest controller_decisions.jsonl traces from a benchmark sweep.

This is the bridge between the smoke sweep (writes one
``controller_decisions.jsonl`` per task) and the controller policy trainer
(:mod:`apex.scripts.train_controller_policy`).

Walks a benchmark sweep's output directory, finds every
``controller_decisions.jsonl`` (one per task is the canonical layout — the
runners write them under ``<run-dir>/<benchmark>/<task_id>/``), and emits a
single concatenated JSONL stream on stdout. Each emitted record is the
original JSONL line with two extra fields stamped on it:

    * ``task_id``   — derived from the parent directory's basename when not
                      already present in the record.
    * ``benchmark`` — derived from the grandparent directory's basename
                      (``commit0_lite``, ``swtbench_lite``, ``testgeneval_lite``)
                      when not already present.

That extra context is what the trainer uses to scope features to a regime
without losing track of which task each example came from.

The downstream consumer (``train_controller_policy.py --traces <file>``)
already accepts a path to a JSONL file — see that script's
``discover_trace_paths`` helper.

Missing files are reported on stderr but never abort the run; this
script is intentionally a best-effort scoop so a half-finished sweep
still produces actionable training data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

_BENCHMARK_DIR_ALIASES = {
    "commit0_lite": "commit0",
    "commit0": "commit0",
    "swtbench_lite": "swt_bench",
    "swtbench": "swt_bench",
    "swt_bench_lite": "swt_bench",
    "testgeneval_lite": "testgeneval",
    "testgeneval": "testgeneval",
    "swebench_pro": "swebench_pro",
    "swe_evo": "swe_evo",
}


def _normalize_benchmark_name(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    return _BENCHMARK_DIR_ALIASES.get(text.lower(), text.lower())


def _iter_trace_paths(root: Path) -> Iterable[Path]:
    """Yield every ``controller_decisions.jsonl`` under ``root`` (recursive)."""
    if not root.exists():
        return iter(())
    if root.is_file():
        # Operator passed a single jsonl path by accident — accept it.
        if root.name == "controller_decisions.jsonl":
            return iter([root])
        return iter(())
    return root.rglob("controller_decisions.jsonl")


def _infer_task_id(trace_path: Path) -> str:
    # <run-dir>/<benchmark>/<task_id>/controller_decisions.jsonl is the
    # canonical layout. Some runners nest the trace one level deeper
    # (e.g. <task_id>/apex_output/controller_decisions.jsonl); peel back
    # until we hit a directory whose name is not a generic apex output.
    skip_names = {"apex_output", "outputs", "output", "run", "."}
    parent = trace_path.parent
    while parent.name in skip_names and parent.parent != parent:
        parent = parent.parent
    return parent.name


def _infer_benchmark(trace_path: Path) -> str:
    parent = trace_path.parent
    skip_names = {"apex_output", "outputs", "output", "run", "."}
    while parent.name in skip_names and parent.parent != parent:
        parent = parent.parent
    benchmark_dir = parent.parent
    return _normalize_benchmark_name(benchmark_dir.name)


def _stamp_record(
    record: dict[str, Any],
    *,
    task_id: str,
    benchmark: str,
    trace_path: Path,
) -> dict[str, Any]:
    out = dict(record)
    if not out.get("task_id"):
        out["task_id"] = task_id
    if not out.get("benchmark") and benchmark:
        out["benchmark"] = benchmark
    if not out.get("trace_source"):
        out["trace_source"] = str(trace_path)
    return out


def harvest_traces(
    root: Path,
    *,
    stream: Optional[Any] = None,
    err_stream: Optional[Any] = None,
) -> dict[str, int]:
    """Stream every harvested record to ``stream`` (default sys.stdout).

    Returns a small summary dict ``{trace_files, records_emitted, skipped}``
    so callers (and tests) can assert on harvest size.
    """
    out_stream = stream if stream is not None else sys.stdout
    err = err_stream if err_stream is not None else sys.stderr

    summary = {
        "trace_files": 0,
        "records_emitted": 0,
        "skipped_files": 0,
        "malformed_lines": 0,
    }

    trace_paths = sorted({path.resolve() for path in _iter_trace_paths(root)})
    if not trace_paths:
        err.write(f"WARNING: no controller_decisions.jsonl files found under {root}\n")
        return summary

    for trace_path in trace_paths:
        summary["trace_files"] += 1
        try:
            text = trace_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            err.write(f"WARNING: cannot read {trace_path}: {exc}\n")
            summary["skipped_files"] += 1
            continue
        task_id = _infer_task_id(trace_path)
        benchmark = _infer_benchmark(trace_path)
        for line_no, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                summary["malformed_lines"] += 1
                err.write(f"WARNING: malformed JSON at {trace_path}:{line_no}; skipping\n")
                continue
            if not isinstance(record, dict):
                summary["malformed_lines"] += 1
                continue
            stamped = _stamp_record(
                record,
                task_id=task_id,
                benchmark=benchmark,
                trace_path=trace_path,
            )
            out_stream.write(json.dumps(stamped, sort_keys=True))
            out_stream.write("\n")
            summary["records_emitted"] += 1
    return summary


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Benchmark sweep root (parent of <benchmark>/<task_id>/...). "
        "All controller_decisions.jsonl files under it are concatenated.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary {trace_files, records_emitted, ...}.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    summary = harvest_traces(args.root)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    sys.stderr.write(json.dumps({"harvest_summary": summary}, sort_keys=True) + "\n")
    return 0 if summary["records_emitted"] > 0 or summary["trace_files"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
