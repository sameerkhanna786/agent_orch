"""A fixed 16-task TestGenEvalLite smoke slice.

The slice is intentionally represented as filtering utilities rather than a
second benchmark runner. Official scoring still flows through
``runners.testgenevallite``; this module only creates a deterministic JSONL
subset and optional run config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from apex.evaluation.runners.testgenevallite import (
    TestGenEvalLiteRunConfig,
    run_testgenevallite,
)

TESTGEN_SMOKE_16_TASK_IDS: tuple[str, ...] = (
    "astropy__astropy-12907",
    "django__django-13925",
    "django__django-15252",
    "django__django-15388",
    "matplotlib__matplotlib-23913",
    "pallets__flask-5063",
    "pydata__xarray-3364",
    "pylint-dev__pylint-5859",
    "pylint-dev__pylint-7993",
    "pytest-dev__pytest-7432",
    "pytest-dev__pytest-8906",
    "scikit-learn__scikit-learn-14087",
    "sphinx-doc__sphinx-8721",
    "sympy__sympy-11400",
    "sympy__sympy-20639",
    "sympy__sympy-24152",
)


def load_prediction_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def select_smoke_records(
    records: Iterable[dict[str, Any]],
    *,
    task_ids: Iterable[str] = TESTGEN_SMOKE_16_TASK_IDS,
) -> list[dict[str, Any]]:
    wanted = {str(task_id) for task_id in task_ids}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        task_id = str(record.get("instance_id") or record.get("task_id") or "")
        if task_id in wanted and task_id not in seen:
            selected.append(dict(record))
            seen.add(task_id)
    return selected


def write_smoke_predictions_jsonl(
    *,
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    task_ids: Iterable[str] = TESTGEN_SMOKE_16_TASK_IDS,
) -> dict[str, Any]:
    records = select_smoke_records(
        load_prediction_records(input_jsonl),
        task_ids=task_ids,
    )
    output = Path(output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    missing = sorted(
        set(task_ids)
        - {str(record.get("instance_id") or record.get("task_id") or "") for record in records}
    )
    return {
        "status": "written",
        "path": str(output),
        "selected_count": len(records),
        "missing_task_ids": missing,
    }


def build_smoke_run_config(
    *,
    official_repo: str,
    predictions_jsonl: str,
    output_dir: str,
    model_name: str = "apex-smoke",
    task_parallelism: int = 1,
    timeout_seconds: int = 300,
) -> TestGenEvalLiteRunConfig:
    return TestGenEvalLiteRunConfig(
        official_repo=official_repo,
        predictions_jsonl=predictions_jsonl,
        output_dir=output_dir,
        model_name=model_name,
        task_parallelism=task_parallelism,
        timeout_seconds=timeout_seconds,
        task_ids=list(TESTGEN_SMOKE_16_TASK_IDS),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or run the Apex TestGenEvalLite smoke-16 slice."
    )
    parser.add_argument("--list-ids", action="store_true")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--official-repo")
    parser.add_argument("--output-dir")
    parser.add_argument("--model-name", default="apex-smoke")
    parser.add_argument("--task-parallelism", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.list_ids:
        print("\n".join(TESTGEN_SMOKE_16_TASK_IDS))
        return 0
    if args.input_jsonl and args.output_jsonl:
        result = write_smoke_predictions_jsonl(
            input_jsonl=args.input_jsonl,
            output_jsonl=args.output_jsonl,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    if args.run:
        if not args.official_repo or not args.output_jsonl or not args.output_dir:
            parser.error("--run requires --official-repo, --output-jsonl, and --output-dir")
        result = run_testgenevallite(
            build_smoke_run_config(
                official_repo=args.official_repo,
                predictions_jsonl=args.output_jsonl,
                output_dir=args.output_dir,
                model_name=args.model_name,
                task_parallelism=args.task_parallelism,
                timeout_seconds=args.timeout_seconds,
            )
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
