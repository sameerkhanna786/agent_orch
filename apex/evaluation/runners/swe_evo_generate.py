"""
CLI entry: generate SWE-EVO predictions through APEX's V5 in-container loop.

Mirrors the flag shape of :mod:`apex.evaluation.runners.testgenevallite_generate`
(plus a few SWE-EVO-specific knobs: ``--arrow-path``, ``--jsonl-path``,
``--skip-clone``, ``--repo``).

Output layout under ``--output-dir``:
  preds.json         — SWE-agent shaped predictions
  report.json        — APEX harness report
  records/           — per-task SWEEvoTaskResult dumps
  workspaces/        — per-task git checkouts (kept for postmortem unless
                       the user passes ``--cleanup-workspaces``)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from ..swe_evo_benchmark import (
    SWE_EVO_DATASET_NAME,
    SWE_EVO_DATASET_SPLIT,
    SWE_EVO_DEFAULT_INSTANCE_COUNT,
    SWEEvoHarness,
    SWEEvoHarnessConfig,
    SWEEvoTask,
    load_swe_evo_tasks,
)

logger = logging.getLogger("apex.evaluation.runners.swe_evo_generate")


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class SWEEvoGenerateConfig:
    """Parsed CLI config for one SWE-EVO generation run."""

    output_dir: str
    model_name: str = "apex-swe-evo"
    arrow_path: Optional[str] = None
    jsonl_path: Optional[str] = None
    dataset_name: str = SWE_EVO_DATASET_NAME
    dataset_split: str = SWE_EVO_DATASET_SPLIT
    task_ids: list[str] = field(default_factory=list)
    repos: list[str] = field(default_factory=list)
    limit: int = 0
    parallelism: int = 1  # accepted for parity; current driver is serial
    max_turns: int = 8
    per_tool_timeout_seconds: int = 60
    max_tool_output_bytes: int = 16_000
    score_per_intermediate_commit: bool = False
    skip_clone: bool = False
    include_intermediate_commits_in_prompt: bool = True
    cleanup_workspaces: bool = False
    config_path: Optional[str] = None  # APEX top-level json config (informational)
    overrides_path: Optional[str] = None  # SWE-EVO override file (informational)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SWE-EVO predictions through APEX's V5 in-container "
            "agent loop. Writes a SWE-agent shaped preds.json consumable "
            "by the official SWE-EVO harness "
            "(SWE-bench/evaluate_instance.py --scaffold SWE-agent)."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="apex-swe-evo")
    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument(
        "--arrow-path",
        default=None,
        help=(
            "Path to a SWE-EVO HF arrow file "
            "(e.g. .../hf_out/hf_dataset/test/data-00000-of-00001.arrow)."
        ),
    )
    src_group.add_argument(
        "--jsonl-path",
        default=None,
        help="Path to a SWE-EVO JSONL file (one task per line).",
    )
    parser.add_argument(
        "--dataset-name",
        default=SWE_EVO_DATASET_NAME,
        help="(Reserved) Dataset name for HuggingFace Hub fallback.",
    )
    parser.add_argument(
        "--dataset-split",
        default=SWE_EVO_DATASET_SPLIT,
        help="Dataset split label.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Filter to specific instance_id(s); repeatable.",
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help="Filter to specific repo(s); repeatable. Format: owner/name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(f"Cap task count (0 = no limit; full split = {SWE_EVO_DEFAULT_INSTANCE_COUNT})."),
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Reserved for forward compat; current loop is serial.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=8,
        help="Per-task in-container agent turn cap.",
    )
    parser.add_argument(
        "--per-tool-timeout-seconds",
        type=int,
        default=60,
        help="Hard timeout for each run_in_container shell call.",
    )
    parser.add_argument(
        "--max-tool-output-bytes",
        type=int,
        default=16_000,
        help="Per-tool stdout/stderr byte cap (output beyond is truncated).",
    )
    parser.add_argument(
        "--score-per-intermediate-commit",
        action="store_true",
        help="(Forward-compat; not yet wired.) Score against each PR checkpoint.",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help=(
            "Skip cloning the upstream repo; the agent loop runs against an "
            "empty workspace dir. Useful for dry-runs / docker-first flows."
        ),
    )
    parser.add_argument(
        "--no-intermediate-commits-in-prompt",
        dest="include_intermediate_commits_in_prompt",
        action="store_false",
        help=(
            "Omit the PR list from the agent prompt. Default ON — intermediate "
            "PRs are evolution evidence the agent benefits from."
        ),
    )
    parser.set_defaults(include_intermediate_commits_in_prompt=True)
    parser.add_argument(
        "--cleanup-workspaces",
        action="store_true",
        help="Delete the workspaces/ dir on success (saves disk).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to APEX-level config JSON (informational; recorded in report).",
    )
    parser.add_argument(
        "--overrides",
        default=None,
        help="Path to SWE-EVO override JSON (informational; recorded in report).",
    )
    return parser


def _parse_args(argv: list[str]) -> SWEEvoGenerateConfig:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return SWEEvoGenerateConfig(
        output_dir=args.output_dir,
        model_name=args.model_name,
        arrow_path=args.arrow_path,
        jsonl_path=args.jsonl_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        task_ids=list(args.task_id or []),
        repos=list(args.repo or []),
        limit=int(args.limit or 0),
        parallelism=int(args.parallelism or 1),
        max_turns=int(args.max_turns),
        per_tool_timeout_seconds=int(args.per_tool_timeout_seconds),
        max_tool_output_bytes=int(args.max_tool_output_bytes),
        score_per_intermediate_commit=bool(args.score_per_intermediate_commit),
        skip_clone=bool(args.skip_clone),
        include_intermediate_commits_in_prompt=bool(args.include_intermediate_commits_in_prompt),
        cleanup_workspaces=bool(args.cleanup_workspaces),
        config_path=args.config,
        overrides_path=args.overrides,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_tasks_from_config(cfg: SWEEvoGenerateConfig) -> list[SWEEvoTask]:
    iids = cfg.task_ids or None
    repos = cfg.repos or None
    limit = cfg.limit if cfg.limit > 0 else None
    if cfg.arrow_path:
        return load_swe_evo_tasks(
            arrow_path=cfg.arrow_path,
            instance_ids=iids,
            repos=repos,
            limit=limit,
        )
    if cfg.jsonl_path:
        return load_swe_evo_tasks(
            jsonl_path=cfg.jsonl_path,
            instance_ids=iids,
            repos=repos,
            limit=limit,
        )
    raise SystemExit(
        "swe_evo_generate: must pass --arrow-path or --jsonl-path. "
        "(HuggingFace Hub fetch not wired in V1; the dataset ships in-repo.)"
    )


def run_generate(config: SWEEvoGenerateConfig) -> dict[str, Any]:
    """Programmatic entrypoint mirrored after testgenevallite.run_generate."""
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    tasks = _load_tasks_from_config(config)
    if not tasks:
        return {
            "status": "no_tasks",
            "output_dir": output_dir,
            "model_name": config.model_name,
            "loaded": 0,
        }

    llm_config = None
    if config.config_path:
        from apex.core.config import ApexConfig

        apex_config = ApexConfig.from_file(config.config_path)
        if not apex_config.llm_configs:
            raise SystemExit(f"SWE-EVO config has no llm_configs: {config.config_path}")
        llm_config = apex_config.llm_configs[0]

    harness = SWEEvoHarness(
        output_dir=output_dir,
        config=SWEEvoHarnessConfig(
            model_name=config.model_name,
            max_turns=config.max_turns,
            per_tool_timeout_seconds=config.per_tool_timeout_seconds,
            max_tool_output_bytes=config.max_tool_output_bytes,
            score_per_intermediate_commit=config.score_per_intermediate_commit,
            skip_clone=config.skip_clone,
            include_intermediate_commits_in_prompt=(config.include_intermediate_commits_in_prompt),
        ),
        llm_config=llm_config,
    )
    report = harness.run(tasks)

    if config.cleanup_workspaces:
        ws = os.path.join(output_dir, "workspaces")
        if os.path.isdir(ws):
            shutil.rmtree(ws, ignore_errors=True)

    return {
        "status": "ok",
        "output_dir": output_dir,
        "model_name": config.model_name,
        "loaded": len(tasks),
        "succeeded": report.succeeded,
        "submission_ready": sum(1 for r in report.results if r.submission_ready),
        "success_authority": "official_swe_evo_evaluator_only",
        "failed": report.failed,
        "errored": report.errored,
        "success_rate": report.success_rate(),
        "preds_path": os.path.join(output_dir, "preds.json"),
        "report_path": os.path.join(output_dir, "report.json"),
    }


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=os.environ.get("APEX_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = _parse_args(list(argv or sys.argv[1:]))
    summary = run_generate(cfg)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
