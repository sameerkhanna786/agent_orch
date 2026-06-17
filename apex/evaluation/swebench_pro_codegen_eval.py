"""SWE-Bench Pro codegen evaluation runner.

Sibling of ``swebench_pro_testgen_eval.py`` that drives the patch-only
workflow against the Pro dataset. Most of the heavy lifting still lives
in the existing ``SWEBenchProBenchmarkRunner`` (which already knows how
to talk to the per-instance Pro docker images and parse their
``output.json``); this CLI provides a thin codegen-shaped entrypoint so
operators can run::

    python -m apex.evaluation.swebench_pro_codegen_eval \\
        --config configs/benchmark_swebench_pro_max.json \\
        --output .apex_swebench_pro_codegen_<stamp>

…and get a per-task patch + Pro-harness verification without going
through the testgen branch.

The codegen flow is exactly what ``SWEBenchProBenchmarkRunner.run`` does
already (orchestrator.solve -> patch -> Pro docker eval), so this module
is a small adapter that builds the runner, invokes ``run``, and writes a
codegen-shaped summary alongside the existing benchmark report.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from .checkpointing import atomic_write_json
from .swebench_pro_benchmark import (
    SWEBENCH_AGENT_VISIBILITY_PUBLISHED_PARITY,
    SWEBENCH_PREPARE_REPO_FROM_DOCKER_IMAGE,
    SWEBENCH_PRO_DATASET_NAME,
    SWEBENCH_PRO_DATASET_SPLIT,
    SWEBENCH_PRO_DOCKERHUB_USERNAME,
    SWEBENCH_ROLLOUT_SELECTION_ORCHESTRATOR,
    SWEBenchProBenchmarkRunner,
    default_swebench_pro_output_dir,
)

logger = logging.getLogger("apex.evaluation.swebench_pro_codegen")


def _build_pro_runner(
    config_path: str,
    *,
    output_dir: Path,
    dataset_name: str,
    dataset_split: str,
    dockerhub_username: str,
    docker_platform: Optional[str],
    block_network: bool,
    agent_visibility_mode: str,
    rollout_selection_policy: str,
    prepare_repo_mode: str,
    scripts_cache_dir: Optional[str],
) -> SWEBenchProBenchmarkRunner:
    """Construct the Pro runner using the existing ApexConfig pipeline."""

    from ..core.config import ApexConfig

    config = ApexConfig.from_file(config_path)
    runner = SWEBenchProBenchmarkRunner(
        config,
        output_dir=output_dir,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dockerhub_username=dockerhub_username,
        scripts_cache_dir=scripts_cache_dir,
        docker_platform=docker_platform,
        block_network=block_network,
        agent_visibility_mode=agent_visibility_mode,
        rollout_selection_policy=rollout_selection_policy,
        prepare_repo_mode=prepare_repo_mode,
    )
    runner.config_source = str(Path(config_path).resolve())
    return runner


def run_pro_codegen_eval(
    *,
    config_path: str,
    output_dir: Path,
    dataset_name: str = SWEBENCH_PRO_DATASET_NAME,
    dataset_split: str = SWEBENCH_PRO_DATASET_SPLIT,
    dockerhub_username: str = SWEBENCH_PRO_DOCKERHUB_USERNAME,
    docker_platform: Optional[str] = None,
    block_network: bool = False,
    agent_visibility_mode: str = SWEBENCH_AGENT_VISIBILITY_PUBLISHED_PARITY,
    rollout_selection_policy: str = SWEBENCH_ROLLOUT_SELECTION_ORCHESTRATOR,
    prepare_repo_mode: str = SWEBENCH_PREPARE_REPO_FROM_DOCKER_IMAGE,
    scripts_cache_dir: Optional[str] = None,
    instances: Optional[list[str]] = None,
    repos: Optional[list[str]] = None,
    languages: Optional[list[str]] = None,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    """Run the Pro codegen workflow and return a summary dict.

    Delegates to :class:`SWEBenchProBenchmarkRunner` for the per-task
    work; this function adds a thin codegen-shaped summary on top.
    """

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = _build_pro_runner(
        config_path,
        output_dir=output_dir,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dockerhub_username=dockerhub_username,
        docker_platform=docker_platform,
        block_network=block_network,
        agent_visibility_mode=agent_visibility_mode,
        rollout_selection_policy=rollout_selection_policy,
        prepare_repo_mode=prepare_repo_mode,
        scripts_cache_dir=scripts_cache_dir,
    )
    started = time.time()
    report = runner.run(
        instances=instances,
        repos=repos,
        languages=languages,
        limit=limit,
    )
    finished = time.time()
    summary = {
        "status": (
            "ok"
            if report.completed_tasks == report.total_tasks and report.failed_tasks == 0
            else "partial"
        ),
        "entrypoint": "swebench-pro-codegen",
        "model_name": "apex",
        "dataset_name": report.dataset_name,
        "dataset_split": report.dataset_split,
        "task_count": report.total_tasks,
        "completed_tasks": report.completed_tasks,
        "solved_tasks": report.solved_tasks,
        "score": report.score,
        "score_percent": report.score_percent,
        "started_at": started,
        "finished_at": finished,
        "elapsed_seconds": finished - started,
        "benchmark_report_path": str(output_dir / "benchmark_report.json"),
    }
    atomic_write_json(output_dir / "codegen_summary.json", summary)
    return summary


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m apex.evaluation.swebench_pro_codegen_eval",
        description=(
            "Drive APEX through the SWE-Bench Pro patch-only (codegen) "
            "workflow. Sibling of swebench_pro_testgen_eval.py."
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--dataset-name", default=SWEBENCH_PRO_DATASET_NAME)
    parser.add_argument("--dataset-split", default=SWEBENCH_PRO_DATASET_SPLIT)
    parser.add_argument("--dockerhub-username", default=SWEBENCH_PRO_DOCKERHUB_USERNAME)
    parser.add_argument("--scripts-cache-dir", default=None)
    parser.add_argument("--docker-platform", default=None)
    parser.add_argument("--block-network", action="store_true")
    parser.add_argument(
        "--agent-visibility-mode",
        default=SWEBENCH_AGENT_VISIBILITY_PUBLISHED_PARITY,
    )
    parser.add_argument(
        "--rollout-selection-policy",
        default=SWEBENCH_ROLLOUT_SELECTION_ORCHESTRATOR,
    )
    parser.add_argument(
        "--prepare-repo-mode",
        default=SWEBENCH_PREPARE_REPO_FROM_DOCKER_IMAGE,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--repos", nargs="*")
    parser.add_argument("--languages", nargs="*")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=os.environ.get("APEX_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(list(argv if argv is not None else sys.argv[1:]))
    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
    else:
        from ..core.config import ApexConfig

        output_dir = default_swebench_pro_output_dir(ApexConfig.from_file(args.config))
    summary = run_pro_codegen_eval(
        config_path=args.config,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        dockerhub_username=args.dockerhub_username,
        docker_platform=args.docker_platform,
        block_network=bool(args.block_network),
        agent_visibility_mode=args.agent_visibility_mode,
        rollout_selection_policy=args.rollout_selection_policy,
        prepare_repo_mode=args.prepare_repo_mode,
        scripts_cache_dir=args.scripts_cache_dir,
        instances=list(args.instances or []) or None,
        repos=list(args.repos or []) or None,
        languages=list(args.languages or []) or None,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
