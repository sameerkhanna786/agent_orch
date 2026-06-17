"""APEX command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from .core.config import (
    AGENT_MODE_CHOICES,
    GLOBAL_DEFAULT_AGENT_MODE,
    ApexConfig,
    LLMBackend,
    LLMConfig,
    apply_localizer_enforcement_override,
    normalize_supported_model_name,
)
from .evaluation.benchmark import BenchmarkRunner
from .evaluation.commit0_benchmark import Commit0BenchmarkRunner, default_commit0_output_dir
from .evaluation.compare import compare_benchmark_reports, render_benchmark_comparison_markdown
from .evaluation.swebench_pro_benchmark import (
    SWEBENCH_AGENT_VISIBILITY_BENCHMARK_AWARE,
    SWEBENCH_AGENT_VISIBILITY_ONLINE_FAIR,
    SWEBENCH_AGENT_VISIBILITY_PUBLISHED_PARITY,
    SWEBENCH_PRO_DATASET_NAME,
    SWEBENCH_PRO_DATASET_SPLIT,
    SWEBENCH_ROLLOUT_SELECTION_OFFICIAL_EVALUATOR,
    SWEBENCH_ROLLOUT_SELECTION_ORCHESTRATOR,
    SWEBenchProBenchmarkRunner,
    default_swebench_pro_output_dir,
)
from .operations import (
    archive_runs,
    cleanup_runs,
    compare_run_directories,
    doctor_summary,
    inspect_run_directory,
    render_doctor_report,
    render_matrix_report,
    render_run_compare,
    render_status_table,
    replay_failure,
    resume_run,
    retry_run,
    run_experiment_matrix,
    watch_run,
)
from .orchestrator import ApexOrchestrator
from .persistence import build_calibration_reports, render_reliability_markdown
from .preprocessing.repo_analyzer import RepoAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="apex",
        description="APEX: Adaptive Parallel Execution for Coding Agents.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    solve_parser = subparsers.add_parser("solve", help="Run the full APEX solve pipeline")
    solve_parser.add_argument("--repo", required=True, help="Path to the repository")
    solve_parser.add_argument(
        "--issue", required=True, help="Issue description or a file containing it"
    )
    solve_parser.add_argument("--config", default=None, help="Optional JSON config path")
    solve_parser.add_argument("--test-command", default=None, help="Command used to verify the fix")
    solve_parser.add_argument("--rollouts", type=int, default=None, help="Override rollout count")
    solve_parser.add_argument("--model", default=None, help="Override the primary model")
    solve_parser.add_argument("--output", default=None, help="Output directory")
    _add_agent_mode_argument(solve_parser, default=_default_agent_mode_for_subcommand("solve"))
    _add_hierarchical_v5_arguments(solve_parser)
    _add_abstention_threshold_argument(solve_parser)
    _add_benchmark_mode_argument(solve_parser)
    solve_parser.add_argument(
        "--docker-image",
        default=None,
        help=(
            "Docker image tag for --agent-mode in_container_v5. When set, the V5 agent "
            "runs inside a ContainerSupervisor with workspace bind-mounted at /workspace."
        ),
    )

    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze a repository and print the repo map"
    )
    analyze_parser.add_argument("--repo", required=True, help="Path to the repository")
    analyze_parser.add_argument("--output", default=None, help="Optional JSON output path")

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run APEX on local benchmark fixtures"
    )
    benchmark_parser.add_argument(
        "--fixtures",
        default=str(Path(__file__).resolve().parent.parent / "benchmarks" / "fixtures"),
        help="Directory containing benchmark fixtures",
    )
    benchmark_parser.add_argument("--config", default=None, help="Optional JSON config path")
    benchmark_parser.add_argument("--output", default=None, help="Directory for benchmark outputs")
    benchmark_parser.add_argument(
        "--rollouts", type=int, default=None, help="Override rollout count"
    )
    benchmark_parser.add_argument("--model", default=None, help="Override the primary model")
    benchmark_parser.add_argument(
        "--tasks", nargs="*", default=None, help="Optional subset of task names"
    )

    benchmark_compare_parser = subparsers.add_parser(
        "benchmark-compare",
        help="Compare two or more benchmark_report.json files on their shared task set",
    )
    benchmark_compare_parser.add_argument(
        "--reports",
        nargs="+",
        required=True,
        help="Benchmark report JSON files to compare. The first report is treated as the reference.",
    )
    benchmark_compare_parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels for the provided reports, in the same order.",
    )
    benchmark_compare_parser.add_argument(
        "--name",
        default=None,
        help="Optional comparison title for generated JSON/Markdown outputs.",
    )
    benchmark_compare_parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for the generated comparison JSON.",
    )
    benchmark_compare_parser.add_argument(
        "--output-markdown",
        default=None,
        help="Optional path for the generated comparison Markdown.",
    )

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run preflight checks for backend health, structured output, and benchmark prerequisites",
    )
    doctor_parser.add_argument("--config", default=None, help="Optional JSON config path")
    doctor_parser.add_argument(
        "--skip-smoke-tests",
        action="store_true",
        help="Skip structured-output backend smoke tests",
    )
    doctor_parser.add_argument(
        "--skip-cli-smoke-tests",
        action="store_true",
        help=(
            "Skip ALL CLI startup health probes (claude/codex/gemini --help/--version) "
            "in addition to --skip-smoke-tests. Useful when one of the installed CLIs "
            "is wrapped in an interactive alias that hangs the probe (e.g. claude "
            "wrapped with --teammate-mode tmux). With this flag the doctor only "
            "verifies binaries are on PATH via shutil.which()."
        ),
    )
    doctor_parser.add_argument("--output-json", default=None, help="Optional JSON output path")

    # Phase 5.7: docker / worktree / result-dir leak audit. Three small
    # subcommands; we register them as flat top-level commands so they
    # don't disturb the existing ``doctor`` subparser shape.
    doctor_scan_parser = subparsers.add_parser(
        "doctor-scan",
        help="Scan for leaked apex_* docker containers and worktree/.apex_* dirs older than --age",
    )
    doctor_scan_parser.add_argument(
        "--age",
        type=float,
        default=7.0,
        help="Age threshold in days; entries older than this count as leaks (default: 7)",
    )
    doctor_scan_parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="Filesystem root(s) to scan for worktree/result dirs (default: cwd). Repeatable.",
    )
    doctor_scan_parser.add_argument("--output-json", default=None, help="Optional JSON output path")
    doctor_scan_parser.add_argument(
        "--assert-zero-leaks",
        action="store_true",
        help="Exit 1 if leak_count > 0 (intended for CI post-test gating)",
    )

    doctor_clean_parser = subparsers.add_parser(
        "doctor-clean",
        help="Remove leaked apex_* docker containers and worktree/.apex_* dirs (dry-run by default)",
    )
    doctor_clean_parser.add_argument(
        "--age",
        type=float,
        default=7.0,
        help="Age threshold in days; entries older than this are removed (default: 7)",
    )
    doctor_clean_parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="Filesystem root(s) to scan/clean (default: cwd). Repeatable.",
    )
    doctor_clean_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually remove. Without this flag the command stays in dry-run mode.",
    )
    doctor_clean_parser.add_argument(
        "--output-json", default=None, help="Optional JSON output path"
    )

    doctor_verify_manifest_parser = subparsers.add_parser(
        "doctor-verify-manifest",
        help="Re-inspect docker images recorded in a run's run_manifest.json and report digest drift",
    )
    doctor_verify_manifest_parser.add_argument(
        "--run",
        required=True,
        help="Run directory containing run_manifest.json",
    )
    doctor_verify_manifest_parser.add_argument(
        "--output-json", default=None, help="Optional JSON output path"
    )

    status_parser = subparsers.add_parser(
        "status",
        help="Inspect a run directory and render live task/rollout status",
    )
    status_parser.add_argument("--run", required=True, help="Run directory to inspect")
    status_parser.add_argument("--output-json", default=None, help="Optional JSON output path")

    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume a partial run from its existing artifacts and checkpoints",
    )
    resume_parser.add_argument("--run", required=True, help="Run directory to resume")
    resume_parser.add_argument("--dry-run", action="store_true", help="Show what would be resumed")
    resume_parser.add_argument(
        "--force", action="store_true", help="Resume even if the run appears active"
    )
    resume_parser.add_argument("--output-json", default=None, help="Optional JSON output path")

    retry_parser = subparsers.add_parser(
        "retry",
        help="Retry failed or suspicious tasks in-place while preserving healthy checkpoints",
    )
    retry_parser.add_argument("--run", required=True, help="Run directory to retry")
    retry_parser.add_argument("--failed-only", action="store_true", help="Retry only failed tasks")
    retry_parser.add_argument(
        "--suspicious-only", action="store_true", help="Retry only suspicious/incomplete tasks"
    )
    retry_parser.add_argument(
        "--tasks", nargs="*", default=None, help="Optional explicit task ids to retry"
    )
    retry_parser.add_argument("--dry-run", action="store_true", help="Show what would be retried")
    retry_parser.add_argument(
        "--force", action="store_true", help="Retry even if the run appears active"
    )
    retry_parser.add_argument("--output-json", default=None, help="Optional JSON output path")

    watch_parser = subparsers.add_parser(
        "watch",
        help="Continuously render live run status, failures, recoveries, and backend health",
    )
    watch_parser.add_argument("--run", required=True, help="Run directory to watch")
    watch_parser.add_argument(
        "--refresh-seconds", type=float, default=2.0, help="Refresh interval in seconds"
    )
    watch_parser.add_argument(
        "--iterations", type=int, default=None, help="Optional number of refreshes before exiting"
    )
    watch_parser.add_argument(
        "--no-clear", action="store_true", help="Append frames instead of clearing the terminal"
    )
    watch_parser.add_argument(
        "--output-json", default=None, help="Optional JSON output path for the final frame"
    )

    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help="Kill orphan Apex worker processes and remove stale run workspaces/runtime dirs",
    )
    cleanup_parser.add_argument("--runs", nargs="+", required=True, help="Run directories to clean")
    cleanup_parser.add_argument("--dry-run", action="store_true", help="Show what would be cleaned")
    cleanup_parser.add_argument("--output-json", default=None, help="Optional JSON output path")

    run_compare_parser = subparsers.add_parser(
        "run-compare",
        help="Compare two run directories by manifest and benchmark summary",
    )
    run_compare_parser.add_argument("--left", required=True, help="Reference run directory")
    run_compare_parser.add_argument("--right", required=True, help="Candidate run directory")
    run_compare_parser.add_argument("--output-json", default=None, help="Optional JSON output path")

    replay_parser = subparsers.add_parser(
        "replay",
        help="Replay one failed task or one failure cluster with the original run config",
    )
    replay_parser.add_argument("--run", required=True, help="Source run directory")
    replay_parser.add_argument("--task", default=None, help="Single failed task id to replay")
    replay_parser.add_argument("--cluster", default=None, help="Failure cluster bucket to replay")
    replay_parser.add_argument(
        "--output", default=None, help="Optional output directory for the replay run"
    )
    replay_parser.add_argument("--dry-run", action="store_true", help="Show what would be replayed")
    replay_parser.add_argument("--output-json", default=None, help="Optional JSON output path")

    # Phase 6.6: deterministic-replay debugging.
    replay_det_parser = subparsers.add_parser(
        "replay-deterministic",
        help=(
            "Deterministic replay of a recorded rollout (LLM + tool calls). "
            "Substitutes recorded responses for live LLM/tool calls."
        ),
    )
    replay_det_parser.add_argument(
        "record_path",
        help="Path to a JSONL record produced by ReplayRecorder",
    )
    replay_det_parser.add_argument(
        "--mutate-turn",
        type=int,
        default=None,
        help=(
            "Turn index whose prompt should be replaced with --mutate-prompt. "
            "Once that turn fires, replay falls back to live LLM calls."
        ),
    )
    replay_det_parser.add_argument(
        "--mutate-prompt",
        default=None,
        help="Replacement prompt text used with --mutate-turn",
    )
    replay_det_parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Strict verify mode: re-record while replaying and fail if "
            "the new recording diverges from --record-path"
        ),
    )
    replay_det_parser.add_argument(
        "--verify-against",
        default=None,
        help=(
            "With --verify, diff against this candidate JSONL recording "
            "instead of re-recording at runtime"
        ),
    )
    replay_det_parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON path for the structured replay summary",
    )

    # Phase 6.7: reviewer-mode benchmark publication.
    publish_parser = subparsers.add_parser(
        "publish-benchmark",
        help=(
            "Build a self-contained reviewer publication bundle from an "
            "APEX benchmark run directory."
        ),
    )
    publish_parser.add_argument(
        "run_dir",
        help="Path to an APEX run directory (contains apex_run_manifest.json)",
    )
    publish_parser.add_argument(
        "--output",
        default=None,
        help="Output directory for the publication bundle",
    )
    publish_parser.add_argument(
        "--include-fairness-audit",
        action="store_true",
        help="Include the fairness-audit deltas in RESULTS.md (opt-in)",
    )
    publish_parser.add_argument(
        "--validate",
        action="store_true",
        help=(
            "Sanity-check that the run dir has every artifact the bundle "
            "needs; do not write anything"
        ),
    )
    publish_parser.add_argument(
        "--contact",
        default=None,
        help="Reviewer contact address embedded in README.md",
    )
    publish_parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON path for the structured bundle summary",
    )

    experiment_matrix_parser = subparsers.add_parser(
        "experiment-matrix",
        help="Run a first-class matrix of benchmark experiments and aggregate paired comparisons",
    )
    experiment_matrix_parser.add_argument(
        "--spec", required=True, help="JSON spec describing the matrix"
    )
    experiment_matrix_parser.add_argument(
        "--output", default=None, help="Optional output root override"
    )
    experiment_matrix_parser.add_argument(
        "--output-json", default=None, help="Optional JSON output path"
    )

    archive_parser = subparsers.add_parser(
        "archive",
        help="Archive completed runs, update latest symlinks, and optionally prune heavy artifacts",
    )
    archive_parser.add_argument(
        "--runs", nargs="+", required=True, help="Run directories to archive"
    )
    archive_parser.add_argument(
        "--archive-root", default=None, help="Directory to store archived tarballs"
    )
    archive_parser.add_argument(
        "--prune-workspaces", action="store_true", help="Delete workspaces after archiving"
    )
    archive_parser.add_argument(
        "--prune-runtime", action="store_true", help="Delete .runtime directories after archiving"
    )
    archive_parser.add_argument(
        "--compress-logs",
        action="store_true",
        help="Gzip .log files inside the run after archiving",
    )
    archive_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be archived/pruned"
    )
    archive_parser.add_argument(
        "--force", action="store_true", help="Archive even if the run appears active"
    )
    archive_parser.add_argument("--output-json", default=None, help="Optional JSON output path")

    commit0_parser = subparsers.add_parser(
        "commit0-benchmark",
        help="Run APEX on the real Commit0 benchmark used by the CAID paper",
    )
    commit0_parser.add_argument("--config", default=None, help="Optional JSON config path")
    commit0_parser.add_argument(
        "--output",
        default=None,
        help="Directory for benchmark outputs. Defaults to .apex_commit0_<backend>_<model>.",
    )
    commit0_parser.add_argument("--rollouts", type=int, default=None, help="Override rollout count")
    commit0_parser.add_argument("--model", default=None, help="Override the primary model")
    commit0_parser.add_argument(
        "--split",
        default="lite",
        help="Commit0 split to use: lite, all, or a single repo name",
    )
    commit0_parser.add_argument(
        "--repos",
        nargs="*",
        default=None,
        help="Optional explicit subset of repo names",
    )
    commit0_parser.add_argument(
        "--limit", type=int, default=None, help="Maximum number of repos to run"
    )
    commit0_parser.add_argument(
        "--dataset-name",
        default="wentingzhao/commit0_combined",
        help="HuggingFace dataset name",
    )
    commit0_parser.add_argument(
        "--dataset-split",
        default="test",
        help="HuggingFace dataset split",
    )
    commit0_parser.add_argument(
        "--dataset-revision",
        default=None,
        # Commit0 datasets are mutable on HuggingFace; revision pinning keeps
        # the evaluated repo universe reproducible.
        help="Optional HuggingFace dataset revision/commit to pin",
    )
    commit0_parser.add_argument(
        "--dataset-fallback-revision",
        action="append",
        default=[],
        # Commit0 dataset revisions can have different row universes; fallback
        # revisions recover explicitly requested repos missing from the primary.
        help="Optional HuggingFace dataset revision to consult for repos missing from primary",
    )
    commit0_parser.add_argument(
        "--task-parallelism",
        type=int,
        default=None,
        help="Optional number of benchmark repo tasks to run concurrently",
    )
    commit0_parser.add_argument(
        "--single-model",
        action="store_true",
        help=(
            "Slice config.llm_configs to the primary entry only. Required for "
            "apples-to-apples CAID head-to-head when the config carries a "
            "model portfolio. Combine with --model X to lock to a specific "
            "model class."
        ),
    )
    commit0_parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help=(
            "Run the benchmark N independent times into output/seed_0/, "
            "output/seed_1/, ... and emit a per-repo Wilson 95%% CI aggregate "
            "summary at the parent. Default 1 = single-run, no aggregation."
        ),
    )
    _add_agent_mode_argument(
        commit0_parser,
        default=_default_agent_mode_for_subcommand("commit0-benchmark"),
    )
    commit0_parser.add_argument(
        "--docker-image",
        default=None,
        help="Docker image tag for --agent-mode in_container_v5 runs.",
    )
    _add_benchmark_mode_argument(commit0_parser)

    swebench_pro_parser = subparsers.add_parser(
        "swebench-pro-benchmark",
        help="Run APEX on the SWE-Bench Pro public benchmark with Docker-backed evaluation",
    )
    swebench_pro_parser.add_argument("--config", default=None, help="Optional JSON config path")
    swebench_pro_parser.add_argument(
        "--output",
        default=None,
        help="Directory for benchmark outputs. Defaults to .apex_swebench_pro_<backend>_<model>.",
    )
    swebench_pro_parser.add_argument(
        "--rollouts", type=int, default=None, help="Override rollout count"
    )
    swebench_pro_parser.add_argument("--model", default=None, help="Override the primary model")
    swebench_pro_parser.add_argument(
        "--limit", type=int, default=None, help="Maximum number of tasks to run"
    )
    swebench_pro_parser.add_argument(
        "--instances", nargs="*", default=None, help="Optional explicit subset of instance ids"
    )
    swebench_pro_parser.add_argument(
        "--repos", nargs="*", default=None, help="Optional explicit subset of repo names"
    )
    swebench_pro_parser.add_argument(
        "--languages", nargs="*", default=None, help="Optional subset of repo languages"
    )
    swebench_pro_parser.add_argument(
        "--dataset-name",
        default=SWEBENCH_PRO_DATASET_NAME,
        help="HuggingFace dataset name",
    )
    swebench_pro_parser.add_argument(
        "--dataset-split",
        default=SWEBENCH_PRO_DATASET_SPLIT,
        help="HuggingFace dataset split",
    )
    swebench_pro_parser.add_argument(
        "--dockerhub-username",
        default="jefzda",
        help="Docker Hub username hosting the sweap-images repository",
    )
    swebench_pro_parser.add_argument(
        "--docker-platform",
        default=None,
        help="Docker platform override, e.g. linux/amd64",
    )
    swebench_pro_parser.add_argument(
        "--block-network",
        action="store_true",
        help="Disable network access inside evaluation containers",
    )
    swebench_pro_parser.add_argument(
        "--scripts-cache-dir",
        default=None,
        help="Optional directory to cache official run scripts and parsers",
    )
    swebench_pro_parser.add_argument(
        "--agent-visibility-mode",
        choices=[
            SWEBENCH_AGENT_VISIBILITY_PUBLISHED_PARITY,
            SWEBENCH_AGENT_VISIBILITY_ONLINE_FAIR,
            SWEBENCH_AGENT_VISIBILITY_BENCHMARK_AWARE,
        ],
        default=SWEBENCH_AGENT_VISIBILITY_PUBLISHED_PARITY,
        help=(
            "How much benchmark-only information APEX agents see. "
            "'published_parity' matches public repo+issue inference, while "
            "'benchmark_aware' exposes benchmark-only evaluator details. "
            "'online_fair' is kept as a compatibility alias."
        ),
    )
    swebench_pro_parser.add_argument(
        "--rollout-selection-policy",
        choices=[
            SWEBENCH_ROLLOUT_SELECTION_ORCHESTRATOR,
            SWEBENCH_ROLLOUT_SELECTION_OFFICIAL_EVALUATOR,
        ],
        default=SWEBENCH_ROLLOUT_SELECTION_ORCHESTRATOR,
        help=(
            "How final rollout selection is done. 'orchestrator' uses APEX's own "
            "selection pipeline and reserves the official evaluator for baseline/final scoring."
        ),
    )
    _add_agent_mode_argument(
        swebench_pro_parser,
        default=_default_agent_mode_for_subcommand("swebench-pro-benchmark"),
    )
    _add_benchmark_mode_argument(swebench_pro_parser)

    # ---- SWE-EVO benchmark (V5 in-container agent harness) ----
    swe_evo_parser = subparsers.add_parser(
        "swe-evo-benchmark",
        help=(
            "Run APEX on the SWE-EVO benchmark using the V5 in-container "
            "agent loop (apex.orchestrator_in_container_agent)."
        ),
    )
    swe_evo_parser.add_argument(
        "--output",
        default=None,
        help="Directory for benchmark outputs (default: .apex_swe_evo).",
    )
    swe_evo_parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON config path (used to derive llm_config).",
    )
    swe_evo_parser.add_argument(
        "--model-name",
        default="apex-swe-evo",
        help="Model name to record into preds.json.",
    )
    swe_evo_parser.add_argument(
        "--arrow-path",
        default=None,
        help="Optional path to the SWE-EVO arrow dataset file.",
    )
    swe_evo_parser.add_argument(
        "--jsonl-path",
        default=None,
        help="Optional path to a SWE-EVO JSONL file.",
    )
    swe_evo_parser.add_argument(
        "--instances",
        nargs="*",
        default=None,
        help="Optional explicit subset of instance ids.",
    )
    swe_evo_parser.add_argument(
        "--repos",
        nargs="*",
        default=None,
        help="Optional explicit subset of repo names.",
    )
    swe_evo_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of tasks to evaluate.",
    )
    swe_evo_parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Override max_turns for the in-container agent loop.",
    )
    swe_evo_parser.add_argument(
        "--per-tool-timeout-seconds",
        type=int,
        default=None,
        help="Per-tool wall clock budget in seconds.",
    )
    swe_evo_parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Skip git-cloning each task's repo (useful for offline / docker flows).",
    )
    swe_evo_parser.add_argument(
        "--no-intermediate-commits-in-prompt",
        action="store_true",
        help="Do not surface intermediate PR evidence to the agent.",
    )
    _add_agent_mode_argument(
        swe_evo_parser, default=_default_agent_mode_for_subcommand("swe-evo-benchmark")
    )
    _add_benchmark_mode_argument(swe_evo_parser)

    # ---- The three first-class APEX usage modes ----
    testgen_for_fix_parser = subparsers.add_parser(
        "testgen-for-fix",
        help=(
            "Mode 1: generate test cases for a known code fix. "
            "Caller supplies repo + problem statement + gold patch; "
            "APEX writes tests that should F2P on the patch. Real-world "
            "use: regression-suite augmentation, code review."
        ),
    )
    testgen_for_fix_parser.add_argument("--repo-path", required=True)
    testgen_for_fix_parser.add_argument("--problem-statement", required=True)
    testgen_for_fix_parser.add_argument(
        "--patch-file",
        required=True,
        help="Path to a unified-diff file containing the gold fix.",
    )
    testgen_for_fix_parser.add_argument("--output-dir", required=True)
    testgen_for_fix_parser.add_argument("--output-json", default=None)
    testgen_for_fix_parser.add_argument("--language", default="python")
    testgen_for_fix_parser.add_argument("--install-repo", action="store_true")
    _add_agent_mode_argument(
        testgen_for_fix_parser,
        default=_default_agent_mode_for_subcommand("testgen-for-fix"),
    )
    _add_benchmark_mode_argument(testgen_for_fix_parser)
    testgen_for_fix_parser.add_argument(
        "--test-file",
        action="append",
        default=None,
        dest="prewritten_test_files",
        help=(
            "Optional pre-written test file(s) to use as the agent's "
            "output (skips actual LLM generation). Repeat for multiple. "
            "If omitted, the default no-op generator returns no artifacts "
            "and the result reports 'no artifacts produced'."
        ),
    )

    codegen_for_tests_parser = subparsers.add_parser(
        "codegen-for-tests",
        help=(
            "Mode 2: generate the code change that makes a given test "
            "suite pass. Caller supplies repo + problem statement + "
            "test artifacts; APEX writes the patch. Classic TDD."
        ),
    )
    codegen_for_tests_parser.add_argument("--repo-path", required=True)
    codegen_for_tests_parser.add_argument("--problem-statement", required=True)
    codegen_for_tests_parser.add_argument(
        "--test-file",
        action="append",
        required=True,
        dest="test_files",
        help="Path to a gold test file. Repeat for multiple.",
    )
    codegen_for_tests_parser.add_argument("--output-dir", required=True)
    codegen_for_tests_parser.add_argument("--output-json", default=None)
    codegen_for_tests_parser.add_argument("--language", default="python")
    codegen_for_tests_parser.add_argument("--install-repo", action="store_true")
    _add_agent_mode_argument(
        codegen_for_tests_parser,
        default=_default_agent_mode_for_subcommand("codegen-for-tests"),
    )
    _add_hierarchical_v5_arguments(codegen_for_tests_parser)
    _add_abstention_threshold_argument(codegen_for_tests_parser)
    _add_benchmark_mode_argument(codegen_for_tests_parser)
    codegen_for_tests_parser.add_argument(
        "--docker-image",
        default=None,
        help="Docker image tag for --agent-mode in_container_v5 runs.",
    )
    codegen_for_tests_parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Override max_turns for --agent-mode in_container_v5.",
    )
    codegen_for_tests_parser.add_argument(
        "--patch-file",
        default=None,
        help=(
            "Optional pre-written patch file to use as the agent's "
            "output (skips actual LLM generation). If omitted, the "
            "default no-op generator returns None and the result reports "
            "'no patch produced'."
        ),
    )

    generate_both_parser = subparsers.add_parser(
        "generate-both",
        help=(
            "Mode 3: chained testgen → codegen from a problem statement "
            "only. Caller supplies just repo + problem statement; APEX "
            "produces both tests and code."
        ),
    )
    generate_both_parser.add_argument("--repo-path", required=True)
    generate_both_parser.add_argument("--problem-statement", required=True)
    generate_both_parser.add_argument("--output-dir", required=True)
    generate_both_parser.add_argument("--output-json", default=None)
    generate_both_parser.add_argument("--language", default="python")
    generate_both_parser.add_argument("--install-repo", action="store_true")
    _add_agent_mode_argument(
        generate_both_parser,
        default=_default_agent_mode_for_subcommand("generate-both"),
    )
    _add_benchmark_mode_argument(generate_both_parser)
    generate_both_parser.add_argument(
        "--test-file",
        action="append",
        default=None,
        dest="prewritten_test_files",
        help=(
            "Optional pre-written test files to use as the testgen "
            "phase output (Phase A); skips actual LLM generation."
        ),
    )
    generate_both_parser.add_argument(
        "--patch-file",
        default=None,
        help=(
            "Optional pre-written patch to use as the codegen phase "
            "output (Phase B); skips actual LLM generation."
        ),
    )

    tdd_evaluate_parser = subparsers.add_parser(
        "tdd-evaluate",
        help=(
            "Evaluate a test suite against two pre-prepared sandboxes "
            "(broken vs fixed). Real-world / TDD entry point — no benchmark "
            "task object required. Reports F2P transitions, mutation "
            "score, and minimization recommendations as JSON."
        ),
    )
    tdd_evaluate_parser.add_argument(
        "--broken-dir",
        required=True,
        help="Path to the sandbox in its current (pre-fix) state",
    )
    tdd_evaluate_parser.add_argument(
        "--fixed-dir",
        required=True,
        help="Path to the sandbox after the fix candidate is applied",
    )
    tdd_evaluate_parser.add_argument(
        "--test-file",
        action="append",
        required=True,
        dest="test_files",
        help=(
            "Path to a test file the agent generated. May be passed "
            "multiple times for a portfolio of tests."
        ),
    )
    tdd_evaluate_parser.add_argument(
        "--language",
        default="python",
        help="Repo language (default python). Passes through to the F2P oracle.",
    )
    tdd_evaluate_parser.add_argument(
        "--install-repo",
        action="store_true",
        help=(
            "Run the per-language environment install (pip / npm ci / cargo "
            "fetch / etc.) in each sandbox before running tests. Required for "
            "non-trivial repos that the test files import from."
        ),
    )
    tdd_evaluate_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="Per-side test wall-clock budget (default 300s).",
    )
    tdd_evaluate_parser.add_argument(
        "--enable-mutation",
        action="store_true",
        help=(
            "Also run mutation discrimination against the fixed sandbox if "
            "F2P confirms the suite catches the bug."
        ),
    )
    tdd_evaluate_parser.add_argument(
        "--mutation-target",
        action="append",
        default=None,
        dest="mutation_targets",
        help=(
            "Path of a source file to mutate (relative to fixed-dir). Repeat "
            "for multiple files. Required when --enable-mutation is set."
        ),
    )
    tdd_evaluate_parser.add_argument(
        "--enable-minimization",
        action="store_true",
        help="Greedy set-cover over F2P + mutation kills; report dropped vs kept files.",
    )
    tdd_evaluate_parser.add_argument(
        "--output-json",
        default=None,
        help="Path to write the structured report (default: stdout).",
    )
    tdd_evaluate_parser.add_argument(
        "--output-dir",
        default=None,
        help="Working directory for sandbox artifacts (default: a tempdir).",
    )

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help=(
            "Compute Brier score, expected calibration error, and reliability tables "
            "for APEX selection scores by walking past run directories"
        ),
    )
    calibrate_parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="One or more directories containing apex_result.json files (walked recursively)",
    )
    calibrate_parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of equal-width reliability bins (default: 10)",
    )
    calibrate_parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the structured calibration report",
    )
    calibrate_parser.add_argument(
        "--output-markdown",
        default=None,
        help="Optional path to write a Markdown reliability table",
    )

    args = parser.parse_args()

    if args.command == "solve":
        _cmd_solve(args)
        return
    if args.command == "analyze":
        _cmd_analyze(args)
        return
    if args.command == "benchmark":
        _cmd_benchmark(args)
        return
    if args.command == "benchmark-compare":
        _cmd_benchmark_compare(args)
        return
    if args.command == "doctor":
        _cmd_doctor(args)
        return
    if args.command == "doctor-scan":
        _cmd_doctor_scan(args)
        return
    if args.command == "doctor-clean":
        _cmd_doctor_clean(args)
        return
    if args.command == "doctor-verify-manifest":
        _cmd_doctor_verify_manifest(args)
        return
    if args.command == "status":
        _cmd_status(args)
        return
    if args.command == "resume":
        _cmd_resume(args)
        return
    if args.command == "retry":
        _cmd_retry(args)
        return
    if args.command == "watch":
        _cmd_watch(args)
        return
    if args.command == "cleanup":
        _cmd_cleanup(args)
        return
    if args.command == "run-compare":
        _cmd_run_compare(args)
        return
    if args.command == "replay":
        _cmd_replay(args)
        return
    if args.command == "replay-deterministic":
        _cmd_replay_deterministic(args)
        return
    if args.command == "publish-benchmark":
        _cmd_publish_benchmark(args)
        return
    if args.command == "experiment-matrix":
        _cmd_experiment_matrix(args)
        return
    if args.command == "archive":
        _cmd_archive(args)
        return
    if args.command == "commit0-benchmark":
        _cmd_commit0_benchmark(args)
        return
    if args.command == "swebench-pro-benchmark":
        _cmd_swebench_pro_benchmark(args)
        return
    if args.command == "swe-evo-benchmark":
        _cmd_swe_evo_benchmark(args)
        return
    if args.command == "calibrate":
        _cmd_calibrate(args)
        return
    if args.command == "tdd-evaluate":
        _cmd_tdd_evaluate(args)
        return
    if args.command == "testgen-for-fix":
        _cmd_testgen_for_fix(args)
        return
    if args.command == "codegen-for-tests":
        _cmd_codegen_for_tests(args)
        return
    if args.command == "generate-both":
        _cmd_generate_both(args)
        return

    parser.print_help()
    sys.exit(1)


# The agent-surface vocabulary has ONE definition (apex.core.config). Aliased
# here so existing references keep working; never re-define the list locally.
_AGENT_MODE_CHOICES = AGENT_MODE_CHOICES

# Phase A.1 (Decisive-Edge): per-benchmark default agent surface.
# CLI subcommands that benefit from V5 in-container agent dispatch
# (Commit0, SWE-Bench Pro, SWT-Bench) opt in here. Remaining
# subcommands (solve, testgen-for-fix, codegen-for-tests,
# generate-both, swe-evo-benchmark) keep the legacy ``scaffolded``
# default so non-benchmark callers and Mode 1/3 stay on MASAI.
_PER_SUBCOMMAND_DEFAULT_AGENT_MODE: dict[str, str] = {
    "commit0-benchmark": "in_container_v5",
    "swebench-pro-benchmark": "in_container_v5",
    "swt-bench-benchmark": "in_container_v5",
}


def _default_agent_mode_for_subcommand(subcommand: str) -> str:
    """Return the per-subcommand default ``--agent-mode`` value.

    Falls back to ``"scaffolded"`` for unknown / non-benchmark
    subcommands so behaviour is unchanged for the legacy paths.
    """
    return _PER_SUBCOMMAND_DEFAULT_AGENT_MODE.get(subcommand, "scaffolded")


def _add_agent_mode_argument(
    parser: argparse.ArgumentParser,
    *,
    default: Optional[str] = None,
) -> None:
    """Attach a uniform ``--agent-mode`` flag to *parser*.

    Four values are valid:

      * ``scaffolded`` (default for non-benchmark subcommands,
        back-compat): the legacy MASAI
        Reproducer/Localizer/Patcher path through ApexOrchestrator.
      * ``cli_agent``: existing CLI-backend rollouts (codex / claude /
        gemini / opencode CLIs which are themselves agent loops).
      * ``in_container_v5``: the V5 in-container agent loop. Routes
        through ``apex.modes`` and (when ``--docker-image`` is set)
        through ``ContainerSupervisor`` for true container isolation.
        Phase A.1: this is the new default for ``commit0-benchmark``,
        ``swebench-pro-benchmark``, and ``swt-bench-benchmark`` — the
        V5 surface dominates patch-generation benchmarks.
      * ``hierarchical_v5`` (Phase 6.5): the planner-above-V5 agent.
        Decomposes the problem into sub-tasks and rebalances the V5
        turn budget across them. ``--total-budget N`` sets the global
        cap (defaults to 8 × 3 = 24); ``--n-subtasks K`` sets the
        decomposition target (default 3).
    """
    effective_default = default if default in _AGENT_MODE_CHOICES else "scaffolded"
    parser.add_argument(
        "--agent-mode",
        default=effective_default,
        choices=list(_AGENT_MODE_CHOICES),
        help=(
            "Which agent surface to use. 'scaffolded' is the legacy MASAI "
            "orchestrator; 'cli_agent' uses CLI-backend rollouts; "
            "'in_container_v5' is the V5 in-container loop (use "
            "--docker-image for true container isolation); "
            "'hierarchical_v5' is the Phase 6.5 planner-above-V5 with "
            "per-subtask budget management (use --total-budget / "
            "--n-subtasks to tune). "
            f"Default for this subcommand: {effective_default!r}."
        ),
    )


# Phase A.3 (Decisive-Edge): preset config files shipped with APEX.
# These overlay onto ApexConfig BEFORE explicit --key value overrides.
_BENCHMARK_MODE_OFF = "off"
_BENCHMARK_MODE_PUBLICATION = "publication"
_BENCHMARK_MODE_HEADLINE = "headline"
_BENCHMARK_MODE_CHOICES = (
    _BENCHMARK_MODE_OFF,
    _BENCHMARK_MODE_PUBLICATION,
    _BENCHMARK_MODE_HEADLINE,
)
_BENCHMARK_MODE_PRESET_FILES: dict[str, str] = {
    _BENCHMARK_MODE_PUBLICATION: "publication_mode.json",
    _BENCHMARK_MODE_HEADLINE: "benchmark_mode.json",
}


def _add_benchmark_mode_argument(parser: argparse.ArgumentParser) -> None:
    """Attach the ``--benchmark-mode {publication,headline,off}`` flag.

    Phase A.3: presets that overlay strategic-flip choices onto the
    config BEFORE explicit ``--rollouts`` / ``--model`` overrides.
    Tradeoff documented in :mod:`apex.configs.publication_mode` and
    :mod:`apex.configs.benchmark_mode`:

      * ``publication`` — defensible numbers. All four strategic flips
        ON: official upstream audit, parallel fairness audit, strict
        (env-skip-counts-as-zero) headline metric, no rollout salvage,
        calibrated abstention threshold (0.50), controller-models
        library enabled. Use for academic publication.
      * ``headline`` — leaderboard-optimised. Strategic flips relaxed:
        APEX-private scoring only, no fairness audit overhead,
        runnable-denominator headline (env skips excluded), salvage
        candidates surface as success, looser abstention (0.30),
        controller-models library disabled. Use for leaderboard runs.
      * ``off`` (default) — no preset applied; explicit overrides only.

    All evaluation reports always emit BOTH ``score_strict`` and
    ``score_runnable`` regardless of the headline; the preset only
    controls which one is rendered as the leaderboard number in the
    markdown summary.
    """
    parser.add_argument(
        "--benchmark-mode",
        default=_BENCHMARK_MODE_OFF,
        choices=list(_BENCHMARK_MODE_CHOICES),
        help=(
            "Apply a strategic-flip preset before explicit overrides. "
            "'publication' optimises for defensible numbers (official "
            "audit + parallel fairness scoring + strict headline + no "
            "salvage + calibrated abstention 0.50 + controller-models "
            "library on). 'headline' optimises for leaderboard rank "
            "(APEX-private scoring + runnable-denominator headline + "
            "salvage on + abstention 0.30 + controller-models library "
            "off). 'off' (default) applies no preset. "
            "Reports always emit BOTH score_strict and score_runnable."
        ),
    )


def _resolve_benchmark_mode_preset_path(mode: str) -> Optional[Path]:
    filename = _BENCHMARK_MODE_PRESET_FILES.get(mode)
    if not filename:
        return None
    candidate = Path(__file__).resolve().parent / "configs" / filename
    return candidate if candidate.exists() else None


def _apply_benchmark_mode_preset(
    config: "ApexConfig",
    args: argparse.Namespace,
) -> "ApexConfig":
    """Phase A.3: overlay the ``--benchmark-mode`` preset onto ``config``.

    The preset is a JSON file with the same shape as a normal APEX
    config. We walk the dict and assign matching dataclass attributes;
    unknown keys are ignored with a logged warning so future preset
    keys can be added without breaking older binaries.

    The preset is applied BEFORE explicit per-flag overrides so that
    callers can still surgically override a single value with e.g.
    ``--abstention-threshold 0.40`` while the rest of the preset
    remains in effect.
    """
    mode = str(getattr(args, "benchmark_mode", _BENCHMARK_MODE_OFF) or _BENCHMARK_MODE_OFF)
    if mode == _BENCHMARK_MODE_OFF:
        return config
    preset_path = _resolve_benchmark_mode_preset_path(mode)
    if preset_path is None:
        print(
            f"Warning: --benchmark-mode={mode!r} preset file not found; "
            "continuing without preset overlay.",
            file=sys.stderr,
        )
        return config
    try:
        payload = json.loads(preset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"Warning: failed to load --benchmark-mode preset {preset_path}: {exc}",
            file=sys.stderr,
        )
        return config
    if not isinstance(payload, dict):
        return config
    for section_name, section_payload in payload.items():
        if section_name.startswith("_"):
            # Comment / metadata fields. Ignore.
            continue
        if not isinstance(section_payload, dict):
            continue
        section = getattr(config, section_name, None)
        if section is None:
            continue
        for key, value in section_payload.items():
            if not hasattr(section, key):
                continue
            try:
                setattr(section, key, value)
            except Exception:
                # Defensive: a coerced enum field may reject a raw
                # string. The preset stays best-effort; we keep going.
                continue
    return config


def _add_hierarchical_v5_arguments(parser: argparse.ArgumentParser) -> None:
    """Phase 6.5: tuning knobs for ``--agent-mode hierarchical_v5``."""
    parser.add_argument(
        "--total-budget",
        type=int,
        default=None,
        help=(
            "Total turn budget across all sub-tasks (--agent-mode "
            "hierarchical_v5 only). Defaults to 8 × n_subtasks (so 24 "
            "with the default --n-subtasks=3)."
        ),
    )
    parser.add_argument(
        "--n-subtasks",
        type=int,
        default=None,
        help=("Sub-task decomposition target (--agent-mode hierarchical_v5 only). Default 3."),
    )
    parser.add_argument(
        "--rebalance-strategy",
        default="feedback",
        choices=("feedback", "static"),
        help=(
            "Budget rebalancing strategy (--agent-mode hierarchical_v5 "
            "only). 'feedback' (default) reallocates after each sub-task "
            "based on actuals; 'static' keeps the initial split."
        ),
    )


def _add_abstention_threshold_argument(parser: argparse.ArgumentParser) -> None:
    """Phase 6.3: configure the calibrated abstention threshold."""
    parser.add_argument(
        "--abstention-threshold",
        type=float,
        default=None,
        help=(
            "Calibrated abstention threshold in [0,1]. When the calibrated "
            "ConfidenceScorer aggregate is below this threshold AND "
            "rollout.allow_salvage is False, the orchestrator downgrades "
            "the result from SOLVED to ABSTAINED. Default: 0.50 "
            "(literature-informed). Pass 0.0 to fully disable the override."
        ),
    )


def _resolve_agent_mode(config: ApexConfig, args: argparse.Namespace) -> str:
    """Resolve the orchestration agent surface to ONE value and record it.

    This is the single place CLI runs decide which agent surface the
    orchestration uses. Precedence (highest first), each surfaced with its
    provenance so a developer / researcher / agent can always see — and trust —
    which path a run took:

      1. explicit ``--agent-mode`` on the command line
      2. ``benchmark.default_agent_mode`` set in the loaded config file
      3. the per-subcommand default (``_default_agent_mode_for_subcommand``)
      4. ``GLOBAL_DEFAULT_AGENT_MODE``

    The resolved value is written back to ``config.benchmark.default_agent_mode``
    (the single field every consumer reads) and printed loudly. An explicit
    ``--agent-mode`` that overrides a config-file value is called out so the
    override is never silent — eliminating the class of bug where a canonical
    command silently runs a different surface than the documented default.
    """
    subcommand = str(getattr(args, "command", "") or "").strip()
    per_subcommand = _default_agent_mode_for_subcommand(subcommand)
    cli_value = str(getattr(args, "agent_mode", "") or "").strip()
    file_value = str(getattr(config.benchmark, "default_agent_mode", "") or "").strip()
    # argparse injects the per-subcommand default into args.agent_mode, so a value
    # equal to that default is indistinguishable from "not passed" (and yields the
    # same surface either way). A value that DIFFERS is an explicit user choice.
    explicit = cli_value in AGENT_MODE_CHOICES and cli_value != per_subcommand

    if explicit:
        mode, source = cli_value, "explicit --agent-mode"
        if file_value in AGENT_MODE_CHOICES and file_value != mode:
            print(
                f"[agent-mode] NOTE: explicit --agent-mode={mode!r} overrides the "
                f"config-file value {file_value!r}."
            )
    elif file_value in AGENT_MODE_CHOICES:
        mode, source = file_value, "config file (benchmark.default_agent_mode)"
    elif cli_value in AGENT_MODE_CHOICES:
        mode, source = cli_value, f"per-subcommand default for {subcommand or 'this command'!r}"
    else:
        mode, source = GLOBAL_DEFAULT_AGENT_MODE, "global fallback"

    try:
        config.benchmark.default_agent_mode = mode
    except AttributeError:
        pass
    print(f"[agent-mode] orchestration surface = {mode!r}  (source: {source})")
    return mode


def _apply_phase6_overrides(
    config: ApexConfig,
    args: argparse.Namespace,
) -> ApexConfig:
    """Phase 6.3 / 6.5 + Phase A.1 / A.3: lift CLI flags onto the ApexConfig.

    Order of operations matters:

      1. ``--benchmark-mode`` preset overlay (Phase A.3) — applied
         first so explicit per-flag overrides win.
      2. ``--agent-mode`` lift onto ``benchmark.default_agent_mode``
         (Phase A.1) — explicit ``--agent-mode`` always wins over the
         preset.
      3. ``--abstention-threshold`` (Phase 6.3) — explicit threshold
         overrides preset.
    """
    # 1. Preset first, so explicit overrides win.
    config = _apply_benchmark_mode_preset(config, args)

    # 2. Resolve the orchestration agent surface to a single, logged value.
    _resolve_agent_mode(config, args)

    # 3. Explicit threshold override.
    threshold = getattr(args, "abstention_threshold", None)
    if threshold is not None:
        try:
            config.orchestration.abstention_threshold = float(threshold)
        except (TypeError, ValueError):
            pass
    config = apply_localizer_enforcement_override(config)
    return config


def _load_config(config_path: str | None) -> ApexConfig:
    if config_path:
        return ApexConfig.from_file(config_path)
    # No explicit JSON config — return a default ApexConfig but route the
    # primary llm_config through whatever CLI agent is installed on PATH
    # rather than the legacy OPENAI_API default that requires a shell-
    # level API key. APEX is fully CLI-backed: project memory note "CLI
    # backends are agents not LLMs" + "Optimize for SOTA results, never
    # for cost" mean we should always pick the strongest installed CLI.
    config = ApexConfig()
    config.llm_configs = [_default_cli_llm_config()]
    return config


def _default_cli_llm_config() -> LLMConfig:
    """Build the default ``LLMConfig`` by detecting the strongest installed CLI.

    Falls back to a no-op codex CLI configuration when no CLI is on PATH
    so callers still see a usable ApexConfig; the missing-CLI condition
    will surface later via ``cli_backend_unavailable_reason`` (and is
    reported by ``apex doctor``). We deliberately do NOT raise at import
    time because legacy tests instantiate ``ApexConfig`` for serialization
    snapshots without ever invoking the backend.
    """

    from .core.cli_backend import (
        NoCLIBackendAvailable,
        detect_default_cli_backend,
    )

    try:
        identifier = detect_default_cli_backend()
    except NoCLIBackendAvailable:
        # Fall back to codex_cli:gpt-5.5 — keeps the orchestrator's
        # routing table happy; the backend availability probe will mark
        # it unhealthy and the doctor / smoke sweep will flag the
        # missing CLI explicitly.
        identifier = "codex_cli:gpt-5.5"
    backend_token, _, model = identifier.partition(":")
    backend = LLMBackend(backend_token)
    return LLMConfig(
        backend=backend,
        model=model or None,
        cli_timeout=1200,
        cli_disable_osx_sandbox=True,
        cli_permission_mode=None,
    )


def _apply_common_overrides(
    config: ApexConfig, rollouts: int | None, model: str | None, output: str | None
) -> ApexConfig:
    if rollouts is not None:
        bounded_rollouts = max(1, int(rollouts))
        config.rollout.num_rollouts = bounded_rollouts
        config.rollout.min_rollouts = bounded_rollouts
        config.rollout.max_rollouts = bounded_rollouts
        if config.rollout.rollout_buckets:
            config.rollout.rollout_buckets = [bounded_rollouts]
    if model:
        normalized_model = normalize_supported_model_name(model)
        matching = next(
            (candidate for candidate in config.llm_configs if candidate.model == normalized_model),
            None,
        )
        if matching is None:
            if normalized_model == "opus":
                backend = LLMBackend.CLAUDE_CLI
            elif normalized_model == "gemini-3.1-pro":
                backend = LLMBackend.GEMINI_CLI
            elif normalized_model == "meta/avocado-tester":
                backend = LLMBackend.OPENCODE_CLI
            elif normalized_model == "meta/avocado-code-latest":
                backend = LLMBackend.METACODE_CLI
            else:
                backend = LLMBackend.CODEX_CLI
            matching = LLMConfig(
                backend=backend,
                model=normalized_model,
                cli_permission_mode=None,
            )
        config.llm_configs = [
            LLMConfig(
                backend=matching.backend,
                model=normalized_model,
                api_key_env=matching.api_key_env,
                base_url=matching.base_url,
                temperature=matching.temperature,
                max_tokens=matching.max_tokens,
                timeout=matching.timeout,
                cli_command=matching.cli_command,
                cli_args=list(matching.cli_args),
                cli_timeout=matching.cli_timeout,
                cli_hard_timeout_seconds=matching.cli_hard_timeout_seconds,
                cli_strict_hard_timeout=matching.cli_strict_hard_timeout,
                cli_disable_osx_sandbox=matching.cli_disable_osx_sandbox,
                cli_permission_mode=matching.cli_permission_mode,
                cli_env_overrides=dict(matching.cli_env_overrides),
                cli_env_redaction_disabled=matching.cli_env_redaction_disabled,
            )
        ]
        if config.rollout.llm_profiles:
            config.rollout.llm_profiles = []
        if config.rollout.scaffold_stage_llm_indices:
            config.rollout.scaffold_stage_llm_indices = {
                stage_name: 0 for stage_name in config.rollout.scaffold_stage_llm_indices
            }
        if config.planning.planner_llm_index is not None:
            config.planning.planner_llm_index = 0
        if config.planning.planner_model:
            config.planning.planner_model = normalized_model
    if output:
        config.output_dir = output
    return config


def _cmd_solve(args: argparse.Namespace) -> None:
    config = _apply_common_overrides(
        _load_config(args.config), args.rollouts, args.model, args.output
    )
    config = _apply_phase6_overrides(config, args)
    issue_path = Path(args.issue)
    issue_description = issue_path.read_text() if issue_path.exists() else args.issue

    # Read the RESOLVED surface (set by _apply_phase6_overrides above), not the
    # raw arg, so config-file values and provenance logging are honoured.
    agent_mode = str(getattr(config.benchmark, "default_agent_mode", "") or "") or GLOBAL_DEFAULT_AGENT_MODE
    if agent_mode == "in_container_v5":
        _cmd_solve_in_container_v5(args, config, issue_description)
        return
    if agent_mode == "hierarchical_v5":
        _cmd_solve_hierarchical_v5(args, config, issue_description)
        return

    # ``scaffolded`` and ``cli_agent`` both currently route through the
    # legacy ApexOrchestrator (``cli_agent`` is selected at the rollout
    # level via the configured CLI backend; the ``--agent-mode cli_agent``
    # flag is an explicit label so callers can audit/lock that choice).
    result = ApexOrchestrator(config).solve(
        repo_path=args.repo,
        issue_description=issue_description,
        test_command=args.test_command,
    )

    if not result.success:
        print("APEX failed to produce a patch.")
        print(result.explanation or "No explanation available.")
        sys.exit(1)

    print(f"Selected rollout: {result.selected_rollout_id}")
    print(f"Changed files: {', '.join(result.selected_changed_files) or 'n/a'}")
    print(f"Tokens used: {result.total_tokens}")
    print(f"Duration: {result.total_duration_seconds:.1f}s")
    print(result.explanation or "")
    print()
    print(result.patch or "")


def _cmd_solve_in_container_v5(
    args: argparse.Namespace,
    config: ApexConfig,
    issue_description: str,
) -> None:
    """Wire ``apex solve --agent-mode in_container_v5`` to the modes API.

    Bridges through ``apex.modes._invoke_in_container_v5_agent`` (with an
    empty test set on the codegen path) so the V5 agent can be exercised
    end-to-end without a full benchmark harness. Successful patches are
    printed to stdout for compatibility with the rest of the solve flow.
    """
    from .modes import _invoke_in_container_v5_agent

    repo_path = Path(args.repo).expanduser().resolve()
    if not repo_path.exists():
        print(f"Error: --repo {repo_path} does not exist.")
        sys.exit(1)

    docker_image = getattr(args, "docker_image", None)
    outcome = _invoke_in_container_v5_agent(
        broken_dir=repo_path,
        problem_statement=issue_description,
        docker_image=docker_image,
        llm_config=config.llm_configs[0] if config.llm_configs else None,
    )
    if outcome.error:
        print(f"in_container_v5 agent failed: {outcome.error}")
        sys.exit(1)
    if outcome.patch is None:
        marker = outcome.diagnostics.get("status_marker", "ABSTAINED")
        reason = outcome.diagnostics.get("terminated_reason", "no_patch")
        print(f"in_container_v5 agent {marker} ({reason}); no patch emitted.")
        sys.exit(2)
    print(f"in_container_v5 agent submitted patch ({len(outcome.patch)} chars)")
    print()
    print(outcome.patch)


def _cmd_solve_hierarchical_v5(
    args: argparse.Namespace,
    config: ApexConfig,
    issue_description: str,
) -> None:
    """Phase 6.5: wire ``apex solve --agent-mode hierarchical_v5`` to the modes API.

    Bridges through ``apex.modes._invoke_hierarchical_v5_agent``. Same
    contract as the V5 entrypoint plus the planner-above-V5 flags
    (``--total-budget``, ``--n-subtasks``, ``--rebalance-strategy``).
    """
    from .modes import _invoke_hierarchical_v5_agent

    repo_path = Path(args.repo).expanduser().resolve()
    if not repo_path.exists():
        print(f"Error: --repo {repo_path} does not exist.")
        sys.exit(1)

    docker_image = getattr(args, "docker_image", None)
    outcome = _invoke_hierarchical_v5_agent(
        broken_dir=repo_path,
        problem_statement=issue_description,
        docker_image=docker_image,
        llm_config=config.llm_configs[0] if config.llm_configs else None,
        total_budget=getattr(args, "total_budget", None),
        n_subtasks=getattr(args, "n_subtasks", None),
        rebalance_strategy=getattr(args, "rebalance_strategy", "feedback"),
    )
    if outcome.error:
        print(f"hierarchical_v5 agent failed: {outcome.error}")
        sys.exit(1)
    if outcome.patch is None:
        marker = outcome.diagnostics.get("status_marker", "ABSTAINED")
        reason = outcome.diagnostics.get("terminated_reason", "no_patch")
        print(f"hierarchical_v5 agent {marker} ({reason}); no patch emitted.")
        sys.exit(2)
    budget_view = outcome.diagnostics.get("budget_view") or {}
    print(
        f"hierarchical_v5 agent submitted patch ({len(outcome.patch)} chars); "
        f"turns_used={budget_view.get('turns_used')}/{budget_view.get('total_turns')}"
    )
    print()
    print(outcome.patch)


def _cmd_analyze(args: argparse.Namespace) -> None:
    context = RepoAnalyzer(args.repo).analyze()
    print(f"Repository: {args.repo}")
    print(f"Files analyzed: {len(context.files)}")
    print(f"Symbols found: {sum(len(file_info.symbols) for file_info in context.files)}")
    print()
    print(context.get_repo_map())
    if args.output:
        context.save(args.output)
        print()
        print(f"Saved context to {args.output}")


def _cmd_benchmark(args: argparse.Namespace) -> None:
    config = _apply_common_overrides(_load_config(args.config), args.rollouts, args.model, None)
    output_dir = args.output or str(Path(config.output_dir) / "benchmarks")
    runner = BenchmarkRunner(
        config=config,
        fixtures_dir=args.fixtures,
        output_dir=output_dir,
    )
    runner.config_source = str(Path(args.config).resolve()) if args.config else None
    report = runner.run(task_names=args.tasks)
    print(f"Resolved tasks: {report.resolved_tasks}/{report.total_tasks}")
    print(f"Report: {Path(output_dir) / 'benchmark_report.json'}")


def _cmd_benchmark_compare(args: argparse.Namespace) -> None:
    payload = compare_benchmark_reports(
        args.reports,
        labels=args.labels,
        comparison_name=args.name,
    )
    markdown = render_benchmark_comparison_markdown(payload)

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2))
    else:
        output_json = None

    if args.output_markdown:
        output_markdown = Path(args.output_markdown)
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(markdown)
    else:
        output_markdown = None

    print(f"Benchmark family: {payload['benchmark_family']}")
    print(f"Reference run: {payload['reference_label']}")
    print(f"Matched tasks: {payload['common_task_count']}")
    for comparison in payload["pairwise_comparisons"]:
        print(
            f"{comparison['candidate_label']}: "
            f"score_delta={comparison['average_score_delta_percent']:+.2f}% "
            f"solve_delta={comparison['solve_rate_delta_percent']:+.2f}% "
            f"score_wlt={comparison['score_wins']}/{comparison['score_losses']}/{comparison['score_ties']}"
        )
    if output_json is not None:
        print(f"JSON: {output_json}")
    if output_markdown is not None:
        print(f"Markdown: {output_markdown}")


def _cmd_doctor(args: argparse.Namespace) -> None:
    # ``_load_config(None)`` returns an ApexConfig whose primary llm_config
    # points at whatever CLI agent is installed (codex / claude / gemini /
    # opencode). The doctor runs against that default so the report
    # reflects the host's *actual* CLI rather than the legacy OPENAI_API
    # placeholder that always shows "no API key".
    config = _load_config(args.config)
    payload = doctor_summary(
        config,
        config_source=str(Path(args.config).resolve()) if args.config else None,
        run_smoke_tests=not args.skip_smoke_tests,
        run_cli_health_probes=not bool(getattr(args, "skip_cli_smoke_tests", False)),
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
    sys.stdout.write(render_doctor_report(payload))
    if not payload.get("success"):
        sys.exit(1)


def _cmd_doctor_scan(args: argparse.Namespace) -> None:
    """Phase 5.7: scan for apex-owned docker container / dir leaks."""
    from .scripts.apex_doctor import render_scan
    from .scripts.apex_doctor import scan as _scan

    roots = [Path(r) for r in (args.root or [])] if args.root else None
    report = _scan(age_days=float(args.age), filesystem_roots=roots)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report.to_dict(), indent=2))
    sys.stdout.write(render_scan(report) + "\n")
    if bool(getattr(args, "assert_zero_leaks", False)) and report.leak_count > 0:
        sys.stderr.write(
            f"doctor-scan: leak_count={report.leak_count} > 0; failing per --assert-zero-leaks\n"
        )
        sys.exit(1)


def _cmd_doctor_clean(args: argparse.Namespace) -> None:
    """Phase 5.7: remove leaked apex-owned containers / dirs (dry-run by default)."""
    from .scripts.apex_doctor import clean as _clean
    from .scripts.apex_doctor import render_clean

    roots = [Path(r) for r in (args.root or [])] if args.root else None
    report = _clean(
        age_days=float(args.age),
        filesystem_roots=roots,
        confirm=bool(args.confirm),
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report.to_dict(), indent=2))
    sys.stdout.write(render_clean(report) + "\n")


def _cmd_doctor_verify_manifest(args: argparse.Namespace) -> None:
    """Phase 5.7: re-inspect docker images vs run_manifest.json digests."""
    from .scripts.apex_doctor import (
        render_verify_manifest,
    )
    from .scripts.apex_doctor import (
        verify_manifest as _verify_manifest,
    )

    report = _verify_manifest(Path(args.run))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report.to_dict(), indent=2))
    sys.stdout.write(render_verify_manifest(report) + "\n")
    if report.error or report.drift_detected:
        sys.exit(1)


def _cmd_status(args: argparse.Namespace) -> None:
    payload = inspect_run_directory(args.run)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
    sys.stdout.write(render_status_table(payload))


def _cmd_resume(args: argparse.Namespace) -> None:
    payload = resume_run(args.run, dry_run=bool(args.dry_run), force=bool(args.force))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
    print(f"Action: {payload['action']}")
    print(f"Selected tasks: {len(payload['selected_task_ids'])}")
    if payload.get("no_op"):
        print("No resume work required.")
        return
    if payload.get("dry_run"):
        print("Dry run: yes")
        return
    print(
        f"{payload['primary_metric_name']}: {float(payload.get('primary_metric_percent') or 0.0):.2f}%"
    )
    print(f"Report: {payload.get('report_path')}")


def _cmd_retry(args: argparse.Namespace) -> None:
    payload = retry_run(
        args.run,
        failed_only=bool(args.failed_only),
        suspicious_only=bool(args.suspicious_only),
        task_ids=args.tasks,
        dry_run=bool(args.dry_run),
        force=bool(args.force),
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
    print(f"Action: {payload['action']}")
    print(f"Selected tasks: {len(payload['selected_task_ids'])}")
    print(f"Cleared paths: {len(payload['removed_paths'])}")
    if payload.get("no_op"):
        print("No retry work required.")
        return
    if payload.get("dry_run"):
        print("Dry run: yes")
        return
    print(
        f"{payload['primary_metric_name']}: {float(payload.get('primary_metric_percent') or 0.0):.2f}%"
    )
    print(f"Report: {payload.get('report_path')}")


def _cmd_watch(args: argparse.Namespace) -> None:
    payload = watch_run(
        args.run,
        refresh_seconds=float(args.refresh_seconds),
        iterations=args.iterations,
        no_clear=bool(args.no_clear),
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))


def _cmd_cleanup(args: argparse.Namespace) -> None:
    payload = cleanup_runs(args.runs, dry_run=bool(args.dry_run))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
    print(f"Dry run: {'yes' if payload['dry_run'] else 'no'}")
    print(f"Stale run dirs: {len(payload['stale_run_dirs'])}")
    print(f"Killed processes: {len(payload['killed_processes'])}")
    print(f"Removed directories: {len(payload['removed_directories'])}")


def _cmd_run_compare(args: argparse.Namespace) -> None:
    payload = compare_run_directories(args.left, args.right)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
    sys.stdout.write(render_run_compare(payload))


def _cmd_replay(args: argparse.Namespace) -> None:
    payload = replay_failure(
        args.run,
        task_id=args.task,
        cluster=args.cluster,
        output_dir=args.output,
        dry_run=bool(args.dry_run),
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
    print(f"Action: {payload['action']}")
    print(f"Replay output: {payload['output_dir']}")
    print(f"Selected tasks: {len(payload['selected_task_ids'])}")
    if payload.get("dry_run"):
        print("Dry run: yes")
        return
    print(
        f"{payload['primary_metric_name']}: {float(payload.get('primary_metric_percent') or 0.0):.2f}%"
    )
    print(f"Report: {payload.get('report_path')}")


def _cmd_replay_deterministic(args: argparse.Namespace) -> None:
    """Phase 6.6: deterministic replay of a recorded rollout."""
    from .replay import (
        LiveCallDuringReplayError,
        ReplayDivergenceError,
        ReplayPlayer,
    )

    record_path = Path(args.record_path)
    if not record_path.exists():
        print(f"Error: record path does not exist: {record_path}", file=sys.stderr)
        sys.exit(2)

    mutate: dict[str, str] | None = None
    if args.mutate_turn is not None:
        if args.mutate_prompt is None:
            print(
                "Error: --mutate-turn requires --mutate-prompt",
                file=sys.stderr,
            )
            sys.exit(2)
        mutate = {f"turn_{int(args.mutate_turn)}_prompt": str(args.mutate_prompt)}

    summary: dict[str, Any] = {
        "record_path": str(record_path),
        "mode": "verify" if args.verify else ("mutate" if mutate else "replay"),
    }

    try:
        if args.verify and args.verify_against:
            staged = ReplayPlayer(record_path)
            staged._records = staged._load_records(record_path)
            staged.verify_against(Path(args.verify_against))
            summary["verified"] = True
            summary["records"] = len(staged.records)
        else:
            with ReplayPlayer.replay(
                record_path,
                mutate=mutate,
                strict=bool(args.verify and not mutate),
            ) as player:
                summary["records"] = len(player.records)
                summary["loaded"] = True
                summary["diverged"] = player.diverged
                summary["mutations_applied"] = list(player.mutations_applied)
                summary["live_calls"] = list(player.live_calls)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    except ReplayDivergenceError as exc:
        summary["verified"] = False
        summary["error"] = str(exc)
        print(f"Verification failed: {exc}", file=sys.stderr)
        if args.output_json:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_json).write_text(json.dumps(summary, indent=2))
        sys.exit(1)
    except LiveCallDuringReplayError as exc:
        summary["verified"] = False
        summary["error"] = str(exc)
        print(f"Strict replay aborted: {exc}", file=sys.stderr)
        if args.output_json:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_json).write_text(json.dumps(summary, indent=2))
        sys.exit(1)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(summary, indent=2))

    print(f"Replay mode: {summary['mode']}")
    print(f"Records: {summary.get('records', 'unknown')}")
    if summary.get("diverged"):
        print(f"Diverged: yes (mutations: {summary.get('mutations_applied')})")
    elif summary.get("mode") == "verify":
        print("Verification: passed")
    else:
        print("Diverged: no")


def _cmd_publish_benchmark(args: argparse.Namespace) -> None:
    """Phase 6.7: build a reviewer-ready benchmark publication bundle."""
    from .publish import BundleValidationError, PublicationBundle

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"Error: run_dir is not a directory: {run_dir}", file=sys.stderr)
        sys.exit(2)

    try:
        bundle = PublicationBundle.from_run_dir(
            run_dir,
            include_fairness_audit=bool(args.include_fairness_audit),
            contact=args.contact,
        )
    except BundleValidationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    summary: dict[str, Any] = {
        "run_dir": str(run_dir.resolve()),
        "include_fairness_audit": bool(args.include_fairness_audit),
        "manifest_present": bundle.artifacts.manifest_path is not None,
        "predictions_count": len(bundle.artifacts.predictions),
    }

    if args.validate:
        errors = bundle.validate()
        summary["validation_errors"] = errors
        if errors:
            for line in errors:
                print(f"validation error: {line}", file=sys.stderr)
            if args.output_json:
                Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output_json).write_text(json.dumps(summary, indent=2))
            sys.exit(1)
        print("Validation: passed")
        if args.output_json:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_json).write_text(json.dumps(summary, indent=2))
        return

    output = Path(args.output) if args.output else run_dir / "publication_bundle"
    try:
        written = bundle.write_to(output)
    except BundleValidationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    summary["output_dir"] = str(output.resolve())
    summary["files_written"] = sorted(written.keys())

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(summary, indent=2))

    print(f"Publication bundle written to: {output}")
    for name in sorted(written.keys()):
        print(f"  - {name}")


def _cmd_experiment_matrix(args: argparse.Namespace) -> None:
    payload = run_experiment_matrix(args.spec, output=args.output)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
    sys.stdout.write(render_matrix_report(payload) + "\n")
    print(f"JSON: {payload['summary_json']}")
    print(f"Markdown: {payload['summary_markdown']}")


def _cmd_archive(args: argparse.Namespace) -> None:
    archive_root = args.archive_root or str(Path.cwd() / ".apex_archives")
    payload = archive_runs(
        args.runs,
        archive_root=archive_root,
        prune_workspaces=bool(args.prune_workspaces),
        prune_runtime=bool(args.prune_runtime),
        compress_logs=bool(args.compress_logs),
        dry_run=bool(args.dry_run),
        force=bool(args.force),
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
    print(f"Dry run: {'yes' if payload['dry_run'] else 'no'}")
    print(f"Archives: {len(payload['archives'])}")
    print(f"Pruned paths: {len(payload['pruned_paths'])}")
    print(f"Compressed logs: {len(payload['compressed_logs'])}")
    print(f"Latest symlinks: {len(payload['latest_symlinks'])}")


def _cmd_commit0_benchmark(args: argparse.Namespace) -> None:
    config = _apply_common_overrides(_load_config(args.config), args.rollouts, args.model, None)
    config = _apply_phase6_overrides(config, args)
    # GOLD SCORING REQUIRED: refuse to run if the resolved commit0 contract does not require
    # gold expected-id scoring — otherwise an empty/failed expected-id inventory could fall
    # through to the visible-suite (pytest_summary) acceptance path and bank a false solve.
    from .evaluation.commit0_benchmark import _commit0_expected_id_scoring_required
    if not _commit0_expected_id_scoring_required(config):
        raise SystemExit(
            "commit0-benchmark: gold expected-id scoring is REQUIRED but the resolved "
            "evaluation contract does not require it; refusing to run with a visible-suite "
            "(pytest_summary) acceptance fallback. Set benchmark.evaluation_contract to the "
            "gold contract (mode=gold_suite_visible, scoring_universe=expected_test_ids).")
    if args.task_parallelism is not None:
        config.benchmark.task_parallelism = max(1, int(args.task_parallelism))
    if getattr(args, "single_model", False) and config.llm_configs:
        # Lock to the primary entry — required for the CAID head-to-head
        # so the model class is not the confound. Drops portfolio entries
        # and clears llm_profiles / scaffold_stage_llm_indices that
        # implicitly depend on multiple models being present.
        config.llm_configs = [config.llm_configs[0]]
        if config.rollout.llm_profiles:
            config.rollout.llm_profiles = []
        if config.rollout.scaffold_stage_llm_indices:
            config.rollout.scaffold_stage_llm_indices = {}
    seeds = max(1, int(getattr(args, "seeds", 1) or 1))
    if seeds > 1:
        _run_commit0_benchmark_multi_seed(args, config, seeds=seeds)
        return
    output_dir = args.output or str(default_commit0_output_dir(config, run_kind="apex"))
    runner = Commit0BenchmarkRunner(
        config=config,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        dataset_revision=args.dataset_revision,
        dataset_fallback_revisions=args.dataset_fallback_revision,
        split=args.split,
    )
    runner.config_source = str(Path(args.config).resolve()) if args.config else None
    report = runner.run(repos=args.repos, limit=args.limit)
    print(f"Commit0 average test pass rate: {report.average_pass_rate_percent:.1f}%")
    print(f"Baseline average test pass rate: {report.average_baseline_pass_rate_percent:.1f}%")
    print(f"Average test pass rate delta: {report.average_pass_rate_improvement_percent:+.1f}%")
    print(
        "Commit0 repo solve rate: "
        f"{report.solved_tasks}/{report.total_tasks} ({report.solved_rate_percent:.1f}%)"
    )
    if report.skipped_tasks:
        print(
            "Commit0 average test pass rate (runnable repos only): "
            f"{report.runnable_average_pass_rate_percent:.1f}%"
        )
        print(
            "Baseline average test pass rate (runnable repos only): "
            f"{report.runnable_average_baseline_pass_rate_percent:.1f}%"
        )
        print(
            "Average test pass rate delta (runnable repos only): "
            f"{report.runnable_average_pass_rate_improvement_percent:+.1f}%"
        )
        print(
            "Commit0 repo solve rate (runnable repos only): "
            f"{report.solved_runnable_tasks}/{report.runnable_tasks} "
            f"({report.runnable_solved_rate_percent:.1f}%)"
        )
        print(f"Skipped repos: {report.skipped_tasks}")
    print(f"Scoring method: {report.scoring_method}")
    print(f"Report: {Path(output_dir) / 'benchmark_report.json'}")


def _run_commit0_benchmark_multi_seed(
    args: argparse.Namespace,
    config,  # type: ApexConfig
    *,
    seeds: int,
) -> None:
    """Run the Commit0 benchmark `seeds` times, each into its own
    output_dir/seed_N/, and emit a Wilson 95% CI aggregate at the parent.

    Each seed run is independent (no shared state — workspace_dir is
    rebuilt per-seed under the seed dir). The aggregate is consumed
    downstream by reviewers asking "does the CAID head-to-head margin
    survive variance?". With seeds=5 across the 16-repo Lite split this
    is ~80 task-runs of compute, expected to take days.
    """
    import copy as _copy
    import json as _json

    from .evaluation.commit0_benchmark import (
        aggregate_seed_reports,
        render_seed_aggregate_markdown,
    )

    parent_output = Path(
        args.output or str(default_commit0_output_dir(config, run_kind="apex"))
    ).resolve()
    parent_output.mkdir(parents=True, exist_ok=True)

    seed_reports: list[dict[str, Any]] = []
    for seed_index in range(seeds):
        seed_dir = parent_output / f"seed_{seed_index}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Re-derive a per-seed config so each run uses an isolated
        # workspace_dir — important so concurrent worktrees, repo_memory
        # snapshots, and any per-run caches do not bleed across seeds.
        seed_config = _copy.deepcopy(config)
        seed_config.workspace_dir = str(seed_dir / ".workspaces")

        runner = Commit0BenchmarkRunner(
            config=seed_config,
            output_dir=str(seed_dir),
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            dataset_revision=args.dataset_revision,
            dataset_fallback_revisions=args.dataset_fallback_revision,
            split=args.split,
        )
        runner.config_source = str(Path(args.config).resolve()) if args.config else None
        print(f"[seed {seed_index + 1}/{seeds}] writing into {seed_dir}")
        report = runner.run(repos=args.repos, limit=args.limit)
        # Pull the per-seed JSON report so the aggregator sees the same
        # canonical structure that downstream tooling consumes.
        report_path = seed_dir / "benchmark_report.json"
        try:
            seed_reports.append(_json.loads(report_path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            print(f"[seed {seed_index + 1}/{seeds}] failed to load report ({exc})")
            continue
        print(
            f"[seed {seed_index + 1}/{seeds}] done — "
            f"runnable solved {report.solved_runnable_tasks}/{report.runnable_tasks}"
        )

    if not seed_reports:
        print("No seed reports produced; skipping aggregate.")
        return

    aggregate = aggregate_seed_reports(seed_reports)
    (parent_output / "caid_head_to_head_summary.json").write_text(
        _json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    (parent_output / "caid_head_to_head_summary.md").write_text(
        render_seed_aggregate_markdown(aggregate), encoding="utf-8"
    )
    print(
        f"Aggregate solve rate: "
        f"{aggregate['aggregate_successes']}/{aggregate['aggregate_trials']} "
        f"({100.0 * aggregate['aggregate_mean_solve_rate']:.1f}%) "
        f"[95% CI: {100.0 * aggregate['aggregate_wilson_ci_low']:.1f}%–"
        f"{100.0 * aggregate['aggregate_wilson_ci_high']:.1f}%]"
    )
    print(f"Aggregate JSON: {parent_output / 'caid_head_to_head_summary.json'}")
    print(f"Aggregate MD:   {parent_output / 'caid_head_to_head_summary.md'}")


def _cmd_swebench_pro_benchmark(args: argparse.Namespace) -> None:
    config = _apply_common_overrides(_load_config(args.config), args.rollouts, args.model, None)
    config = _apply_phase6_overrides(config, args)
    output_dir = args.output or str(default_swebench_pro_output_dir(config, run_kind="apex"))
    runner = SWEBenchProBenchmarkRunner(
        config=config,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        dockerhub_username=args.dockerhub_username,
        scripts_cache_dir=args.scripts_cache_dir,
        docker_platform=args.docker_platform,
        block_network=args.block_network,
        agent_visibility_mode=args.agent_visibility_mode,
        rollout_selection_policy=args.rollout_selection_policy,
    )
    runner.config_source = str(Path(args.config).resolve()) if args.config else None
    report = runner.run(
        instances=args.instances,
        repos=args.repos,
        languages=args.languages,
        limit=args.limit,
    )
    print(f"SWE-Bench Pro accuracy: {report.score_percent:.1f}%")
    print(f"Baseline accuracy: {report.baseline_score_percent:.1f}%")
    print(f"Accuracy delta: {report.score_improvement_percent:+.1f}%")
    print(f"Solved tasks: {report.solved_tasks}/{report.total_tasks}")
    if report.skipped_tasks:
        print(f"SWE-Bench Pro accuracy (runnable tasks only): {report.runnable_score_percent:.1f}%")
        print(
            f"Baseline accuracy (runnable tasks only): {report.runnable_baseline_score_percent:.1f}%"
        )
        print(
            f"Accuracy delta (runnable tasks only): {report.runnable_score_improvement_percent:+.1f}%"
        )
        print(f"Runnable tasks solved: {report.solved_runnable_tasks}/{report.runnable_tasks}")
        print(f"Skipped tasks: {report.skipped_tasks}")
    print(f"Scoring method: {report.scoring_method}")
    print(f"Report: {Path(output_dir) / 'benchmark_report.json'}")


def _cmd_swe_evo_benchmark(args: argparse.Namespace) -> None:
    """Run the SWE-EVO benchmark via the V5 in-container agent harness.

    Wraps :class:`apex.evaluation.swe_evo_benchmark.SWEEvoHarness`,
    surfaces the existing ``--agent-mode`` flag, and emits the canonical
    ``preds.json`` + ``report.json`` to ``--output``.
    """
    from .evaluation.swe_evo_benchmark import (
        SWEEvoHarness,
        SWEEvoHarnessConfig,
        load_swe_evo_tasks,
    )

    # ``_load_config(None)`` returns an ApexConfig whose default llm_config
    # routes through an installed CLI agent rather than the legacy
    # OPENAI_API backend; matches every other ``apex`` subcommand.
    config = _load_config(args.config)
    config = _apply_phase6_overrides(config, args)
    output_dir = (
        Path(args.output).expanduser().resolve() if args.output else Path.cwd() / ".apex_swe_evo"
    )

    sources_set = sum(1 for src in (args.arrow_path, args.jsonl_path) if src is not None)
    if sources_set != 1:
        print("Error: --swe-evo-benchmark requires exactly one of --arrow-path / --jsonl-path.")
        sys.exit(1)

    tasks = load_swe_evo_tasks(
        arrow_path=args.arrow_path,
        jsonl_path=args.jsonl_path,
        instance_ids=args.instances,
        repos=args.repos,
        limit=args.limit,
    )
    if not tasks:
        print("No SWE-EVO tasks resolved from the supplied filters.")
        sys.exit(1)

    harness_config = SWEEvoHarnessConfig(
        model_name=args.model_name,
        skip_clone=bool(args.skip_clone),
        include_intermediate_commits_in_prompt=not bool(args.no_intermediate_commits_in_prompt),
    )
    if args.max_turns is not None:
        harness_config.max_turns = int(args.max_turns)
    if args.per_tool_timeout_seconds is not None:
        harness_config.per_tool_timeout_seconds = int(args.per_tool_timeout_seconds)

    harness = SWEEvoHarness(
        output_dir=output_dir,
        config=harness_config,
        llm_config=config.llm_configs[0] if config.llm_configs else None,
    )
    report = harness.run(tasks)
    print(f"SWE-EVO total tasks: {report.total}")
    print(f"  succeeded: {report.succeeded}")
    print(f"  failed   : {report.failed}")
    print(f"  errored  : {report.errored}")
    print(f"  duration : {report.duration_seconds:.1f}s")
    print(f"  preds    : {output_dir / 'preds.json'}")
    print(f"  report   : {output_dir / 'report.json'}")
    if getattr(args, "agent_mode", "scaffolded") != "in_container_v5":
        print(
            f"Note: --agent-mode={getattr(args, 'agent_mode', 'scaffolded')!r} was supplied but the "
            "SWE-EVO harness is hard-wired to the in_container_v5 agent surface."
        )


def _cmd_tdd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a test suite against two pre-prepared sandboxes.

    Real-world / TDD entry point — no benchmark task object required.
    Loads test files from disk, materializes them into both sandboxes,
    runs the F2P oracle, and (optionally) layers mutation discrimination
    and minimization on top.
    """
    import json as _json
    import tempfile as _tempfile
    from pathlib import Path as _Path

    from .evaluation import (
        evaluate_mutation_score,
        evaluate_tdd_iteration,
        generate_mutants,
        minimize_suite,
    )

    test_artifacts: list[dict[str, Any]] = []
    for tf in args.test_files or []:
        tf_path = _Path(tf).expanduser()
        if not tf_path.exists():
            print(f"Warning: test file {tf_path} does not exist; skipping.")
            continue
        try:
            content = tf_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"Warning: could not read {tf_path}: {exc}")
            continue
        test_artifacts.append(
            {
                "path": str(tf_path.name if "/" not in tf else tf),
                "content": content,
            }
        )
    if not test_artifacts:
        print("No usable test files supplied; nothing to evaluate.")
        sys.exit(1)

    output_dir = (
        _Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _Path(_tempfile.mkdtemp(prefix="apex-tdd-evaluate-"))
    )

    f2p_report = evaluate_tdd_iteration(
        broken_dir=_Path(args.broken_dir).expanduser().resolve(),
        fixed_dir=_Path(args.fixed_dir).expanduser().resolve(),
        test_artifacts=test_artifacts,
        output_dir=output_dir,
        language=args.language,
        timeout_seconds=args.timeout_seconds,
        install_repo=args.install_repo,
    )

    mutation_payload: dict[str, Any] = {}
    if args.enable_mutation and f2p_report.get("summary", {}).get("any_f2p"):
        targets = list(args.mutation_targets or [])
        if not targets:
            print("--enable-mutation requires --mutation-target — skipping mutation step.")
        else:
            mutants = []
            fixed_dir_resolved = _Path(args.fixed_dir).expanduser().resolve()
            for target in targets:
                target_path = fixed_dir_resolved / target
                if not target_path.exists():
                    print(f"Warning: mutation target {target_path} does not exist; skipping.")
                    continue
                file_mutants = generate_mutants(
                    source_path=target_path,
                    language=args.language,
                    max_mutants=8,
                )
                for m in file_mutants:
                    m.source_path = target
                mutants.extend(file_mutants)
            if mutants:
                test_paths = [a["path"] for a in test_artifacts if a.get("path")]
                report = evaluate_mutation_score(
                    fixed_dir=fixed_dir_resolved,
                    mutants=mutants,
                    test_paths=test_paths,
                    language=args.language,
                )
                mutation_payload = report.to_dict()
            else:
                mutation_payload = {"skip_reason": "no_mutants_generated"}

    minimization_payload: dict[str, Any] = {}
    if args.enable_minimization:
        kept_artifacts, min_report = minimize_suite(
            test_artifacts=test_artifacts,
            f2p_payload=f2p_report,
            mutation_report=mutation_payload or None,
        )
        minimization_payload = {
            "kept": [a.get("path") for a in kept_artifacts],
            "report": min_report.to_dict(),
        }

    final_payload = {
        "f2p": f2p_report,
        "mutation": mutation_payload,
        "minimization": minimization_payload,
        "summary": {
            "any_f2p": bool(f2p_report.get("summary", {}).get("any_f2p")),
            "f2p_count": int(f2p_report.get("summary", {}).get("f2p_count") or 0),
            "f2p_rate": float(f2p_report.get("summary", {}).get("f2p_rate") or 0.0),
            "mutation_score": float(mutation_payload.get("mutation_score") or 0.0),
            "tests_evaluated": int(f2p_report.get("summary", {}).get("tests_observed") or 0),
            "minimized_count": int(
                (minimization_payload.get("report") or {}).get("minimized_count") or 0
            ),
        },
    }

    serialized = _json.dumps(final_payload, indent=2, default=str)
    if args.output_json:
        output_json_path = _Path(args.output_json).expanduser().resolve()
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(serialized, encoding="utf-8")
        print(f"Report: {output_json_path}")
    else:
        print(serialized)


def _load_prewritten_test_artifacts(
    paths: list[str] | None,
) -> list[dict[str, Any]]:
    """Load pre-written test files from disk for the H.2 CLI commands.

    Returns artifacts in the same shape the modes API expects
    (``[{"path": ..., "content": ...}]``). Used to bypass real LLM
    generation when the caller already has tests in hand.

    The artifact's ``path`` is set to ``tests/<basename>`` (or the
    relative path if the caller passes one) so the materializer drops
    the test file at a sensible repo-relative location inside each
    sandbox. The caller can override by passing a path that already
    starts with ``tests/`` or ``test/`` — that's preserved verbatim.
    """
    artifacts: list[dict[str, Any]] = []
    for tf in paths or []:
        path_obj = Path(tf).expanduser()
        if not path_obj.exists():
            print(f"Warning: test file {path_obj} does not exist; skipping.")
            continue
        try:
            content = path_obj.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"Warning: could not read {path_obj}: {exc}")
            continue
        # Pick the in-sandbox path: prefer a relative form starting
        # with tests/ or test/ when the caller already provided one,
        # otherwise default to tests/<basename> so the materializer
        # writes into a conventional location inside each sandbox.
        if not path_obj.is_absolute() and tf.replace("\\", "/").startswith(("tests/", "test/")):
            in_sandbox_path = tf
        else:
            in_sandbox_path = f"tests/{path_obj.name}"
        artifacts.append({"path": in_sandbox_path, "content": content})
    return artifacts


def _emit_mode_result(payload: dict[str, Any], output_json: str | None) -> None:
    """Serialize a ModeResult to stdout or to --output-json."""
    import json as _json

    serialized = _json.dumps(payload, indent=2, default=str)
    if output_json:
        out_path = Path(output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(serialized, encoding="utf-8")
        print(f"Report: {out_path}")
    else:
        print(serialized)


def _cmd_testgen_for_fix(args: argparse.Namespace) -> None:
    """Mode 1 CLI: generate tests for a known fix.

    The CLI accepts a `--patch-file` (the gold fix) and optional
    `--test-file` arguments to bypass real LLM generation. Without a
    bundled generator, the default no-op returns no artifacts and the
    output reports a clear "no artifacts produced" error. Callers who
    want LLM-backed generation should use the apex.modes Python API
    directly with a custom test_generator callable.
    """
    from .modes import run_testgen_with_fix

    patch_path = Path(args.patch_file).expanduser()
    if not patch_path.exists():
        print(f"Error: --patch-file {patch_path} does not exist.")
        sys.exit(1)
    gold_patch = patch_path.read_text(encoding="utf-8")

    prewritten = _load_prewritten_test_artifacts(args.prewritten_test_files)
    test_generator = (lambda r, p: prewritten) if prewritten else None

    result = run_testgen_with_fix(
        repo_path=args.repo_path,
        problem_statement=args.problem_statement,
        gold_patch=gold_patch,
        output_dir=args.output_dir,
        test_generator=test_generator,
        language=args.language,
        install_repo=args.install_repo,
    )
    _emit_mode_result(result.to_dict(), args.output_json)
    sys.exit(0 if result.success else 2)


def _cmd_codegen_for_tests(args: argparse.Namespace) -> None:
    """Mode 2 CLI: generate code for a given test suite.

    Symmetric to testgen-for-fix: takes `--test-file` for the gold
    tests and an optional `--patch-file` to bypass LLM generation.
    """
    from .modes import run_codegen_with_tests

    test_artifacts = _load_prewritten_test_artifacts(args.test_files)
    if not test_artifacts:
        print("Error: at least one --test-file must produce a usable artifact.")
        sys.exit(1)

    code_generator = None
    if args.patch_file:
        patch_path = Path(args.patch_file).expanduser()
        if not patch_path.exists():
            print(f"Error: --patch-file {patch_path} does not exist.")
            sys.exit(1)
        prewritten_patch = patch_path.read_text(encoding="utf-8")

        def _stub_code_gen(_r, _p, _t):
            return prewritten_patch

        code_generator = _stub_code_gen

    result = run_codegen_with_tests(
        repo_path=args.repo_path,
        problem_statement=args.problem_statement,
        gold_test_artifacts=test_artifacts,
        output_dir=args.output_dir,
        code_generator=code_generator,
        language=args.language,
        install_repo=args.install_repo,
        agent_mode=getattr(args, "agent_mode", "scaffolded"),
        docker_image=getattr(args, "docker_image", None),
        max_turns=getattr(args, "max_turns", None),
        # Phase 6.5: forwarded only when --agent-mode hierarchical_v5.
        total_budget=getattr(args, "total_budget", None),
        n_subtasks=getattr(args, "n_subtasks", None),
        rebalance_strategy=getattr(args, "rebalance_strategy", "feedback"),
    )
    _emit_mode_result(result.to_dict(), args.output_json)
    sys.exit(0 if result.success else 2)


def _cmd_generate_both(args: argparse.Namespace) -> None:
    """Mode 3 CLI: chained testgen → codegen.

    Optional `--test-file` / `--patch-file` let the caller short-circuit
    either phase by supplying the artifact directly. Useful for testing
    the chain wiring without LLMs.
    """
    from .modes import run_generate_both

    prewritten_tests = _load_prewritten_test_artifacts(args.prewritten_test_files)
    test_generator = (lambda r, p: prewritten_tests) if prewritten_tests else None

    code_generator = None
    if args.patch_file:
        patch_path = Path(args.patch_file).expanduser()
        if not patch_path.exists():
            print(f"Error: --patch-file {patch_path} does not exist.")
            sys.exit(1)
        prewritten_patch = patch_path.read_text(encoding="utf-8")

        def _stub_code_gen(_r, _p, _t):
            return prewritten_patch

        code_generator = _stub_code_gen

    result = run_generate_both(
        repo_path=args.repo_path,
        problem_statement=args.problem_statement,
        output_dir=args.output_dir,
        test_generator=test_generator,
        code_generator=code_generator,
        language=args.language,
        install_repo=args.install_repo,
    )
    _emit_mode_result(result.to_dict(), args.output_json)
    sys.exit(0 if result.success else 2)


def _cmd_calibrate(args: argparse.Namespace) -> None:
    """Compute Brier / ECE / reliability tables for past APEX runs."""

    bins = max(2, int(args.bins))
    aggregate = []
    for run_root in args.runs:
        root_path = Path(run_root).expanduser()
        if not root_path.exists():
            print(f"Warning: {root_path} does not exist; skipping.")
            continue
        reports = build_calibration_reports(root_path, bin_count=bins)
        for report in reports:
            aggregate.append((str(root_path), report))

    if not aggregate:
        print("No apex_result.json files found under the supplied --runs paths.")
        sys.exit(1)

    print()
    for root_path, report in aggregate:
        print(f"=== {root_path} :: {report.score_name} → {report.label_name} ===")
        print(f"  samples              : {report.sample_count}")
        print(f"  base_rate            : {report.base_rate:.4f}")
        print(f"  mean_predicted       : {report.mean_predicted:.4f}")
        print(f"  brier_score          : {report.brier_score:.4f}")
        print(f"  expected_calibration : {report.expected_calibration_error:.4f}")
        print()

    if args.output_json:
        json_path = Path(args.output_json).expanduser()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                [
                    {"runs_root": root_path, "report": report.to_dict()}
                    for root_path, report in aggregate
                ],
                indent=2,
            )
        )
        print(f"Wrote JSON: {json_path}")
    if args.output_markdown:
        md_path = Path(args.output_markdown).expanduser()
        md_path.parent.mkdir(parents=True, exist_ok=True)
        sections = []
        for root_path, report in aggregate:
            sections.append(f"### Source: `{root_path}`")
            sections.append("")
            sections.append(render_reliability_markdown(report))
            sections.append("")
        md_path.write_text("\n".join(sections))
        print(f"Wrote Markdown: {md_path}")


if __name__ == "__main__":
    main()
