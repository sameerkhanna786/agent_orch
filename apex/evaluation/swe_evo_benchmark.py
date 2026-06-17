"""
SWE-EVO benchmark driver.

SWE-EVO is a long-horizon software-evolution benchmark from Fsoft-AIC
(arXiv 2512.18470). Each task captures a multi-PR evolution between two
versions of an open-source Python project (Django, NumPy, dvc, dask,
requests, modin, pydantic, conan, scikit-learn) and is scored via an
SWE-bench-style ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` rule against the
**final** (``end_version``) commit.

See ``tools/SWE_EVO_NOTES.md`` for the full schema notes (commit
``9b83d5af943ba7a17567336f5b18239f73960219``).

DESIGN NOTES
------------
- The intermediate-PR list (`PRs`) is surfaced as **planning context** in
  the agent prompt rather than as separate scoring checkpoints. The
  upstream harness only scores the final ``end_version_commit`` patch, so
  our driver mirrors that. A configuration knob
  (``score_per_intermediate_commit``) is wired in for forward
  compatibility but defaults to False.
- The agent surface is the V5 in-container loop
  (``apex.orchestrator_in_container_agent``). Each task gets its own
  workspace dir; the loop calls back into the workspace via shell
  ``run_in_container`` tool calls. The host-side V1 sandbox is "cwd-only,
  no docker" — see ``InContainerAgent`` module docstring for limitations.
- The driver writes a SWE-agent-shaped ``preds.json``
  (``{instance_id: {model_patch, model_name_or_path, instance_id}}``) that
  the upstream ``SWE-bench/evaluate_instance.py --scaffold SWE-agent``
  consumes without modification.

PUBLIC INTERFACE
----------------
- :class:`SWEEvoTask`
- :class:`SWEEvoTaskResult`
- :class:`SWEEvoBenchmarkReport`
- :class:`SWEEvoHarness` — orchestrates per-task agent runs + writes preds.json
- :func:`load_swe_evo_tasks` — iterate the in-repo arrow dataset / JSONL
- :func:`build_problem_statement` — compose the agent's prompt from a task
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from ..orchestrator_in_container_agent import (
    DEFAULT_MAX_OUTPUT_BYTES_PER_TOOL,
    DEFAULT_MAX_TURNS,
    DEFAULT_TURN_TIMEOUT_SECONDS,
    AgentRunSummary,
    InContainerAgent,
    LLMCaller,
)
from .target_runtime import host_env_runtime, target_tool_env_overrides

logger = logging.getLogger("apex.evaluation.swe_evo")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SWE_EVO_DATASET_NAME = "Fsoft-AIC/SWE-EVO"
SWE_EVO_DATASET_SPLIT = "test"
SWE_EVO_DEFAULT_INSTANCE_COUNT = 48  # test split as of 2026-05
SWE_EVO_HARNESS_NAME = "swe_evo_official_harness"
SWE_EVO_HARNESS_VERSION = "2026-05-07.1"
SWE_EVO_REPORT_KIND_APEX = "apex_swe_evo"
SWE_EVO_PREDS_FILENAME = "preds.json"
SWE_EVO_REPORT_FILENAME = "report.json"
SWE_EVO_RECORDS_DIR = "records"
SWE_EVO_LOGS_DIR = "logs"
SWE_EVO_WORKSPACES_DIR = "workspaces"


def _sanitize_path_segment(value: str) -> str:
    import re as _re

    cleaned = _re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._-")
    return cleaned or "task"


def _make_jsonl_transcript_sink(path: "Path"):
    """1B: an append-only, fsync-durable JSONL sink (flock-guarded where
    available) for V5 per-turn transcript records, so working memory survives a
    crash/container reap and can be preloaded on restart."""
    try:
        import fcntl  # type: ignore

        have_fcntl = True
    except ImportError:  # pragma: no cover - non-POSIX
        fcntl = None  # type: ignore
        have_fcntl = False

    def _sink(record: dict) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                if have_fcntl:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    handle.write(json.dumps(record, default=repr) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    if have_fcntl:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            logger.debug("swe_evo transcript sink write failed", exc_info=True)

    return _sink


# ---------------------------------------------------------------------------
# Task / result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SWEEvoPullRequest:
    """One intermediate PR between ``start_version`` and ``end_version``.

    SWE-EVO tasks ship a list of these as ``PRs``. We surface them to the
    agent as planning context (they describe how the human-authored
    evolution actually happened).
    """

    pr_link: str = ""
    pr_url: str = ""
    pr_number: Optional[int] = None
    pr_title: Optional[str] = None
    is_issue: bool = False
    is_mentioned_in_release_note: bool = False
    changed_test_files: list[str] = field(default_factory=list)
    patch_without_test: str = ""
    test_patch: str = ""

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "SWEEvoPullRequest":
        if not isinstance(row, dict):
            return cls()
        return cls(
            pr_link=str(row.get("pr_link") or ""),
            pr_url=str(row.get("pr_url") or ""),
            pr_number=(
                int(row["pr_number"]) if row.get("pr_number") not in (None, "", "None") else None
            ),
            pr_title=(str(row["pr_title"]) if row.get("pr_title") else None),
            is_issue=bool(row.get("is_issue") or False),
            is_mentioned_in_release_note=bool(row.get("is_mentioned_in_release_note") or False),
            changed_test_files=[str(p) for p in (row.get("changed_test_files") or []) if p],
            patch_without_test=str(row.get("patch_without_test") or ""),
            test_patch=str(row.get("test_patch") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pr_link": self.pr_link,
            "pr_url": self.pr_url,
            "pr_number": self.pr_number,
            "pr_title": self.pr_title,
            "is_issue": self.is_issue,
            "is_mentioned_in_release_note": self.is_mentioned_in_release_note,
            "changed_test_files": list(self.changed_test_files),
            "patch_without_test": self.patch_without_test,
            "test_patch": self.test_patch,
        }


@dataclass
class SWEEvoTask:
    """One SWE-EVO benchmark task (multi-commit evolution)."""

    instance_id: str
    repo: str
    base_commit: str
    end_version_commit: str
    start_version: str
    end_version: str
    problem_statement: str
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    intermediate_commits: list[SWEEvoPullRequest] = field(default_factory=list)
    image: str = ""
    test_cmds: str = ""
    log_parser: str = ""
    version: str = ""
    environment_setup_commit: str = ""
    gold_patch: str = ""  # in-tree row.patch — agent must not see this
    gold_test_patch: str = ""  # in-tree row.test_patch — agent must not see this
    bench: str = ""
    instance_id_swe: str = ""

    # ------------------------------------------------------------------
    # Properties / helpers
    # ------------------------------------------------------------------

    @property
    def required_tests(self) -> list[str]:
        return sorted(set(self.fail_to_pass) | set(self.pass_to_pass))

    @property
    def repo_name(self) -> str:
        return self.repo.split("/")[-1]

    @property
    def intermediate_commit_count(self) -> int:
        return len(self.intermediate_commits)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "end_version_commit": self.end_version_commit,
            "start_version": self.start_version,
            "end_version": self.end_version,
            "problem_statement": self.problem_statement,
            "fail_to_pass": list(self.fail_to_pass),
            "pass_to_pass": list(self.pass_to_pass),
            "intermediate_commits": [pr.to_dict() for pr in self.intermediate_commits],
            "image": self.image,
            "test_cmds": self.test_cmds,
            "log_parser": self.log_parser,
            "version": self.version,
            "environment_setup_commit": self.environment_setup_commit,
            "bench": self.bench,
            "instance_id_swe": self.instance_id_swe,
            # gold_* deliberately excluded from default to_dict — caller
            # must opt in via ``to_dict_with_gold`` if they want it (e.g.
            # for harness re-scoring of the human reference).
        }

    def to_dict_with_gold(self) -> dict[str, Any]:
        d = self.to_dict()
        d["gold_patch"] = self.gold_patch
        d["gold_test_patch"] = self.gold_test_patch
        return d

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "SWEEvoTask":
        prs_raw = row.get("PRs") or row.get("intermediate_commits") or []
        intermediate = [SWEEvoPullRequest.from_row(pr) for pr in prs_raw if isinstance(pr, dict)]
        # Tolerate both upstream casing (FAIL_TO_PASS) and the lower-case
        # variant some local fixtures use.
        f2p = row.get("FAIL_TO_PASS") or row.get("fail_to_pass") or []
        p2p = row.get("PASS_TO_PASS") or row.get("pass_to_pass") or []
        return cls(
            instance_id=str(row["instance_id"]),
            repo=str(row["repo"]),
            base_commit=str(row.get("base_commit") or ""),
            end_version_commit=str(
                row.get("end_version_commit")
                or row.get("target_commit")
                or row.get("environment_setup_commit")
                or ""
            ),
            start_version=str(row.get("start_version") or ""),
            end_version=str(row.get("end_version") or row.get("version") or ""),
            problem_statement=str(row.get("problem_statement") or ""),
            fail_to_pass=[str(t) for t in f2p],
            pass_to_pass=[str(t) for t in p2p],
            intermediate_commits=intermediate,
            image=str(row.get("image") or ""),
            test_cmds=str(row.get("test_cmds") or ""),
            log_parser=str(row.get("log_parser") or ""),
            version=str(row.get("version") or row.get("end_version") or ""),
            environment_setup_commit=str(row.get("environment_setup_commit") or ""),
            gold_patch=str(row.get("patch") or ""),
            gold_test_patch=str(row.get("test_patch") or ""),
            bench=str(row.get("bench") or ""),
            instance_id_swe=str(row.get("instance_id_swe") or ""),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "SWEEvoTask":
        return cls.from_row(json.loads(Path(path).read_text()))


@dataclass
class SWEEvoTaskResult:
    """Per-task outcome of one APEX/agent run."""

    instance_id: str
    success: bool
    final_patch: Optional[str] = None
    submission_ready: bool = False
    officially_evaluated: bool = False
    official_success: Optional[bool] = None
    terminated_reason: str = ""  # "submit_patch" / "give_up" / "max_turns" / ...
    give_up_reason: Optional[str] = None
    turn_count: int = 0
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    workspace_dir: Optional[str] = None
    summary: Optional[dict[str, Any]] = None  # AgentRunSummary.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "success": self.success,
            "submission_ready": self.submission_ready,
            "officially_evaluated": self.officially_evaluated,
            "official_success": self.official_success,
            "final_patch_chars": len(self.final_patch or ""),
            "terminated_reason": self.terminated_reason,
            "give_up_reason": self.give_up_reason,
            "turn_count": self.turn_count,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "error": self.error,
            "workspace_dir": self.workspace_dir,
            "summary": self.summary,
        }

    def to_swebench_prediction(self, model_name: str) -> dict[str, Any]:
        """Render this result in the SWE-agent ``preds.json`` shape."""
        return {
            "instance_id": self.instance_id,
            "model_patch": self.final_patch or "",
            "model_name_or_path": model_name,
        }


@dataclass
class SWEEvoBenchmarkReport:
    """Top-level report aggregating per-task results."""

    model_name: str
    total: int
    succeeded: int
    failed: int
    errored: int
    started_at: float
    finished_at: float
    results: list[SWEEvoTaskResult] = field(default_factory=list)
    harness_name: str = SWE_EVO_HARNESS_NAME
    harness_version: str = SWE_EVO_HARNESS_VERSION
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.finished_at - self.started_at)

    def success_rate(self) -> float:
        return (self.succeeded / self.total) if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "harness_name": self.harness_name,
            "harness_version": self.harness_version,
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "errored": self.errored,
            "success_rate": round(self.success_rate(), 6),
            "duration_seconds": round(self.duration_seconds, 4),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "notes": dict(self.notes),
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Dataset loading (handles HF arrow + JSONL + raw dicts)
# ---------------------------------------------------------------------------


def load_swe_evo_tasks(
    *,
    arrow_path: Optional[str | Path] = None,
    jsonl_path: Optional[str | Path] = None,
    rows: Optional[Iterable[dict[str, Any]]] = None,
    instance_ids: Optional[Iterable[str]] = None,
    repos: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
) -> list[SWEEvoTask]:
    """Load SWE-EVO tasks from an arrow file, JSONL file, or in-memory rows.

    Exactly one of ``arrow_path`` / ``jsonl_path`` / ``rows`` must be set.
    Filtering by ``instance_ids`` and/or ``repos`` is applied after load.
    """
    sources_set = sum(1 for src in (arrow_path, jsonl_path, rows) if src is not None)
    if sources_set != 1:
        raise ValueError("load_swe_evo_tasks: pass exactly one of arrow_path / jsonl_path / rows")

    raw_rows: list[dict[str, Any]] = []
    if arrow_path is not None:
        try:
            from datasets import Dataset  # local import to keep test cost low
        except ImportError as exc:  # pragma: no cover — environment issue
            raise RuntimeError("datasets package required to load SWE-EVO arrow files") from exc
        ds = Dataset.from_file(str(arrow_path))
        raw_rows.extend(ds[i] for i in range(len(ds)))
    elif jsonl_path is not None:
        with open(str(jsonl_path), "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "load_swe_evo_tasks: skipping malformed JSONL row: %s",
                        exc,
                    )
    else:
        for row in rows or []:
            if isinstance(row, dict):
                raw_rows.append(row)

    iid_filter = set(instance_ids or [])
    repo_filter = set(repos or [])
    out: list[SWEEvoTask] = []
    for row in raw_rows:
        try:
            task = SWEEvoTask.from_row(row)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "load_swe_evo_tasks: skipping row that failed to parse: %s",
                exc,
            )
            continue
        if iid_filter and task.instance_id not in iid_filter:
            continue
        if repo_filter and task.repo not in repo_filter:
            continue
        out.append(task)
        if limit is not None and len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Workspace prep
# ---------------------------------------------------------------------------


def _git_or_die(*args: str, cwd: Optional[str] = None) -> subprocess.CompletedProcess[str]:
    """Run a ``git`` subcommand; raise on failure."""
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed (rc={proc.returncode}): {proc.stderr}")
    return proc


def prepare_task_workspace(
    task: SWEEvoTask,
    *,
    workspace_root: str | Path,
    repo_clone_url: Optional[str] = None,
    skip_clone: bool = False,
) -> str:
    """Materialize a fresh workspace at ``base_commit`` for one task.

    Default: clones from the canonical GitHub URL ``https://github.com/<repo>``
    and checks out ``base_commit``. When ``skip_clone=True`` (used by tests
    and for offline / docker-only flows) we just create an empty workspace
    dir; the agent's first ``run_in_container`` calls would normally pull
    the prepared image.
    """
    workspace_root = Path(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    workspace_dir = workspace_root / task.instance_id
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=False)

    if skip_clone:
        return str(workspace_dir)

    url = repo_clone_url or f"https://github.com/{task.repo}.git"
    try:
        _git_or_die("clone", "--quiet", url, str(workspace_dir))
        if task.base_commit:
            _git_or_die("checkout", "--quiet", task.base_commit, cwd=str(workspace_dir))
    except RuntimeError as exc:
        logger.warning(
            "prepare_task_workspace: git operation failed for %s: %s",
            task.instance_id,
            exc,
        )
        # Leave the (possibly partial) workspace in place; caller decides
        # whether to proceed or skip.
    return str(workspace_dir)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def build_problem_statement(
    task: SWEEvoTask,
    *,
    include_intermediate_commits: bool = True,
    max_intermediate_commits_in_prompt: int = 12,
    max_chars_per_intermediate: int = 1200,
) -> str:
    """Compose the agent prompt from a SWE-EVO task.

    The intermediate ``PRs`` are surfaced as planning context (a short
    summary per PR, capped to ``max_chars_per_intermediate`` characters).
    Tests' contents are NOT shown — only changed-test filenames and PR
    titles. Required-test names are listed at the bottom.
    """
    lines: list[str] = [
        "Evolve this repository from one version to the next.",
        "",
        f"Repository: {task.repo}",
        f"Start version: {task.start_version} (base commit: {task.base_commit})",
        f"End version: {task.end_version} (target commit: {task.end_version_commit})",
        "",
        "## Software Requirement Specification",
        task.problem_statement.strip() or "(no SRS provided)",
    ]
    if include_intermediate_commits and task.intermediate_commits:
        lines.append("")
        lines.append(f"## Intermediate evolution evidence ({task.intermediate_commit_count} PRs)")
        for pr in task.intermediate_commits[:max_intermediate_commits_in_prompt]:
            header = f"- PR #{pr.pr_number or '?'}"
            if pr.pr_title:
                header += f": {pr.pr_title}"
            if pr.pr_url:
                header += f" ({pr.pr_url})"
            lines.append(header)
            if pr.changed_test_files:
                preview = ", ".join(pr.changed_test_files[:6])
                more = (
                    f" (+{len(pr.changed_test_files) - 6} more)"
                    if len(pr.changed_test_files) > 6
                    else ""
                )
                lines.append(f"  changed tests: {preview}{more}")
            if pr.patch_without_test:
                snippet = pr.patch_without_test
                if len(snippet) > max_chars_per_intermediate:
                    snippet = snippet[:max_chars_per_intermediate] + "\n…[truncated]"
                lines.append("  patch (truncated):")
                lines.extend(f"    {line}" for line in snippet.splitlines()[:40])
        if task.intermediate_commit_count > max_intermediate_commits_in_prompt:
            lines.append(
                f"... ({task.intermediate_commit_count - max_intermediate_commits_in_prompt}"
                " additional PRs not shown)"
            )
    if task.required_tests:
        lines.append("")
        lines.append("## Required tests (must pass after your patch)")
        preview = task.required_tests[:25]
        lines.extend(f"  - {t}" for t in preview)
        if len(task.required_tests) > 25:
            lines.append(f"  ... (+{len(task.required_tests) - 25} more)")
    lines.append("")
    lines.append(
        "Submit a single unified-diff patch via the submit_patch tool when ready. "
        "You may run shell commands inside the workspace via run_in_container "
        "to inspect the codebase or run tests."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


# Type alias: a TaskAgentRunner takes (task, workspace_dir) and returns
# the final unified diff (or None). Default implementation runs the V5
# in-container loop. Tests inject a stub for hermetic runs.
TaskAgentRunner = Callable[[SWEEvoTask, str], Optional[str]]


@dataclass
class SWEEvoHarnessConfig:
    """Configuration for one :class:`SWEEvoHarness` run."""

    model_name: str = "apex-swe-evo"
    max_turns: int = DEFAULT_MAX_TURNS
    per_tool_timeout_seconds: int = DEFAULT_TURN_TIMEOUT_SECONDS
    max_tool_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES_PER_TOOL
    score_per_intermediate_commit: bool = False  # forward-compat; default off
    skip_clone: bool = False
    include_intermediate_commits_in_prompt: bool = True


class SWEEvoHarness:
    """Run an agent across a set of :class:`SWEEvoTask` instances.

    The harness is intentionally thin — it owns task-level orchestration,
    workspace materialization, prediction-file emission, and per-task
    error isolation. It delegates the actual solving to a
    :class:`TaskAgentRunner` callable; the default uses the V5
    in-container agent loop.
    """

    def __init__(
        self,
        *,
        output_dir: str | Path,
        config: Optional[SWEEvoHarnessConfig] = None,
        llm_config: Any = None,
        llm_caller: Optional[LLMCaller] = None,
        task_agent_runner: Optional[TaskAgentRunner] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or SWEEvoHarnessConfig()
        self.llm_config = llm_config
        self.llm_caller = llm_caller
        self._task_agent_runner = task_agent_runner

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        tasks: Iterable[SWEEvoTask],
        *,
        repo_clone_url_override: Optional[Callable[[SWEEvoTask], Optional[str]]] = None,
    ) -> SWEEvoBenchmarkReport:
        tasks = list(tasks)
        records_dir = self.output_dir / SWE_EVO_RECORDS_DIR
        logs_dir = self.output_dir / SWE_EVO_LOGS_DIR
        workspaces_dir = self.output_dir / SWE_EVO_WORKSPACES_DIR
        records_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        workspaces_dir.mkdir(parents=True, exist_ok=True)

        results: list[SWEEvoTaskResult] = []
        succeeded = failed = errored = 0
        started_at = time.time()
        for task in tasks:
            url_override = repo_clone_url_override(task) if repo_clone_url_override else None
            result = self._run_one(
                task,
                workspaces_root=workspaces_dir,
                repo_clone_url=url_override,
            )
            results.append(result)
            (records_dir / f"{task.instance_id}.json").write_text(
                json.dumps(result.to_dict(), indent=2)
            )
            if result.error:
                errored += 1
            elif result.success:
                succeeded += 1
            else:
                failed += 1
        finished_at = time.time()

        report = SWEEvoBenchmarkReport(
            model_name=self.config.model_name,
            total=len(tasks),
            succeeded=succeeded,
            failed=failed,
            errored=errored,
            started_at=started_at,
            finished_at=finished_at,
            results=results,
            notes={
                "in_container_loop_v1_limitations": (
                    "target-runtime shell shims + cwd pin only; official "
                    "SWE-EVO evaluate_instance.py scoring is authoritative."
                ),
                "scoring_model": (
                    "final-commit only; intermediate PRs surfaced as planning context"
                    if not self.config.score_per_intermediate_commit
                    else "per-intermediate-commit (forward-compat; not yet wired)"
                ),
                "submission_shape": "swe_agent_preds_json",
                "success_authority": "official_swe_evo_evaluator_only",
            },
        )
        self._write_predictions(report)
        self._write_report(report)
        return report

    def run_one_task(self, task: SWEEvoTask) -> SWEEvoTaskResult:
        """Convenience: run exactly one task. Materializes a temp workspace."""
        workspaces_dir = self.output_dir / SWE_EVO_WORKSPACES_DIR
        workspaces_dir.mkdir(parents=True, exist_ok=True)
        return self._run_one(task, workspaces_root=workspaces_dir)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_one(
        self,
        task: SWEEvoTask,
        *,
        workspaces_root: Path,
        repo_clone_url: Optional[str] = None,
    ) -> SWEEvoTaskResult:
        run_started = time.time()
        workspace_dir: Optional[str] = None
        try:
            workspace_dir = prepare_task_workspace(
                task,
                workspace_root=workspaces_root,
                repo_clone_url=repo_clone_url,
                skip_clone=self.config.skip_clone,
            )
        except (OSError, RuntimeError) as exc:
            logger.warning("swe_evo: workspace prep failed for %s: %s", task.instance_id, exc)
            return SWEEvoTaskResult(
                instance_id=task.instance_id,
                success=False,
                error=f"workspace_prep_failed: {exc}",
                elapsed_seconds=time.time() - run_started,
                workspace_dir=workspace_dir,
            )

        problem_statement = build_problem_statement(
            task,
            include_intermediate_commits=(self.config.include_intermediate_commits_in_prompt),
        )

        # If the caller supplied a custom runner (tests, alt agents), use it
        # directly. Otherwise drive the V5 in-container loop.
        try:
            if self._task_agent_runner is not None:
                final_patch = self._task_agent_runner(task, workspace_dir)
                summary_dict: Optional[dict[str, Any]] = None
                terminated_reason = "submit_patch" if final_patch else "give_up"
                turn_count = 0
                give_up_reason = None if final_patch else "custom_runner_returned_none"
            else:
                target_tool_env, _target_tool_diag = target_tool_env_overrides(
                    workdir=Path(workspace_dir),
                    output_dir=self.output_dir / "target_runtime_tools" / task.instance_id,
                    timeout_seconds=self.config.per_tool_timeout_seconds,
                    runtime=host_env_runtime(
                        os.environ,
                        description="swe_evo_workspace_runtime",
                    ),
                    label=f"swe_evo_{task.instance_id}",
                )
                # 1B: durable append-only transcript so working memory survives a
                # crash/reap; preload it on restart before the loop resumes.
                transcript_path = (
                    self.output_dir
                    / SWE_EVO_RECORDS_DIR
                    / f"{_sanitize_path_segment(task.instance_id)}.transcript.jsonl"
                )
                agent = InContainerAgent(
                    llm_config=self.llm_config,
                    workspace_dir=workspace_dir,
                    max_turns=self.config.max_turns,
                    per_tool_timeout_seconds=self.config.per_tool_timeout_seconds,
                    max_tool_output_bytes=self.config.max_tool_output_bytes,
                    llm_caller=self.llm_caller,
                    env_overrides=target_tool_env,
                    transcript_sink=_make_jsonl_transcript_sink(transcript_path),
                )
                try:
                    agent.preload_transcript(str(transcript_path))
                except Exception:  # noqa: BLE001 - preload is best-effort
                    logger.debug("swe_evo: transcript preload failed", exc_info=True)
                summary: AgentRunSummary = agent.solve_with_summary(problem_statement)
                final_patch = summary.final_patch
                summary_dict = summary.to_dict()
                terminated_reason = summary.terminated_reason
                give_up_reason = summary.give_up_reason
                turn_count = len(summary.turns)
        except Exception as exc:  # noqa: BLE001 — per-task error isolation
            logger.exception("swe_evo: agent loop crashed for %s", task.instance_id)
            return SWEEvoTaskResult(
                instance_id=task.instance_id,
                success=False,
                error=f"agent_crash: {exc}\n{traceback.format_exc()[:2000]}",
                elapsed_seconds=time.time() - run_started,
                workspace_dir=workspace_dir,
            )

        return SWEEvoTaskResult(
            instance_id=task.instance_id,
            success=False,
            final_patch=final_patch,
            submission_ready=bool(final_patch),
            officially_evaluated=False,
            official_success=None,
            terminated_reason=terminated_reason,
            give_up_reason=give_up_reason,
            turn_count=turn_count,
            elapsed_seconds=time.time() - run_started,
            workspace_dir=workspace_dir,
            summary=summary_dict,
        )

    def _write_predictions(self, report: SWEEvoBenchmarkReport) -> None:
        """Write a SWE-agent shaped ``preds.json``.

        Format: ``{instance_id: {model_patch, model_name_or_path, instance_id}}``
        — matches what ``SWE-bench/evaluate_instance.py --scaffold SWE-agent``
        reads at scoring time.
        """
        preds: dict[str, dict[str, Any]] = {}
        for result in report.results:
            preds[result.instance_id] = result.to_swebench_prediction(
                model_name=report.model_name,
            )
        (self.output_dir / SWE_EVO_PREDS_FILENAME).write_text(
            json.dumps(preds, indent=2, sort_keys=True)
        )

    def _write_report(self, report: SWEEvoBenchmarkReport) -> None:
        (self.output_dir / SWE_EVO_REPORT_FILENAME).write_text(
            json.dumps(report.to_dict(), indent=2)
        )


__all__ = [
    "SWE_EVO_DATASET_NAME",
    "SWE_EVO_DATASET_SPLIT",
    "SWE_EVO_DEFAULT_INSTANCE_COUNT",
    "SWE_EVO_HARNESS_NAME",
    "SWE_EVO_HARNESS_VERSION",
    "SWE_EVO_PREDS_FILENAME",
    "SWE_EVO_REPORT_FILENAME",
    "SWE_EVO_REPORT_KIND_APEX",
    "SWEEvoBenchmarkReport",
    "SWEEvoHarness",
    "SWEEvoHarnessConfig",
    "SWEEvoPullRequest",
    "SWEEvoTask",
    "SWEEvoTaskResult",
    "TaskAgentRunner",
    "build_problem_statement",
    "load_swe_evo_tasks",
    "prepare_task_workspace",
]
