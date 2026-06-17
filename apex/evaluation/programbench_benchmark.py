"""ProgramBench benchmark driver.

ProgramBench (Yang et al., 2026 — facebookresearch/ProgramBench) asks an
LLM-based SWE-agent to *rebuild* a real-world program from scratch given only
its compiled binary and documentation. APEX's role per task is:

1. Materialize an empty workspace + the task's specification + its hidden test
   manifest.
2. Hand the workspace to ``ApexOrchestrator.solve(...)``; APEX edits the
   workspace in place (writing source files, build configs, etc.).
3. Pack the resulting workspace as ``submission.tar.gz`` so the upstream
   ``programbench eval`` CLI can score it inside its per-instance Docker
   container, then parse the produced ``<instance_id>.eval.json`` into a
   ``ProgramBenchScoreReport``.

Schema source: cloned ``https://github.com/facebookresearch/ProgramBench`` to
``/tmp/programbench_inspect`` on 2026-05-07 at commit
``1fe64c87a1318998636850cead00b0db80e2d928``. See
``tools/PROGRAMBENCH_NOTES.md`` for the inspection write-up.

This module mirrors ``apex.evaluation.commit0_benchmark`` in shape (per-task
dataclass + harness wrapper + ``ScoreReport``) but is intentionally far smaller
because ProgramBench's per-task compilation lives entirely inside the
upstream Docker image — APEX does not maintain its own per-language test
runners. We only own:

* dataset discovery (``ProgramBenchHarness.discover_tasks``),
* workspace materialization (``ProgramBenchHarness.prepare_workspace``),
* submission packaging (``ProgramBenchHarness.package_submission``),
* harness shell-out + report parsing (``ProgramBenchHarness.evaluate_solution``).

NOTE for operators: real evaluation requires Linux x86_64 (the upstream
images are ``linux/amd64`` only). The driver detects ``programbench`` on the
PATH and shells out; if it is missing, ``evaluate_solution`` returns a
``ScoreReport`` with ``solved=False`` and ``error_code='programbench_cli_missing'``
rather than raising — so a smoke run on macOS still produces a JSONL row that
the operator can inspect.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger("apex.programbench_benchmark")


# ---------------------------------------------------------------------------
# Constants captured from the upstream repo (see tools/PROGRAMBENCH_NOTES.md).
# ---------------------------------------------------------------------------

PROGRAMBENCH_HARNESS_NAME = "programbench_official_cli"
PROGRAMBENCH_HARNESS_VERSION = "2026-05-07.1"
PROGRAMBENCH_REPORT_KIND_APEX = "apex_programbench"
PROGRAMBENCH_DEFAULT_HF_DATASET = "programbench/ProgramBench-Tests"
PROGRAMBENCH_DEFAULT_DOCKER_ORG = "programbench"
PROGRAMBENCH_DEFAULT_IMAGE_TAG = "task_cleanroom"
PROGRAMBENCH_DEFAULT_SUBMISSION_BRANCH = "submission"
PROGRAMBENCH_DEFAULT_SUBMISSION_FILENAME = "submission.tar.gz"
PROGRAMBENCH_TASK_YAML_FILENAME = "task.yaml"
PROGRAMBENCH_TESTS_JSON_FILENAME = "tests.json"
PROGRAMBENCH_DEFAULT_EVAL_TIMEOUT_SECONDS = 1800
PROGRAMBENCH_KNOWN_LANGUAGES = (
    "c",
    "c++",
    "cpp",
    "go",
    "java",
    "javascript",
    "python",
    "rust",
    "typescript",
)


def image_name_from_instance_id(
    instance_id: str,
    *,
    docker_org: str = PROGRAMBENCH_DEFAULT_DOCKER_ORG,
) -> str:
    """Mirror upstream ``constants.image_name_from_instance_id``.

    The upstream replaces ``__`` with ``_1776_`` to keep the repo name a
    valid Docker Hub image path (Docker Hub disallows double underscores).
    """

    return f"{docker_org}/{instance_id.replace('__', '_1776_')}"


# ---------------------------------------------------------------------------
# Task dataclass.
# ---------------------------------------------------------------------------


@dataclass
class ProgramBenchTask:
    """One ProgramBench instance.

    Fields are derived from the upstream ``task.yaml`` + ``tests.json``
    (per ``programbench.utils.load_data._load_single_instance``):

    * ``instance_id``    — directory name (e.g. ``ffmpeg__ffmpeg.360a402``).
    * ``repository``     — upstream ``<owner>/<repo>`` slug.
    * ``commit``         — upstream sha the binary was built from.
    * ``language``       — primary language label (lowercase per upstream).
    * ``difficulty``     — upstream-provided difficulty bucket (``easy`` etc.).
    * ``image_name``     — Docker image (``programbench/<...>:task_cleanroom``).
    * ``image_tag``      — image tag operators can pin (default ``task_cleanroom``).
    * ``branches``       — branch_id → branch metadata (test ids, ignored flag).
    * ``spec_text``      — operator-supplied specification handed to the agent.
    * ``spec_path``      — optional source path the spec was loaded from.
    * ``hidden_test_dir``— optional path to a hidden tests bundle (for offline
      smoke runs that prepare the harness payload locally).
    * ``install_cmd``    — optional override for in-container install (rare).
    * ``run_cmd``        — optional override for in-container run.
    * ``eval_clean_hashes`` — known-good build artifact hashes (upstream).
    """

    instance_id: str
    repository: str
    commit: str
    language: str
    difficulty: str = ""
    image_name: str = ""
    image_tag: str = PROGRAMBENCH_DEFAULT_IMAGE_TAG
    branches: dict[str, dict[str, Any]] = field(default_factory=dict)
    spec_text: str = ""
    spec_path: Optional[str] = None
    spec_source: str = ""
    hidden_test_dir: Optional[str] = None
    install_cmd: Optional[str] = None
    run_cmd: Optional[str] = None
    eval_clean_hashes: list[str] = field(default_factory=list)

    @property
    def repo_name(self) -> str:
        return self.repository.split("/")[-1] if self.repository else self.instance_id

    @property
    def active_branches(self) -> list[str]:
        return [
            name
            for name, info in (self.branches or {}).items()
            if not (isinstance(info, dict) and info.get("ignored"))
        ]

    def total_active_tests(self) -> int:
        total = 0
        for name, info in (self.branches or {}).items():
            if not isinstance(info, dict) or info.get("ignored"):
                continue
            tests = info.get("tests") or []
            total += len(tests)
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "repository": self.repository,
            "commit": self.commit,
            "language": self.language,
            "difficulty": self.difficulty,
            "image_name": self.image_name,
            "image_tag": self.image_tag,
            "branches": dict(self.branches),
            "spec_text": self.spec_text,
            "spec_path": self.spec_path,
            "hidden_test_dir": self.hidden_test_dir,
            "install_cmd": self.install_cmd,
            "run_cmd": self.run_cmd,
            "eval_clean_hashes": list(self.eval_clean_hashes),
        }


# ---------------------------------------------------------------------------
# Score report.
# ---------------------------------------------------------------------------


@dataclass
class ProgramBenchScoreReport:
    """Structured score for one ProgramBench instance.

    ``program_id`` matches ``ProgramBenchTask.instance_id``. Counters are
    derived from the upstream ``test_results`` list with the same status
    semantics as ``programbench.eval.eval.TestResult``.
    """

    program_id: str
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    tests_errored: int = 0
    total_tests: int = 0
    solved: bool = False
    error_code: Optional[str] = None
    error_details: Optional[str] = None
    raw_output: str = ""
    eval_path: Optional[str] = None
    test_branch_errors: dict[str, str] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        runnable = self.tests_passed + self.tests_failed + self.tests_errored
        if runnable <= 0:
            return 0.0
        return self.tests_passed / runnable

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_skipped": self.tests_skipped,
            "tests_errored": self.tests_errored,
            "total_tests": self.total_tests,
            "solved": bool(self.solved),
            "pass_rate": self.pass_rate,
            "error_code": self.error_code,
            "error_details": self.error_details,
            "eval_path": self.eval_path,
            "test_branch_errors": dict(self.test_branch_errors),
            "raw_output_excerpt": self.raw_output[-2000:] if self.raw_output else "",
        }


# Aliases for callers preferring the generic naming from the spec.
ScoreReport = ProgramBenchScoreReport


# ---------------------------------------------------------------------------
# Per-program task overrides (mirrors commit0_task_overrides.json).
# ---------------------------------------------------------------------------


_TASK_OVERRIDES_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "programbench_task_overrides.json"
)


def _load_task_overrides(
    config_path: Path = _TASK_OVERRIDES_CONFIG_PATH,
) -> dict[str, dict[str, Any]]:
    """Load per-instance overrides; return empty when missing/malformed."""

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        logger.warning(
            "ProgramBench task overrides config missing at %s — running without overrides.",
            config_path,
        )
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "ProgramBench task overrides config at %s could not be loaded (%s).",
            config_path,
            exc,
        )
        return {}
    if not isinstance(raw, dict):
        return {}
    overrides: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if str(key).startswith("_"):
            continue  # informational keys (_doc etc.)
        if not isinstance(value, dict):
            continue
        cleaned = {k: v for k, v in value.items() if not str(k).startswith("_")}
        overrides[str(key)] = cleaned
    return overrides


# ---------------------------------------------------------------------------
# Harness.
# ---------------------------------------------------------------------------


class ProgramBenchHarness:
    """Glue between APEX workspaces and the upstream ``programbench`` CLI."""

    def __init__(
        self,
        *,
        cli_executable: str = "programbench",
        docker_org: str = PROGRAMBENCH_DEFAULT_DOCKER_ORG,
        image_tag: str = PROGRAMBENCH_DEFAULT_IMAGE_TAG,
        eval_timeout_seconds: int = PROGRAMBENCH_DEFAULT_EVAL_TIMEOUT_SECONDS,
        submission_branch: str = PROGRAMBENCH_DEFAULT_SUBMISSION_BRANCH,
        task_overrides: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        self.cli_executable = cli_executable
        self.docker_org = docker_org
        self.image_tag = image_tag
        self.eval_timeout_seconds = int(eval_timeout_seconds)
        self.submission_branch = submission_branch
        self.task_overrides = (
            dict(task_overrides) if task_overrides is not None else _load_task_overrides()
        )

    # -------- discovery ---------------------------------------------------

    def discover_tasks(
        self,
        dataset_name_or_path: str | Path,
        *,
        spec_lookup: Optional[dict[str, str]] = None,
        spec_dir: Optional[str | Path] = None,
        hidden_tests_dir: Optional[str | Path] = None,
    ) -> list[ProgramBenchTask]:
        """Discover ProgramBench tasks.

        ``dataset_name_or_path`` can be:

        * a path to a directory holding per-instance subdirs (each containing
          ``task.yaml`` + optional ``tests.json``) — typically the cloned
          ``src/programbench/data/tasks/`` directory or a subset thereof; or
        * a HuggingFace dataset id (e.g. ``programbench/ProgramBench-Tests``),
          loaded via the ``datasets`` package on demand.

        ``spec_lookup`` maps ``instance_id -> spec_text`` for operators who
        ship specs alongside the dataset; ``spec_dir`` is searched for
        ``<instance_id>/spec.md`` or ``<instance_id>.md`` when ``spec_lookup``
        does not cover an instance. ``hidden_tests_dir`` is the on-disk root
        for ``<instance_id>/`` test bundles (offline smoke / replay).
        """

        path = Path(str(dataset_name_or_path)).expanduser()
        if path.exists() and path.is_dir():
            tasks = self._discover_from_directory(path)
        else:
            tasks = self._discover_from_huggingface(str(dataset_name_or_path))
        return self._enrich_tasks(
            tasks,
            spec_lookup=spec_lookup,
            spec_dir=Path(spec_dir).expanduser() if spec_dir else None,
            hidden_tests_dir=Path(hidden_tests_dir).expanduser() if hidden_tests_dir else None,
        )

    def _discover_from_directory(self, root: Path) -> list[ProgramBenchTask]:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment-only
            raise SystemExit(
                "PyYAML is required to load ProgramBench tasks from a directory. "
                "Install with: pip install pyyaml"
            ) from exc

        tasks: list[ProgramBenchTask] = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            task_yaml = child / PROGRAMBENCH_TASK_YAML_FILENAME
            if not task_yaml.is_file():
                continue
            try:
                payload = yaml.safe_load(task_yaml.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                logger.warning(
                    "Skipping malformed task.yaml at %s: %s",
                    task_yaml,
                    exc,
                )
                continue
            tests_json = child / PROGRAMBENCH_TESTS_JSON_FILENAME
            branches: dict[str, Any] = {}
            if tests_json.is_file():
                try:
                    tests_payload = json.loads(tests_json.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("Skipping malformed tests.json at %s: %s", tests_json, exc)
                    tests_payload = {}
                if isinstance(tests_payload, dict):
                    raw_branches = tests_payload.get("branches") or {}
                    if isinstance(raw_branches, dict):
                        branches = {
                            str(k): dict(v) if isinstance(v, dict) else {}
                            for k, v in raw_branches.items()
                        }
            tasks.append(self._task_from_payload(child.name, payload, branches))
        return tasks

    def _discover_from_huggingface(self, dataset_name: str) -> list[ProgramBenchTask]:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - environment-only
            raise SystemExit(
                f"the 'datasets' package is required to load {dataset_name!r} "
                "from HuggingFace.\ninstall with: pip install datasets"
            ) from exc
        dataset = load_dataset(dataset_name, split="test")
        tasks: list[ProgramBenchTask] = []
        for row in dataset:
            row_dict = dict(row)
            instance_id = str(row_dict.get("instance_id") or "").strip()
            if not instance_id:
                continue
            branches = row_dict.get("branches") or {}
            if not isinstance(branches, dict):
                branches = {}
            tasks.append(self._task_from_payload(instance_id, row_dict, branches))
        return tasks

    def _task_from_payload(
        self,
        instance_id: str,
        payload: dict[str, Any],
        branches: dict[str, Any],
    ) -> ProgramBenchTask:
        repository = str(payload.get("repository") or payload.get("repo") or "")
        commit = str(payload.get("commit") or payload.get("base_commit") or "")
        language = str(payload.get("language") or "").strip().lower()
        difficulty = str(payload.get("difficulty") or "")
        image_tag = str(payload.get("image_tag") or self.image_tag) or self.image_tag
        eval_clean_hashes_raw = payload.get("eval_clean_hashes") or []
        eval_clean_hashes = [str(h) for h in eval_clean_hashes_raw if h]
        return ProgramBenchTask(
            instance_id=instance_id,
            repository=repository,
            commit=commit,
            language=language,
            difficulty=difficulty,
            image_name=image_name_from_instance_id(instance_id, docker_org=self.docker_org),
            image_tag=image_tag,
            branches={k: dict(v) if isinstance(v, dict) else {} for k, v in branches.items()},
            install_cmd=str(payload["install_cmd"]) if payload.get("install_cmd") else None,
            run_cmd=str(payload["run_cmd"]) if payload.get("run_cmd") else None,
            eval_clean_hashes=eval_clean_hashes,
        )

    def _enrich_tasks(
        self,
        tasks: Iterable[ProgramBenchTask],
        *,
        spec_lookup: Optional[dict[str, str]],
        spec_dir: Optional[Path],
        hidden_tests_dir: Optional[Path],
    ) -> list[ProgramBenchTask]:
        enriched: list[ProgramBenchTask] = []
        spec_lookup = spec_lookup or {}
        for task in tasks:
            spec_text = spec_lookup.get(task.instance_id, "")
            spec_path: Optional[str] = None
            spec_source = "lookup" if spec_text else ""
            if not spec_text and spec_dir is not None:
                for candidate in (
                    spec_dir / task.instance_id / "spec.md",
                    spec_dir / task.instance_id / "SPEC.md",
                    spec_dir / f"{task.instance_id}.md",
                ):
                    if candidate.is_file():
                        spec_text = candidate.read_text(encoding="utf-8")
                        spec_path = str(candidate)
                        spec_source = "file"
                        break
            if not spec_text:
                # Synthesize a defensible default so smoke runs work without
                # a curated spec corpus. Operators should override.
                spec_text = self._default_spec_text(task)
                spec_source = "synthetic"
            task.spec_text = spec_text
            task.spec_path = spec_path
            task.spec_source = spec_source
            if hidden_tests_dir is not None:
                candidate_dir = hidden_tests_dir / task.instance_id
                if candidate_dir.is_dir():
                    task.hidden_test_dir = str(candidate_dir)
            override = self.task_overrides.get(task.instance_id)
            if override:
                if override.get("install_cmd"):
                    task.install_cmd = str(override["install_cmd"])
                if override.get("run_cmd"):
                    task.run_cmd = str(override["run_cmd"])
                if override.get("image_tag"):
                    task.image_tag = str(override["image_tag"])
            enriched.append(task)
        return enriched

    @staticmethod
    def _default_spec_text(task: ProgramBenchTask) -> str:
        lines = [
            f"# ProgramBench task {task.instance_id}",
            "",
            (
                "You are rebuilding a real-world program from scratch. The upstream "
                "binary's behavior is the source of truth — you do not have access "
                "to the upstream source code."
            ),
            "",
            f"- Upstream repository: {task.repository or 'unknown'}",
            f"- Pinned commit: {task.commit or 'unknown'}",
            f"- Primary language: {task.language or 'unspecified'}",
        ]
        if task.difficulty:
            lines.append(f"- Difficulty bucket: {task.difficulty}")
        lines.extend(
            [
                "",
                "Goal: produce a working codebase whose hidden test suite passes.",
                "Tests live on git branches inside the upstream Docker image and "
                "will be run by the official `programbench eval` harness.",
                "",
                "Constraints:",
                "- No internet access during inference.",
                "- Match the upstream CLI flags, output formats, and exit codes.",
            ]
        )
        return "\n".join(lines) + "\n"

    # -------- workspace materialization ------------------------------------

    def prepare_workspace(
        self,
        task: ProgramBenchTask,
        workspace_root: str | Path,
        *,
        clean: bool = True,
    ) -> Path:
        """Materialize the per-task workspace tree handed to ``ApexOrchestrator.solve``.

        Layout::

            workspace_root/
              <instance_id>/
                spec.md                  # task.spec_text
                tests.json               # branches manifest (read-only ref)
                task.yaml                # raw task metadata snapshot
                src/                     # empty placeholder for the agent
                README.md                # short orientation note for the agent

        APEX is expected to write its source files under ``src/`` (or
        elsewhere as required by the language convention) — the harness
        tarballs the entire workspace minus ``tests.json``, ``task.yaml``,
        and ``spec.md`` files. Hidden test fixtures stay outside the
        agent-visible workspace.
        """

        workspace = Path(workspace_root).expanduser() / task.instance_id
        if workspace.exists() and clean:
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "src").mkdir(exist_ok=True)
        (workspace / "spec.md").write_text(task.spec_text or "", encoding="utf-8")
        (workspace / "tests.json").write_text(
            json.dumps({"branches": task.branches}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (workspace / "task.yaml").write_text(
            json.dumps(
                {
                    "instance_id": task.instance_id,
                    "repository": task.repository,
                    "commit": task.commit,
                    "language": task.language,
                    "difficulty": task.difficulty,
                    "image_name": task.image_name,
                    "image_tag": task.image_tag,
                    "eval_clean_hashes": list(task.eval_clean_hashes),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        readme = (
            f"# {task.instance_id}\n\n"
            "APEX agent workspace. Implement the program described in `spec.md`.\n"
            "Place source files under `src/` (or follow the language's idioms).\n"
            "The hidden test suite is invoked by the upstream `programbench eval`\n"
            "harness inside its Docker container — see `tests.json` for the\n"
            "branch/test inventory we expect to satisfy.\n"
        )
        (workspace / "README.md").write_text(readme, encoding="utf-8")
        return workspace

    # -------- submission packaging ----------------------------------------

    HARNESS_RESERVED_FILENAMES = frozenset({"spec.md", "tests.json", "task.yaml", "README.md"})

    HARNESS_RESERVED_DIRNAMES = frozenset({"hidden_tests"})

    def package_submission(
        self,
        task: ProgramBenchTask,
        solution_path: str | Path,
        *,
        run_dir: str | Path,
    ) -> Path:
        """Produce ``<run_dir>/<instance_id>/submission.tar.gz`` for ``programbench eval``.

        The tarball contains only the agent-authored files (``src/`` and any
        sibling files the agent wrote). Reserved harness scaffolding files
        (``spec.md``, ``tests.json``, ``task.yaml``, ``README.md``,
        ``hidden_tests/``) are excluded so they don't pollute the submission.
        """

        solution_root = Path(solution_path).expanduser()
        if not solution_root.is_dir():
            raise FileNotFoundError(
                f"solution path {solution_root!r} is not a directory; "
                "did the agent emit any source?"
            )
        instance_run_dir = Path(run_dir).expanduser() / task.instance_id
        instance_run_dir.mkdir(parents=True, exist_ok=True)
        tarball_path = instance_run_dir / PROGRAMBENCH_DEFAULT_SUBMISSION_FILENAME

        def _filter(info: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
            head = info.name.split("/", 1)[0] if "/" in info.name else info.name
            if head in self.HARNESS_RESERVED_DIRNAMES:
                return None
            if head in self.HARNESS_RESERVED_FILENAMES:
                return None
            return info

        with tarfile.open(tarball_path, "w:gz") as tar:
            for entry in sorted(solution_root.iterdir()):
                tar.add(entry, arcname=entry.name, filter=_filter)
        return tarball_path

    # -------- evaluation --------------------------------------------------

    def evaluate_solution(
        self,
        task: ProgramBenchTask,
        solution_path: str | Path,
        *,
        run_dir: Optional[str | Path] = None,
        eval_output_dir: Optional[str | Path] = None,
        env: Optional[dict[str, str]] = None,
    ) -> ProgramBenchScoreReport:
        """Score a candidate solution against ProgramBench's official harness.

        ``solution_path`` is the workspace returned by ``prepare_workspace``
        (after the agent has written source). When ``run_dir`` is omitted a
        fresh tempdir is used. When the upstream ``programbench`` CLI is not
        installed (the typical case on macOS hosts), the report is returned
        with ``error_code='programbench_cli_missing'`` and ``solved=False``
        so the surrounding pipeline keeps moving instead of crashing.
        """

        owns_tempdir = False
        if run_dir is None:
            run_root = Path(tempfile.mkdtemp(prefix="programbench_run_"))
            owns_tempdir = True
        else:
            run_root = Path(run_dir).expanduser()
            run_root.mkdir(parents=True, exist_ok=True)
        try:
            self.package_submission(task, solution_path, run_dir=run_root)
            cli_path = shutil.which(self.cli_executable)
            if cli_path is None:
                return ProgramBenchScoreReport(
                    program_id=task.instance_id,
                    total_tests=task.total_active_tests(),
                    error_code="programbench_cli_missing",
                    error_details=(
                        f"{self.cli_executable!r} not on PATH; install with "
                        "`uv pip install programbench` or `pip install programbench`."
                    ),
                    raw_output="",
                )
            output_arg: list[str] = []
            if eval_output_dir is not None:
                eval_root = Path(eval_output_dir).expanduser()
                eval_root.mkdir(parents=True, exist_ok=True)
                output_arg = ["--output", str(eval_root)]
            cmd = [
                cli_path,
                "eval",
                str(run_root),
                "--filter",
                f"^{task.instance_id}$",
                "--image-tag",
                task.image_tag,
                *output_arg,
            ]
            harness_env = dict(os.environ)
            if env:
                harness_env.update(env)
            try:
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.eval_timeout_seconds,
                    env=harness_env,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return ProgramBenchScoreReport(
                    program_id=task.instance_id,
                    total_tests=task.total_active_tests(),
                    error_code="harness_timeout",
                    error_details=f"timed out after {self.eval_timeout_seconds}s",
                    raw_output=str(exc.stdout or ""),
                )
            raw_output = "\n".join(filter(None, [completed.stdout, completed.stderr]))
            eval_root_for_results = (
                Path(str(eval_output_dir)).expanduser() / run_root.name
                if eval_output_dir is not None
                else run_root
            )
            eval_path = eval_root_for_results / task.instance_id / f"{task.instance_id}.eval.json"
            if not eval_path.is_file():
                # Fall back to the in-place run_root location used by the upstream
                # default mode (no --output arg).
                fallback_path = run_root / task.instance_id / f"{task.instance_id}.eval.json"
                if fallback_path.is_file():
                    eval_path = fallback_path
            if not eval_path.is_file():
                return ProgramBenchScoreReport(
                    program_id=task.instance_id,
                    total_tests=task.total_active_tests(),
                    error_code="harness_no_report",
                    error_details=(
                        f"programbench eval exited rc={completed.returncode} "
                        f"but produced no eval.json at {eval_path}"
                    ),
                    raw_output=raw_output,
                )
            return self.parse_eval_report(
                task,
                eval_path,
                raw_output=raw_output,
            )
        finally:
            if owns_tempdir:
                shutil.rmtree(run_root, ignore_errors=True)

    @staticmethod
    def parse_eval_report(
        task: ProgramBenchTask,
        eval_path: str | Path,
        *,
        raw_output: str = "",
    ) -> ProgramBenchScoreReport:
        """Translate one ``<iid>.eval.json`` payload into a ``ProgramBenchScoreReport``."""

        eval_path = Path(eval_path)
        try:
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return ProgramBenchScoreReport(
                program_id=task.instance_id,
                total_tests=task.total_active_tests(),
                error_code="harness_unreadable_report",
                error_details=f"{type(exc).__name__}: {exc}",
                raw_output=raw_output,
                eval_path=str(eval_path),
            )
        if not isinstance(payload, dict):
            return ProgramBenchScoreReport(
                program_id=task.instance_id,
                total_tests=task.total_active_tests(),
                error_code="harness_malformed_report",
                error_details="eval.json root is not a JSON object",
                raw_output=raw_output,
                eval_path=str(eval_path),
            )

        ignored_branch_test_keys: set[str] = set()
        for branch_name, info in (task.branches or {}).items():
            if not isinstance(info, dict):
                continue
            for entry in info.get("ignored_tests") or []:
                if isinstance(entry, dict) and entry.get("name"):
                    ignored_branch_test_keys.add(f"{branch_name}/{entry['name']}")

        passed = failed = errored = skipped = 0
        results = payload.get("test_results") or []
        for result in results:
            if not isinstance(result, dict):
                continue
            branch = str(result.get("branch") or "")
            name = str(result.get("name") or "")
            full_key = f"{branch}/{name}" if branch else name
            if full_key in ignored_branch_test_keys:
                continue
            status = str(result.get("status") or "").lower()
            if status == "passed":
                passed += 1
            elif status == "failure":
                failed += 1
            elif status in {"error", "system_error"}:
                errored += 1
            elif status == "skipped":
                skipped += 1
            elif status == "not_run":
                # Treat as failure-equivalent for solved-ness but track separately.
                errored += 1
            else:
                logger.warning(
                    "Unknown ProgramBench test status %r for %s/%s — counting as error.",
                    status,
                    task.instance_id,
                    name,
                )
                errored += 1

        total = passed + failed + errored + skipped
        report_error_code = payload.get("error_code")
        report_error_details = payload.get("error_details")
        branch_errors_raw = payload.get("test_branch_errors") or {}
        branch_errors: dict[str, str] = {}
        if isinstance(branch_errors_raw, dict):
            for branch, details in branch_errors_raw.items():
                if isinstance(details, list) and details:
                    first = details[0]
                    if isinstance(first, dict):
                        branch_errors[str(branch)] = str(first.get("error_code") or "")
                    else:
                        branch_errors[str(branch)] = str(first)
                elif isinstance(details, dict):
                    branch_errors[str(branch)] = str(details.get("error_code") or "")
                else:
                    branch_errors[str(branch)] = str(details)

        runnable = passed + failed + errored
        solved = bool(
            runnable > 0
            and failed == 0
            and errored == 0
            and not report_error_code
            and not branch_errors
        )

        return ProgramBenchScoreReport(
            program_id=task.instance_id,
            tests_passed=passed,
            tests_failed=failed,
            tests_skipped=skipped,
            tests_errored=errored,
            total_tests=total,
            solved=solved,
            error_code=str(report_error_code) if report_error_code else None,
            error_details=str(report_error_details) if report_error_details else None,
            raw_output=raw_output,
            eval_path=str(eval_path),
            test_branch_errors=branch_errors,
        )


# ---------------------------------------------------------------------------
# Convenience helpers used by the CLI runner + tests.
# ---------------------------------------------------------------------------


def load_tasks_from_directory(
    dataset_dir: str | Path,
    *,
    spec_dir: Optional[str | Path] = None,
    hidden_tests_dir: Optional[str | Path] = None,
) -> list[ProgramBenchTask]:
    """Convenience wrapper around ``ProgramBenchHarness.discover_tasks``."""

    harness = ProgramBenchHarness()
    return harness.discover_tasks(
        dataset_dir,
        spec_dir=spec_dir,
        hidden_tests_dir=hidden_tests_dir,
    )


def write_prediction_record(
    *,
    task: ProgramBenchTask,
    score: ProgramBenchScoreReport,
    apex_diagnostics: Optional[dict[str, Any]] = None,
    model_name: str = "apex",
) -> dict[str, Any]:
    """Build the JSONL prediction record the runner appends per task."""

    return {
        "instance_id": task.instance_id,
        "model_name_or_path": model_name,
        "language": task.language,
        "difficulty": task.difficulty,
        "image_name": task.image_name,
        "image_tag": task.image_tag,
        "score": score.to_dict(),
        "apex": dict(apex_diagnostics or {}),
        "harness": {
            "name": PROGRAMBENCH_HARNESS_NAME,
            "version": PROGRAMBENCH_HARNESS_VERSION,
        },
    }
