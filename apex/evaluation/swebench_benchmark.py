"""SWE-Bench family benchmark driver.

Covers SWE-Bench classic (`princeton-nlp/SWE-bench`),
SWE-Bench Verified (`princeton-nlp/SWE-bench_Verified`), and
SWE-Bench Multilingual (`SWE-bench/SWE-bench_Multilingual`) on a single
shared runner. The Pro variant has its own dedicated harness in
``apex/evaluation/swebench_pro_benchmark.py`` because the Pro container
shape and parser differ; this module talks to the public ``swebench``
PyPI package via its ``run_evaluation`` entrypoint.

Hygiene constraint: the dataset rows expose ``patch``, ``test_patch``,
``FAIL_TO_PASS``, ``PASS_TO_PASS`` by name. We reuse the scrubbing
constants ``_SWEBENCH_HIDDEN_TEXT_FIELDS`` and
``_SWEBENCH_HIDDEN_LIST_FIELD_COUNTS`` from
``swebench_pro_benchmark`` verbatim so the agent-facing payload can never
accidentally contain a gold patch.

The runner does not vendor the swebench harness — operators install
``pip install swebench`` (or ``pip install swebench[multilingual]``) and
the runner shells ``python -m swebench.harness.run_evaluation`` against
the predictions JSONL we produce.
"""

from __future__ import annotations

import ast
import json
import logging
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .swebench_pro_benchmark import (
    _SWEBENCH_HIDDEN_LIST_FIELD_COUNTS,
    _SWEBENCH_HIDDEN_TEXT_FIELDS,
    _artifact_safe_swebench_payload,
    _scrub_swebench_published_parity_artifact_payload,
)

logger = logging.getLogger("apex.evaluation.swebench")


# Public dataset names we know how to drive on this harness shape.
SWEBENCH_CLASSIC_DATASET_NAME = "princeton-nlp/SWE-bench"
SWEBENCH_VERIFIED_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
SWEBENCH_MULTILINGUAL_DATASET_NAME = "SWE-bench/SWE-bench_Multilingual"
SWEBENCH_DEFAULT_SPLIT = "test"

# Harness modes select which harness binary the runner shells. The Pro
# mode is intentionally allowed for parity with the Pro codegen wrapper,
# but actually invokes ``swebench_pro_benchmark`` rather than this
# module's classic harness invocation.
SWEBENCH_HARNESS_MODE_CLASSIC = "classic"
SWEBENCH_HARNESS_MODE_PRO = "pro"
SWEBENCH_HARNESS_MODE_MULTILINGUAL = "multilingual"
SWEBENCH_HARNESS_MODES = (
    SWEBENCH_HARNESS_MODE_CLASSIC,
    SWEBENCH_HARNESS_MODE_PRO,
    SWEBENCH_HARNESS_MODE_MULTILINGUAL,
)

# Default container namespace the swebench package uses when it builds
# per-instance images. Operators rarely override this; the constant is
# here so test code and configs can refer to it.
SWEBENCH_CLASSIC_IMAGE_NAMESPACE = "swebench/sweb.eval.x86_64"
SWEBENCH_HARNESS_INSTALL_HINT = (
    "Install the official harness with `pip install swebench` (or "
    "`pip install swebench[multilingual]` for the multilingual variant) "
    "and re-run."
)


def _parse_literal_list(value: Any) -> list[str]:
    """Parse a HuggingFace SWE-Bench list-of-strings field.

    SWE-Bench rows store FAIL_TO_PASS / PASS_TO_PASS either as a JSON or
    Python literal string list, or already as a Python list when the row
    has been deserialized previously. Tolerate both — and lone strings.
    """

    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                # The row gave us a single id as a bare string.
                return [text]
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed if item is not None]
        return [str(parsed)]
    return [str(value)]


def _summarize_test_patch(test_patch: str, *, max_chars: int = 240) -> str:
    """Summarize the gold ``test_patch`` without leaking its contents.

    The summary records only the changed file paths and a per-file
    add/remove line count so operator-facing artifacts can show that a
    test_patch was present without exposing the gold tests to any agent
    prompt or downstream prediction record.
    """

    text = str(test_patch or "").strip()
    if not text:
        return ""
    files: list[str] = []
    adds = 0
    removes = 0
    for line in text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            # diff --git a/path b/path
            if len(parts) >= 4:
                rel = parts[-1]
                if rel.startswith("b/"):
                    rel = rel[2:]
                if rel and rel not in files:
                    files.append(rel)
        elif line.startswith("+") and not line.startswith("+++"):
            adds += 1
        elif line.startswith("-") and not line.startswith("---"):
            removes += 1
    summary = (
        f"{len(files)} files modified ({adds} added, {removes} removed)"
        if files
        else "test_patch present (file paths not parsed)"
    )
    files_preview = ", ".join(files[:10])
    if files_preview:
        summary = f"{summary}; files: {files_preview}"
        if len(files) > 10:
            summary = f"{summary} +{len(files) - 10} more"
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3] + "..."
    return summary


@dataclass
class SWEBenchTask:
    """One SWE-Bench / SWE-Bench Verified / Multilingual task.

    Mirrors :class:`SWEBenchProTask` in shape but holds only the fields
    that the public ``swebench`` harness needs. Gold ``patch`` and
    ``test_patch`` are intentionally NOT stored on this dataclass —
    the runner discards them at row-load time so the orchestrator can
    never accidentally see them. Only an opaque
    :attr:`scrubbed_test_patch_summary` (file count + add/remove counts,
    no diff text) is preserved for diagnostics.
    """

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str = ""
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    environment_setup_commit: str = ""
    version: str = ""
    difficulty: str = ""
    repo_language: str = ""
    scrubbed_test_patch_summary: str = ""

    @property
    def repo_name(self) -> str:
        return self.repo.split("/")[-1] if self.repo else ""

    @property
    def required_tests(self) -> list[str]:
        return sorted(set(self.fail_to_pass) | set(self.pass_to_pass))

    def build_issue_description(self) -> str:
        """Build a benchmark-clean issue description for the orchestrator.

        Includes the repository name and the problem statement, and
        nothing from the gold patch or gold tests. Hints are included
        only because they live in the row schema and are visible to the
        published baseline agents (parity with Verified / classic).
        """

        lines = [
            "Resolve the repository issue by changing application code.",
            "Do not modify benchmark-controlled tests or benchmark harness files.",
            "",
            f"Repository: {self.repo}",
        ]
        if self.repo_language:
            lines.append(f"Repository language: {self.repo_language}")
        lines.extend(["", "Problem statement:", self.problem_statement.strip()])
        hints = (self.hints_text or "").strip()
        if hints:
            lines.extend(["", "Hints from the original report:", hints])
        return "\n".join(lines)

    def to_dict(self, *, include_benchmark_metadata: bool = False) -> dict[str, Any]:
        """Serialize for diagnostics. Defaults to scrubbed published-parity."""

        payload: dict[str, Any] = {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "hints_text": self.hints_text,
            "fail_to_pass": list(self.fail_to_pass),
            "pass_to_pass": list(self.pass_to_pass),
            "environment_setup_commit": self.environment_setup_commit,
            "version": self.version,
            "difficulty": self.difficulty,
            "repo_language": self.repo_language,
            "scrubbed_test_patch_summary": self.scrubbed_test_patch_summary,
        }
        if include_benchmark_metadata:
            return payload
        return _scrub_swebench_published_parity_artifact_payload(payload)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "SWEBenchTask":
        """Build from a HuggingFace dataset row.

        Importantly, this method NEVER stores the row's ``patch`` or
        ``test_patch`` on the resulting task. Both fields are
        deliberately discarded after extracting an opaque summary of
        ``test_patch`` for diagnostics. ``FAIL_TO_PASS`` /
        ``PASS_TO_PASS`` are extracted as plain lists for the harness
        invocation but the agent-facing :meth:`to_dict` strips them
        again in published-parity mode via the shared scrubber.
        """

        # SWE-Bench dataset uses both lower and upper case key names
        # depending on the variant; the Multilingual schema uses
        # FAIL_TO_PASS / PASS_TO_PASS while the Verified schema accepts
        # both. Read either.
        fail_to_pass = _parse_literal_list(row.get("FAIL_TO_PASS", row.get("fail_to_pass")))
        pass_to_pass = _parse_literal_list(row.get("PASS_TO_PASS", row.get("pass_to_pass")))
        # Discard patch / test_patch immediately. Only keep an opaque
        # summary of the test patch so artifacts can show that a
        # test_patch was present without leaking its contents.
        test_patch_text = str(row.get("test_patch") or "")
        test_patch_summary = _summarize_test_patch(test_patch_text)
        return cls(
            instance_id=str(row["instance_id"]),
            repo=str(row.get("repo", "") or ""),
            base_commit=str(row.get("base_commit", "") or ""),
            problem_statement=str(row.get("problem_statement", "") or ""),
            hints_text=str(row.get("hints_text", "") or ""),
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            environment_setup_commit=str(row.get("environment_setup_commit", "") or ""),
            version=str(row.get("version", "") or ""),
            difficulty=str(row.get("difficulty", "") or ""),
            repo_language=str(row.get("repo_language", row.get("language", "") or "") or ""),
            scrubbed_test_patch_summary=test_patch_summary,
        )


@dataclass
class ScoreReport:
    """Parsed result of a single instance's swebench harness ``report.json``.

    The classic / Verified / Multilingual harnesses emit a per-instance
    file at ``<run_id>/<model>/<instance_id>/report.json`` whose top
    level looks like::

        {
          "<instance_id>": {
            "patch_is_None": false,
            "patch_exists": true,
            "patch_successfully_applied": true,
            "resolved": true,
            "tests_status": {
              "FAIL_TO_PASS": {"success": [...], "failure": [...]},
              "PASS_TO_PASS": {"success": [...], "failure": [...]}
            }
          }
        }
    """

    instance_id: str
    tests_passed: int = 0
    tests_failed: int = 0
    fail_to_pass_passed: int = 0
    fail_to_pass_failed: int = 0
    pass_to_pass_preserved: int = 0
    pass_to_pass_broken: int = 0
    patch_applied: bool = False
    patch_exists: bool = False
    solved: bool = False
    raw_report_payload: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "fail_to_pass_passed": self.fail_to_pass_passed,
            "fail_to_pass_failed": self.fail_to_pass_failed,
            "pass_to_pass_preserved": self.pass_to_pass_preserved,
            "pass_to_pass_broken": self.pass_to_pass_broken,
            "patch_applied": self.patch_applied,
            "patch_exists": self.patch_exists,
            "solved": self.solved,
            "raw_report_payload": self.raw_report_payload,
            "error": self.error,
        }

    @classmethod
    def from_report_payload(
        cls,
        instance_id: str,
        payload: dict[str, Any],
    ) -> "ScoreReport":
        """Convert an instance-keyed report.json payload to a ScoreReport.

        The harness writes a top-level dict keyed by instance_id; we
        accept either the wrapped form or the inner per-instance dict.
        """

        if not isinstance(payload, dict):
            return cls(
                instance_id=instance_id,
                error=f"unparseable report payload type: {type(payload).__name__}",
            )
        # Unwrap if the top-level key is the instance_id.
        if instance_id in payload and isinstance(payload[instance_id], dict):
            inner = payload[instance_id]
        elif len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
            inner = next(iter(payload.values()))
        else:
            inner = payload
        tests_status = inner.get("tests_status") or {}
        f2p = tests_status.get("FAIL_TO_PASS") or {}
        p2p = tests_status.get("PASS_TO_PASS") or {}
        f2p_success = list(f2p.get("success") or [])
        f2p_failure = list(f2p.get("failure") or [])
        p2p_success = list(p2p.get("success") or [])
        p2p_failure = list(p2p.get("failure") or [])
        return cls(
            instance_id=instance_id,
            tests_passed=len(f2p_success) + len(p2p_success),
            tests_failed=len(f2p_failure) + len(p2p_failure),
            fail_to_pass_passed=len(f2p_success),
            fail_to_pass_failed=len(f2p_failure),
            pass_to_pass_preserved=len(p2p_success),
            pass_to_pass_broken=len(p2p_failure),
            patch_applied=bool(inner.get("patch_successfully_applied", False)),
            patch_exists=bool(inner.get("patch_exists", False)),
            solved=bool(inner.get("resolved", False)),
            raw_report_payload=dict(inner),
        )

    @classmethod
    def from_report_file(
        cls,
        instance_id: str,
        report_path: Path,
    ) -> "ScoreReport":
        if not report_path.exists():
            return cls(
                instance_id=instance_id,
                error=f"report.json not found at {report_path}",
            )
        try:
            payload = json.loads(report_path.read_text())
        except (OSError, ValueError) as exc:
            return cls(
                instance_id=instance_id,
                error=f"failed to parse report.json: {type(exc).__name__}: {exc}",
            )
        return cls.from_report_payload(instance_id, payload)


@dataclass
class SWEBenchPredictionRecord:
    """One JSONL row written to the predictions file.

    The official ``swebench`` harness expects each row to have at minimum
    ``instance_id``, ``model_name_or_path``, and ``model_patch``. All
    other keys are accepted but ignored.
    """

    instance_id: str
    model_name_or_path: str
    model_patch: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "model_name_or_path": self.model_name_or_path,
            "model_patch": self.model_patch or "",
        }


class SWEBenchHarness:
    """Driver for the public ``swebench`` harness.

    Owns dataset discovery, workspace preparation (handing the actual
    repo materialization off to the harness's docker images), and
    invocation of ``python -m swebench.harness.run_evaluation``. The
    workflow is:

    1. ``discover_tasks(dataset_name, split)`` -> ``list[SWEBenchTask]``
    2. (caller) generate a patch per task and write a JSONL preds file
    3. ``run_evaluation(preds_path, run_id, log_dir, ...)`` -> shells
       the harness once across all rows
    4. ``parse_report(task, run_id, log_dir, model_name)`` ->
       ``ScoreReport`` per task
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        dataset_name: str = SWEBENCH_VERIFIED_DATASET_NAME,
        split: str = SWEBENCH_DEFAULT_SPLIT,
        harness_mode: str = SWEBENCH_HARNESS_MODE_CLASSIC,
        max_workers: int = 4,
        cache_level: str = "instance",
        force_rebuild: bool = False,
        image_namespace: str = SWEBENCH_CLASSIC_IMAGE_NAMESPACE,
        timeout_seconds: float = 1800.0,
        python_executable: Optional[str] = None,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        self.split = split
        if harness_mode not in SWEBENCH_HARNESS_MODES:
            raise ValueError(
                f"Unknown harness_mode={harness_mode!r}. Expected one of {SWEBENCH_HARNESS_MODES}."
            )
        self.harness_mode = harness_mode
        self.max_workers = max(1, int(max_workers))
        self.cache_level = cache_level
        self.force_rebuild = bool(force_rebuild)
        self.image_namespace = image_namespace
        self.timeout_seconds = float(timeout_seconds)
        # We let the operator override the python interpreter that runs
        # the harness; this matters when the harness package is installed
        # in a separate venv from APEX.
        self.python_executable = python_executable or "python"

    # ------------------------------------------------------------------
    # Dataset discovery
    # ------------------------------------------------------------------

    def discover_tasks(
        self,
        *,
        instances: Optional[list[str]] = None,
        repos: Optional[list[str]] = None,
        languages: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> list[SWEBenchTask]:
        """Load tasks from HuggingFace and filter by the optional caller args."""

        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - environment-only
            raise SystemExit(
                "the 'datasets' package is required to load SWE-Bench from HF.\n"
                "install with: pip install datasets"
            ) from exc

        dataset = load_dataset(self.dataset_name, split=self.split)
        allowed_instances = {item for item in (instances or [])}
        allowed_repos = {item for item in (repos or [])}
        allowed_repo_suffixes = {item.split("/")[-1] for item in (repos or [])}
        allowed_languages = {item.lower() for item in (languages or [])}

        tasks: list[SWEBenchTask] = []
        for row in dataset:
            row_dict = dict(row)
            instance_id = str(row_dict.get("instance_id") or "")
            if not instance_id:
                continue
            if allowed_instances and instance_id not in allowed_instances:
                continue
            repo_name = str(row_dict.get("repo") or "")
            if (
                allowed_repos
                and repo_name not in allowed_repos
                and repo_name.split("/")[-1] not in allowed_repo_suffixes
            ):
                continue
            language = str(
                row_dict.get("repo_language", row_dict.get("language") or "") or ""
            ).lower()
            if allowed_languages and language not in allowed_languages:
                continue
            tasks.append(SWEBenchTask.from_row(row_dict))
            if limit is not None and len(tasks) >= int(limit):
                break
        logger.info(
            "discovered %d SWE-Bench tasks from %s (split=%s, mode=%s)",
            len(tasks),
            self.dataset_name,
            self.split,
            self.harness_mode,
        )
        return tasks

    # ------------------------------------------------------------------
    # Workspace prep
    # ------------------------------------------------------------------

    def target_image_uri(self, task: SWEBenchTask) -> str:
        return f"{self.image_namespace}.{task.instance_id}:latest"

    def prepare_workspace(self, task: SWEBenchTask) -> Path:
        """Materialize the task repository at the benchmark base commit.

        The official harness still owns final scoring, but Apex codegen agents
        need to edit and inspect the same source tree their patch will apply
        to. Dynamic tool execution is routed separately into the benchmark
        image by the codegen runner.
        """

        workspace = self.output_dir / "workspaces" / task.instance_id
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)
        if not task.repo:
            raise RuntimeError(f"SWE-Bench task {task.instance_id} is missing repo.")
        if not task.base_commit:
            raise RuntimeError(f"SWE-Bench task {task.instance_id} is missing base_commit.")
        workspace.parent.mkdir(parents=True, exist_ok=True)
        repo_url = (
            task.repo
            if task.repo.startswith(("http://", "https://"))
            else (f"https://github.com/{task.repo}.git")
        )
        clone = subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", repo_url, str(workspace)],
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        if clone.returncode != 0:
            raise RuntimeError(
                f"git clone failed for {task.repo}: {(clone.stderr or clone.stdout or '').strip()}"
            )
        fetch = subprocess.run(
            ["git", "fetch", "--depth", "1", "origin", task.base_commit],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        if fetch.returncode != 0:
            raise RuntimeError(
                f"git fetch failed for {task.instance_id}@{task.base_commit}: "
                f"{(fetch.stderr or fetch.stdout or '').strip()}"
            )
        checkout = subprocess.run(
            ["git", "checkout", "--force", task.base_commit],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if checkout.returncode != 0:
            raise RuntimeError(
                f"git checkout failed for {task.instance_id}@{task.base_commit}: "
                f"{(checkout.stderr or checkout.stdout or '').strip()}"
            )
        return workspace

    def write_predictions_file(
        self,
        path: str | Path,
        records: Iterable[SWEBenchPredictionRecord | dict[str, Any]],
    ) -> Path:
        """Atomically write a JSONL predictions file in the harness shape."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for record in records:
            if isinstance(record, SWEBenchPredictionRecord):
                payload = record.to_dict()
            else:
                payload = dict(record)
            # Defense-in-depth: never let a stray ``patch`` field from
            # diagnostics leak into the predictions file. The harness
            # only reads ``model_patch``; ``patch`` here would only be
            # a gold-patch leak.
            for key in _SWEBENCH_HIDDEN_TEXT_FIELDS:
                if key in payload and key not in {"model_patch"}:
                    # ``model_patch`` is the agent-produced patch and is
                    # required by the harness; drop only the gold-name
                    # collisions.
                    if key == "patch":
                        payload.pop(key, None)
            lines.append(json.dumps(payload, sort_keys=True))
        destination.write_text("\n".join(lines) + ("\n" if lines else ""))
        return destination

    # ------------------------------------------------------------------
    # Harness invocation
    # ------------------------------------------------------------------

    def build_run_evaluation_command(
        self,
        *,
        predictions_path: str | Path,
        run_id: str,
        log_dir: Optional[str | Path] = None,
        instance_ids: Optional[list[str]] = None,
        report_dir: Optional[str | Path] = None,
        extra_args: Optional[list[str]] = None,
    ) -> list[str]:
        """Build the argv for ``python -m swebench.harness.run_evaluation``.

        Operators install the harness via ``pip install swebench``. The
        Multilingual variant requires the ``[multilingual]`` extra.
        """

        command: list[str] = [
            self.python_executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            self.dataset_name,
            "--split",
            self.split,
            "--predictions_path",
            str(predictions_path),
            "--max_workers",
            str(self.max_workers),
            "--run_id",
            str(run_id),
            "--cache_level",
            self.cache_level,
        ]
        if self.force_rebuild:
            command.extend(["--force_rebuild", "True"])
        if instance_ids:
            command.append("--instance_ids")
            command.extend(str(i) for i in instance_ids)
        if log_dir is not None:
            # Newer swebench package supports --log_dir for routing
            # per-instance logs; older versions ignore unknown args.
            command.extend(["--log_dir", str(log_dir)])
        if report_dir is not None:
            command.extend(["--report_dir", str(report_dir)])
        if extra_args:
            command.extend(str(arg) for arg in extra_args)
        return command

    def run_evaluation(
        self,
        *,
        predictions_path: str | Path,
        run_id: str,
        log_dir: Optional[str | Path] = None,
        instance_ids: Optional[list[str]] = None,
        report_dir: Optional[str | Path] = None,
        extra_args: Optional[list[str]] = None,
    ) -> subprocess.CompletedProcess[str]:
        """Shell the harness once across the predictions file.

        Raises ``SystemExit`` with an actionable install hint if the
        ``swebench`` package isn't importable from
        :attr:`python_executable`.
        """

        # Probe up front so the operator gets a clear, actionable error
        # rather than a 100-line subprocess dump.
        probe = subprocess.run(
            [self.python_executable, "-c", "import swebench"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if probe.returncode != 0:
            raise SystemExit(
                "the 'swebench' Python package is not installed in "
                f"{self.python_executable}. {SWEBENCH_HARNESS_INSTALL_HINT}\n"
                f"Probe stderr: {probe.stderr.strip()}"
            )
        command = self.build_run_evaluation_command(
            predictions_path=predictions_path,
            run_id=run_id,
            log_dir=log_dir,
            instance_ids=instance_ids,
            report_dir=report_dir,
            extra_args=extra_args,
        )
        logger.info(
            "invoking swebench harness: %s",
            " ".join(shlex.quote(part) for part in command),
        )
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=self.timeout_seconds,
        )

    # ------------------------------------------------------------------
    # Report parsing
    # ------------------------------------------------------------------

    def report_path_for_task(
        self,
        task: SWEBenchTask,
        *,
        run_id: str,
        log_dir: str | Path,
        model_name: str,
    ) -> Path:
        """Compute the per-instance ``report.json`` location.

        The default harness layout is::

            <log_dir>/<run_id>/<model>/<instance_id>/report.json

        Operators who pass a custom ``--report_dir`` or pre-2024 layouts
        can override this by handing the path directly to
        :meth:`parse_report_for_task_at`.
        """

        return Path(log_dir) / run_id / model_name / task.instance_id / "report.json"

    def parse_report_for_task(
        self,
        task: SWEBenchTask,
        *,
        run_id: str,
        log_dir: str | Path,
        model_name: str,
    ) -> ScoreReport:
        return self.parse_report_for_task_at(
            task,
            self.report_path_for_task(task, run_id=run_id, log_dir=log_dir, model_name=model_name),
        )

    def parse_report_for_task_at(
        self,
        task: SWEBenchTask,
        report_path: str | Path,
    ) -> ScoreReport:
        return ScoreReport.from_report_file(task.instance_id, Path(report_path))

    def evaluate_patch(
        self,
        task: SWEBenchTask,
        patch: str,
        *,
        run_id: str,
        log_dir: str | Path,
        model_name: str,
        predictions_path: Optional[str | Path] = None,
    ) -> ScoreReport:
        """Single-task convenience: write preds, shell harness, parse report.

        Most callers should use the loop in
        :func:`apex.evaluation.swebench_codegen_eval.run_codegen_eval`
        instead, which batches all rows into one harness invocation
        rather than paying the docker image-build overhead per task.
        This method exists for the smoke-validation path and for the
        rare ablation that wants to score one task at a time.
        """

        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        if predictions_path is None:
            with tempfile.NamedTemporaryFile(
                mode="w",
                prefix=f"swebench-preds-{task.instance_id}-",
                suffix=".jsonl",
                dir=str(self.output_dir),
                delete=False,
            ) as fh:
                preds_path: Path = Path(fh.name)
        else:
            preds_path = Path(predictions_path)
        self.write_predictions_file(
            preds_path,
            [
                SWEBenchPredictionRecord(
                    instance_id=task.instance_id,
                    model_name_or_path=model_name,
                    model_patch=patch or "",
                )
            ],
        )
        completed = self.run_evaluation(
            predictions_path=preds_path,
            run_id=run_id,
            log_dir=log_dir_path,
            instance_ids=[task.instance_id],
        )
        report = self.parse_report_for_task(
            task,
            run_id=run_id,
            log_dir=log_dir_path,
            model_name=model_name,
        )
        if not report.raw_report_payload and not report.error:
            report.error = (
                f"swebench harness exited with code {completed.returncode}; "
                f"no report.json was produced. "
                f"stderr tail: {(completed.stderr or '')[-512:]}"
            )
        return report

    # ------------------------------------------------------------------
    # Artifact-safe helpers (delegate to Pro module so the constants stay
    # canonical)
    # ------------------------------------------------------------------

    def artifact_safe_payload(self, payload: Any) -> Any:
        """Scrub a diagnostic payload of any benchmark-private fields.

        Thin re-export of the Pro module's helper so callers don't need
        to reach across modules. Always uses ``include_benchmark_metadata=False``
        because this driver runs in published-parity mode by design.
        """

        return _artifact_safe_swebench_payload(payload, include_benchmark_metadata=False)


__all__ = [
    "SWEBENCH_CLASSIC_DATASET_NAME",
    "SWEBENCH_VERIFIED_DATASET_NAME",
    "SWEBENCH_MULTILINGUAL_DATASET_NAME",
    "SWEBENCH_DEFAULT_SPLIT",
    "SWEBENCH_HARNESS_MODE_CLASSIC",
    "SWEBENCH_HARNESS_MODE_PRO",
    "SWEBENCH_HARNESS_MODE_MULTILINGUAL",
    "SWEBENCH_HARNESS_MODES",
    "SWEBENCH_CLASSIC_IMAGE_NAMESPACE",
    "SWEBENCH_HARNESS_INSTALL_HINT",
    "SWEBenchTask",
    "SWEBenchPredictionRecord",
    "ScoreReport",
    "SWEBenchHarness",
    "_summarize_test_patch",
    "_parse_literal_list",
    # Re-export so callers don't have to know the constants live in the
    # Pro module:
    "_SWEBENCH_HIDDEN_TEXT_FIELDS",
    "_SWEBENCH_HIDDEN_LIST_FIELD_COUNTS",
]
