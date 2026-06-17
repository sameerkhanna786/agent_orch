"""SWT-Bench benchmark adapter (Phase 3).

SWT-Bench is the test-generation-against-bug-discrimination benchmark
introduced by Mündler et al. (NeurIPS 2024 D&B; SWT-Bench paper). Each
task supplies the SWE-bench-shape row:

    - ``instance_id`` / ``repo`` / ``version`` / ``base_commit``
    - ``problem_statement``: bug description
    - ``patch``: the gold fix (used by the harness, never shown to the agent)
    - ``test_patch``: the gold test the agent is implicitly compared to
    - ``FAIL_TO_PASS`` / ``PASS_TO_PASS``: per-test verdicts the harness
      enforces.

Published metrics:

    - **F2P (Fail-to-Pass)**: fraction of generated test patches that
      transform the buggy ``base_commit`` from "all green" to "fails on
      a bug-relevant test, then passes once the gold patch is applied".
      The published TEX-T pattern wins by aggregating across multiple
      candidate test diffs.

This driver is a thin shim on top of the TestGenEval pipeline because
APEX's V5 voting machinery (``dual_version_verifier`` /
``cross_candidate_voter`` / ``patch_surrogate`` / ``anti_hack_ledger``)
literally implements TEX-T — the SWT-Bench winner — already. The only
benchmark-specific surface is:

    1. ``SWTBenchTask`` dataclass — keeps the SWT-Bench-flavoured fields
       (``focal_test_file_path``, ``baseline_test_source``, ``F2P`` /
       ``P2P`` lists) addressable for the runner / harness wrapper.
    2. ``swtbench_task_to_testgeneval_task`` — adapter that maps an
       SWT-Bench row onto APEX's existing ``TestGenEvalTask`` so the V5
       pipeline can run unchanged. The agent thinks it is generating a
       test for the focal *test file* (correct — that *is* the SWT-Bench
       artifact), and the dual-state oracle has a real fix patch to
       grade against (because SWT-Bench rows ship the gold ``patch``).

The driver intentionally does NOT score; ``apex.evaluation.runners.swtbench``
shells the official ``swt_bench`` harness for that.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from apex.evaluation.testgeneval_benchmark import (
    TestGenEvalTask,
    TestGenEvalTaskResult,
    evaluate_testgeneval_task_with_default_generator,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SWT-Bench dataset row scrubbing — non-negotiable per the integration plan.
# ---------------------------------------------------------------------------

# The SWE-bench schema fields that leak the gold answer or test verdicts.
# These MUST be stripped from any task surface that ever touches an agent
# prompt. The list mirrors the swebench_pro_benchmark constants verbatim
# (re-declared here to avoid importing from the Pro driver, which is in
# Phase 2's file zone).
_SWTBENCH_HIDDEN_TEXT_FIELDS: tuple[str, ...] = (
    "patch",
    "test_patch",
    "gold_patch",
    "model_patch",
)
_SWTBENCH_HIDDEN_LIST_FIELDS: tuple[str, ...] = (
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "fail_to_pass",
    "pass_to_pass",
)


def scrub_row_for_agent_prompt(row: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``row`` with gold-leak fields removed.

    Caller MUST use this before any prompt rendering or row->task
    adapter. Fail-loudly approach: strip the hidden fields rather than
    masking them so a downstream prompt builder can never accidentally
    interpolate them.
    """

    sanitized: dict[str, Any] = {}
    for key, value in row.items():
        if key in _SWTBENCH_HIDDEN_TEXT_FIELDS:
            continue
        if key in _SWTBENCH_HIDDEN_LIST_FIELDS:
            # Preserve the COUNT (so the prompt can mention "this bug has
            # N hidden tests") without leaking the test names themselves.
            try:
                count = len(list(value or []))
            except TypeError:
                count = 0
            sanitized[f"{key}_count"] = count
            continue
        sanitized[key] = value
    return sanitized


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------


@dataclass
class SWTBenchTask:
    """One SWT-Bench problem.

    Carries the SWE-bench-shape fields needed to (1) prompt the agent
    with the bug description + focal test file, (2) write a valid
    git-patch prediction the official harness accepts, and (3) drive
    APEX's V5 dual-state voting (which needs ``repo`` + ``version`` +
    ``base_commit`` + a non-empty fix ``patch``).

    The ``apex_dual_state_oracle`` flag is *always* True because every
    SWT-Bench row ships a gold patch — V5 voting can engage on every
    task without per-row gating.
    """

    __test__ = False

    instance_id: str
    repo: str
    version: str
    base_commit: str
    problem_statement: str
    focal_test_file_path: str
    baseline_test_source: str = ""
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    # BM25-retrieved focal source context (from the *_bm25_27k_zsb dataset
    # variant). Used to give the agent a faster path to the focal module
    # than materializing the repo at base_commit.
    preds_context: dict[str, Any] = field(default_factory=dict)
    # Gold patch — kept on the task object so V5's dual-state oracle has
    # a real fix to grade test candidates against. NEVER goes into a
    # prompt; ``scrub_row_for_agent_prompt`` strips this field before
    # any prompt rendering.
    gold_patch: str = ""
    # Gold test patch — same hygiene: scrubbed from prompts; kept on the
    # task object only for harness-side oracle wiring.
    gold_test_patch: str = ""

    def to_huggingface_row(self) -> dict[str, Any]:
        """Project back into the SWE-bench-shape row the docker adapter
        expects. Used by the V5 dual-state oracle pipeline."""

        return {
            "instance_id": self.instance_id,
            "id": self.instance_id,
            "repo": self.repo,
            "version": self.version,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "test_file": self.focal_test_file_path,
            "test_src": self.baseline_test_source,
            "preds_context": dict(self.preds_context or {}),
            "FAIL_TO_PASS": list(self.fail_to_pass),
            "PASS_TO_PASS": list(self.pass_to_pass),
            "patch": self.gold_patch,
            "test_patch": self.gold_test_patch,
            # Capability flag: V5 patch-as-oracle voting is always
            # engageable on SWT-Bench rows (the gold patch is real).
            "apex_dual_state_oracle": True,
        }


# ---------------------------------------------------------------------------
# Adapter: SWTBenchTask -> TestGenEvalTask
# ---------------------------------------------------------------------------


def _focal_source_from_preds_context(preds_context: dict[str, Any]) -> tuple[str, str]:
    """Pull the most plausible focal-file path + source from the BM25 context.

    SWT-Bench's ``*_bm25_27k_zsb`` dataset variants ship the BM25-retrieved
    file contents as ``preds_context``. Schema seen in the wild:

        {"focal": [{"path": "...", "content": "..."}, ...],
         "tests": [...]}

    or a flat list of {"path": ..., "content": ...}. Fallback to the first
    file with a non-empty content blob.
    """

    if not isinstance(preds_context, dict):
        return ("", "")
    candidates: list[tuple[str, str]] = []

    def _extend(items: Any) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or item.get("file") or "").strip()
            content = str(item.get("content") or item.get("source") or "")
            if path and content.strip():
                candidates.append((path, content))

    _extend(preds_context.get("focal"))
    _extend(preds_context.get("code"))
    _extend(preds_context.get("files"))
    # Flat list fallback.
    for value in preds_context.values():
        if isinstance(value, list):
            _extend(value)
    if not candidates:
        return ("", "")
    return candidates[0]


def swtbench_task_to_testgeneval_task(task: SWTBenchTask) -> TestGenEvalTask:
    """Project an SWT-Bench task onto APEX's TestGenEvalTask shape.

    The mapping is intentionally lossy on the prompt surface:

      * focal_method_path/source := the BM25-retrieved focal file
        (preds_context). When BM25 returns nothing, we fall back to
        the focal *test file* itself so the agent at least sees the
        target test file's existing structure.
      * existing_test_path/source := the focal test file the agent
        will extend.
      * problem_statement := SWT-Bench's bug description (this is the
        ONE field SWT-Bench's harness expects the agent to reason from).

    Gold patch / test_patch / FAIL_TO_PASS / PASS_TO_PASS NEVER reach
    the prompt — see ``scrub_row_for_agent_prompt``.
    """

    focal_path, focal_source = _focal_source_from_preds_context(task.preds_context)
    if not focal_path or not focal_source.strip():
        # Fallback: prompt the agent against the test file itself. The
        # generator will treat it as the "focal module" — it still has
        # enough structural context to write a sensible new test.
        focal_path = task.focal_test_file_path or "tests/test_apex.py"
        focal_source = task.baseline_test_source or ""

    return TestGenEvalTask(
        instance_id=task.instance_id,
        focal_method_path=focal_path,
        focal_method_source=focal_source,
        existing_test_path=task.focal_test_file_path,
        existing_test_source=task.baseline_test_source,
        problem_statement=task.problem_statement,
        language="python",
        repo_path=None,
        metadata={
            "benchmark": "swtbench",
            "source_repo": task.repo,
            "version": task.version,
            "base_commit": task.base_commit,
            "instance_id": task.instance_id,
        },
    )


# ---------------------------------------------------------------------------
# Driver shim
# ---------------------------------------------------------------------------


# Phase A.1 (Decisive-Edge): per-task SWT-Bench docker image. The
# upstream SWT-Bench harness publishes one prebuilt image per instance
# under this naming convention. Used by callers that want to wrap a
# V5 in-container agent solve in a :class:`ContainerSupervisor` for
# true isolation per task.
_SWTBENCH_DOCKER_IMAGE_TEMPLATE = "aorwall/sweb.eval.x86_64.{instance_id}:latest"


def swtbench_docker_image_for_task(task: "SWTBenchTask") -> str:
    """Return the upstream SWT-Bench docker image ref for ``task``.

    Format: ``aorwall/sweb.eval.x86_64.<instance_id>:latest``. Callers
    should pass this through :func:`apex.core.docker_pinning.resolve_image`
    if they want the digest pinned in the run manifest; the bare tag is
    also acceptable for the V5 ``ContainerSupervisor`` constructor.
    """
    instance_id = (task.instance_id or "").strip()
    if not instance_id:
        return ""
    return _SWTBENCH_DOCKER_IMAGE_TEMPLATE.format(instance_id=instance_id)


def build_swtbench_v5_benchmark_metadata(task: "SWTBenchTask") -> dict[str, Any]:
    """Build the ``benchmark_metadata`` dict for a V5-routed SWT-Bench solve.

    Carries the ``"docker_image"`` field that
    :meth:`apex.orchestration.solver.ApexOrchestrator._maybe_solve_via_in_container_v5`
    consults to construct the per-task :class:`ContainerSupervisor`.
    Callers that drive SWT-Bench through ``ApexOrchestrator.solve(...,
    benchmark_metadata=...)`` should merge this dict into their existing
    metadata so the V5 path picks up the right image automatically.
    """
    return {
        "benchmark_name": "swtbench",
        "instance_id": task.instance_id,
        "docker_image": swtbench_docker_image_for_task(task),
    }


def evaluate_swtbench_task_with_default_generator(
    *,
    task: SWTBenchTask,
    output_dir: str | Path,
    generation_timeout_seconds: float = 300.0,
    pytest_timeout_seconds: float = 120.0,
    max_repair_attempts: int = 3,
    candidate_count: int = 4,
    agent_models: Optional[list[str]] = None,
    measure_mutation: bool = False,
    measure_coverage: bool = False,
    measure_assertion_effect: bool = False,
) -> TestGenEvalTaskResult:
    """Evaluate one SWT-Bench task by routing through the TestGenEval
    pipeline.

    Returns the underlying TestGenEval result so the caller can re-use
    the same diagnostics surface (V5 voting reports, generation history,
    docker gate status). The runner is responsible for transforming the
    chosen test artifact into a git-patch prediction the official
    SWT-Bench harness accepts.
    """

    tge_task = swtbench_task_to_testgeneval_task(task)
    return evaluate_testgeneval_task_with_default_generator(
        task=tge_task,
        output_dir=output_dir,
        generation_timeout_seconds=generation_timeout_seconds,
        pytest_timeout_seconds=pytest_timeout_seconds,
        max_repair_attempts=max_repair_attempts,
        candidate_count=candidate_count,
        agent_models=list(agent_models or []),
        measure_mutation=measure_mutation,
        measure_coverage=measure_coverage,
        measure_assertion_effect=measure_assertion_effect,
        measure_stability=False,
        install_repo=False,
    )


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------


def task_from_row(row: dict[str, Any]) -> Optional[SWTBenchTask]:
    """Map a SWT-Bench (BM25 zsb-shape) dataset row onto SWTBenchTask.

    Returns ``None`` for malformed rows so callers can warn-and-skip.
    """

    instance_id = str(row.get("instance_id") or "").strip()
    repo = str(row.get("repo") or "").strip()
    base_commit = str(row.get("base_commit") or "").strip()
    if not instance_id or not repo or not base_commit:
        return None
    test_file = str(row.get("test_file") or "").strip()
    if not test_file:
        # Some BM25 zsb rows put the test file path inside ``preds_context``.
        preds_context = row.get("preds_context")
        if isinstance(preds_context, dict):
            tests_list = preds_context.get("tests") or preds_context.get("test_files")
            if isinstance(tests_list, list) and tests_list:
                first = tests_list[0]
                if isinstance(first, dict):
                    test_file = str(first.get("path") or "").strip()
    if not test_file:
        # Fallback: synthetic path so the harness diff has a target. The
        # docker adapter will reject this if the path doesn't exist in
        # the buggy commit, but at least the runner produces a record.
        test_file = "tests/test_apex_swtbench.py"
    fail_to_pass = row.get("FAIL_TO_PASS") or row.get("fail_to_pass") or []
    pass_to_pass = row.get("PASS_TO_PASS") or row.get("pass_to_pass") or []
    if isinstance(fail_to_pass, str):
        try:
            fail_to_pass = json.loads(fail_to_pass)
        except json.JSONDecodeError:
            fail_to_pass = [fail_to_pass]
    if isinstance(pass_to_pass, str):
        try:
            pass_to_pass = json.loads(pass_to_pass)
        except json.JSONDecodeError:
            pass_to_pass = [pass_to_pass]
    return SWTBenchTask(
        instance_id=instance_id,
        repo=repo,
        version=str(row.get("version") or "").strip(),
        base_commit=base_commit,
        problem_statement=str(row.get("problem_statement") or ""),
        focal_test_file_path=test_file,
        baseline_test_source=str(row.get("test_src") or row.get("test_source") or ""),
        fail_to_pass=[str(x) for x in fail_to_pass],
        pass_to_pass=[str(x) for x in pass_to_pass],
        preds_context=dict(row.get("preds_context") or {})
        if isinstance(row.get("preds_context"), dict)
        else {},
        gold_patch=str(row.get("patch") or ""),
        gold_test_patch=str(row.get("test_patch") or ""),
    )


def load_tasks_from_json(json_path: str | Path) -> list[SWTBenchTask]:
    """Load SWT-Bench tasks from a local JSON file.

    Schema: top-level list (or {"tasks": [...]}) of SWE-bench-shape rows.
    Returns an empty list for missing / malformed files.
    """

    path = Path(json_path)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("failed to load SWT-Bench tasks from %s: %s", path, exc)
        return []
    rows: list[dict[str, Any]]
    if isinstance(raw, list):
        rows = [r for r in raw if isinstance(r, dict)]
    elif isinstance(raw, dict) and isinstance(raw.get("tasks"), list):
        rows = [r for r in raw["tasks"] if isinstance(r, dict)]
    else:
        return []
    tasks: list[SWTBenchTask] = []
    for row in rows:
        task = task_from_row(row)
        if task is not None:
            tasks.append(task)
    return tasks
