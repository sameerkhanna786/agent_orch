"""
LLM-driven solver agents used inside each rollout.
"""

from __future__ import annotations

import ast
import contextvars
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from ..agentic_search import (
    agentic_search_guidance_enabled,
    agentic_search_internet_enabled,
    augment_prompt_with_agentic_search_guidance,
)
from ..core.config import ContextConfig, LLMConfig, PromptStrategy
from ..core.git_utils import (
    expand_changed_paths,
    parse_porcelain_path,
)
from ..core.git_utils import (
    list_changed_files as list_git_changed_files,
)
from ..core.llm import AgentStateMachine, LLMClient, ToolDefinition
from ..core.pytest_report_utils import protected_test_files_from_context
from ..core.pytest_utils import (
    build_targeted_pytest_command,
    should_disable_pytest_plugin_autoload,
)
from ..planning.manager import IssuePlan, RolloutBrief
from ..preprocessing.repo_analyzer import RepoContext
from ..rollout.discovery_scope import build_discovery_scope
from ..tools.aci import (
    ACIToolExecutor,
    build_agent_tool_definitions,
    make_submit_localization_tool,
    make_submit_patch_tool,
    make_submit_reproduction_tool,
    make_submit_test_suite_tool,
)
from . import prompts as _prompts_v1
from . import prompts_v2 as _prompts_v2
from .artifacts import coerce_localization_artifact, coerce_reproduction_artifact
from .prompts import (
    build_localizer_prompt,
    build_reproducer_prompt,
    build_solver_prompt,
    build_stage_system_prompt,
    build_test_writer_prompt,
)

logger = logging.getLogger("apex.agents")


# ---------------------------------------------------------------------------
# Decisive-Edge B.8 — prompts_version registry.
#
# The agents below default to the v1 module (``apex.agents.prompts``); this
# is the prompt surface that produced the published 86.3% Commit0-Lite
# headline. ``RolloutConfig.prompts_version = "v2"`` switches each agent
# (Reproducer / Localizer / Patcher / TestWriter) to its
# ``apex.agents.prompts_v2`` counterpart in lockstep.
#
# The lookup is module-level so test code can monkey-patch
# ``_PROMPTS_REGISTRY["v2"] = some_stub`` to validate the routing.
# ---------------------------------------------------------------------------


_PROMPTS_REGISTRY: dict[str, Any] = {
    "v1": _prompts_v1,
    "v2": _prompts_v2,
}


def load_prompts_module(version: Optional[str]) -> Any:
    """Resolve the prompts module to use for an agent.

    ``version`` is normalised (``None``, empty string, unknown values
    all fall back to v1). The helper is exported so callers (notably
    the rollout engine + the prompts A/B harness) can resolve the
    module without re-implementing the fallback rules.
    """
    if version is None:
        return _prompts_v1
    key = str(version).strip().lower()
    if not key:
        return _prompts_v1
    module = _PROMPTS_REGISTRY.get(key)
    if module is None:
        logger.warning("Unknown prompts_version %r; falling back to v1.", version)
        return _prompts_v1
    return module


def _resolve_agent_prompts_module(rollout_config: Any) -> Any:
    """Pull ``prompts_version`` off the optional ``rollout_config``."""
    if rollout_config is None:
        return _prompts_v1
    version = getattr(rollout_config, "prompts_version", None)
    return load_prompts_module(version)


def _is_v1(module: Any) -> bool:
    """True when ``module`` is the v1 prompts surface.

    When the agent runs the v1 module, callers (esp. tests) often
    monkey-patch the module-level imported names in ``apex.agents.solver``
    (e.g. ``apex.agents.solver.build_solver_prompt``). To preserve that
    contract, the agent dispatches through the module-level alias for
    v1 and through ``self._prompts_module`` for v2.
    """
    return module is _prompts_v1


def _build_reproducer_prompt_dispatch(module: Any, **kwargs: Any) -> str:
    """v1 → module-level alias (test-monkey-patchable);
    v2 → bound module attribute.
    """
    if _is_v1(module):
        return build_reproducer_prompt(**kwargs)
    return module.build_reproducer_prompt(**kwargs)


def _build_localizer_prompt_dispatch(module: Any, **kwargs: Any) -> str:
    if _is_v1(module):
        return build_localizer_prompt(**kwargs)
    return module.build_localizer_prompt(**kwargs)


def _build_solver_prompt_dispatch(module: Any, **kwargs: Any) -> str:
    if _is_v1(module):
        return build_solver_prompt(**kwargs)
    return module.build_solver_prompt(**kwargs)


def _build_test_writer_prompt_dispatch(module: Any, **kwargs: Any) -> str:
    if _is_v1(module):
        return build_test_writer_prompt(**kwargs)
    return module.build_test_writer_prompt(**kwargs)


def _stage_system_prompt_dispatch(
    module: Any,
    base_prompt: str,
    *,
    allow_delegation: bool,
    issue_plan: Optional[IssuePlan] = None,
) -> str:
    if _is_v1(module):
        return build_stage_system_prompt(
            base_prompt,
            allow_delegation=allow_delegation,
            issue_plan=issue_plan,
        )
    return module.build_stage_system_prompt(
        base_prompt,
        allow_delegation=allow_delegation,
        issue_plan=issue_plan,
    )


# ---------------------------------------------------------------------------
# Decisive-Edge C.3 — per-repo episodic memory (cross-solve patterns).
#
# The orchestrator pre-loads a list of :class:`RepoEpisode` objects via
# ``set_active_repo_episodes`` BEFORE constructing the rollout engine and
# therefore the agents. Each agent's __init__ snapshots the active list
# into ``self.repo_episodes``; the prompt builders ``_repo_conventions_block``
# render it into a "# Repo conventions" section that is appended to the
# agent's system prompt (or to the ``# Context`` section in v2 prompts).
#
# A contextvars.ContextVar is used so multiple parallel solves in the same
# process (rare, but possible in tests) don't leak episodes across each
# other. The agents' ``_instantiate_agent`` (rollout/engine.py) doesn't need
# to know about this — the BaseAgent constructor reads the contextvar.
# ---------------------------------------------------------------------------


_ACTIVE_REPO_EPISODES: contextvars.ContextVar[tuple[Any, ...]] = contextvars.ContextVar(
    "apex.agents.solver.active_repo_episodes",
    default=(),
)


def set_active_repo_episodes(
    episodes: Optional[Iterable[Any]],
) -> contextvars.Token:
    """Set the per-solve repo episodes that subsequent agents will read.

    Returns the contextvars Token so callers can ``reset()`` it after the
    solve finishes (the orchestrator does this in a try/finally so the
    context is cleared even when the solve raises).
    """
    snapshot: tuple[Any, ...] = tuple(episodes or ())
    return _ACTIVE_REPO_EPISODES.set(snapshot)


def reset_active_repo_episodes(token: contextvars.Token) -> None:
    """Clear the contextvar set by :func:`set_active_repo_episodes`."""
    try:
        _ACTIVE_REPO_EPISODES.reset(token)
    except (LookupError, ValueError):  # pragma: no cover - defensive
        pass


def get_active_repo_episodes() -> tuple[Any, ...]:
    """Return the current per-solve repo episodes (empty tuple if unset)."""
    return _ACTIVE_REPO_EPISODES.get()


def _render_repo_conventions_block(episodes: Iterable[Any]) -> str:
    """Render the agent-prompt ``# Repo conventions`` block.

    Imported lazily so this module doesn't pull the persistence package
    on import (keeps the test scaffolds light). When the persistence
    package isn't importable for any reason, returns an empty string —
    the prompt is unmodified.
    """
    eps = tuple(episodes or ())
    if not eps:
        return ""
    try:
        from ..persistence.repo_episodic_store import (
            render_repo_episodes_prompt_block,
        )
    except Exception:  # pragma: no cover - defensive import guard
        return ""
    return render_repo_episodes_prompt_block(eps)


@dataclass
class AgentResult:
    """Structured result from an agent execution."""

    success: bool
    output: str
    submission_tool: Optional[str] = None
    submission: dict[str, Any] = field(default_factory=dict)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    iterations_used: int = 0
    tokens_used: int = 0
    # Phase 2C 2.7: per-stage diagnostics (e.g. localizer enforcement
    # outcome, off_target_patches counts). The engine pipes these
    # forward into ``RolloutResult.diagnostics`` for the orchestrator
    # / report layer.
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "submission_tool": self.submission_tool,
            "submission": self.submission,
            "iterations_used": self.iterations_used,
            "tokens_used": self.tokens_used,
            "diagnostics": dict(self.diagnostics),
        }


class BaseAgent:
    """Shared loop setup for the LLM-driven agents."""

    def __init__(
        self,
        llm_config: LLMConfig,
        working_dir: str,
        aci_config: Any,
        agentic_search_config: Any = None,
        context_config: Optional[ContextConfig] = None,
        repo_context: Optional[RepoContext] = None,
        rollout_id: Optional[int] = None,
        memory_bus: Any = None,
        execution_tree: Any = None,
        baseline_commit: Optional[str] = None,
        temperature: float = 0.0,
        max_iterations: int = 30,
        use_concise_prompts: bool = True,
        enable_subagents: bool = False,
        enable_delegate_subtasks: bool = False,
        prompts_version: Optional[str] = None,
        repo_episodes: Optional[Iterable[Any]] = None,
    ):
        self.working_dir = Path(working_dir).resolve()
        self.aci_config = aci_config
        self.agentic_search_config = agentic_search_config
        self.context_config = context_config or ContextConfig()
        self.execution_tree = execution_tree
        self.baseline_commit = baseline_commit
        self.enable_subagents = enable_subagents
        self.enable_delegate_subtasks = enable_delegate_subtasks and enable_subagents
        self.llm_config = llm_config
        self.temperature = temperature
        self.llm: Optional[LLMClient] = None
        self.executor = ACIToolExecutor(
            working_dir,
            aci_config,
            agentic_search_config=agentic_search_config,
            repo_context=repo_context,
            memory_bus=memory_bus,
            rollout_id=rollout_id,
            execution_tree=execution_tree,
            baseline_commit=baseline_commit,
        )
        self.max_iterations = max_iterations
        self.use_concise_prompts = use_concise_prompts
        # Decisive-Edge B.8: optional prompt module override.
        # ``None`` (default) → keep the legacy import path (``prompts.py``).
        # The PatcherAgent overrides this from its rollout_config in __init__.
        self.prompts_version = prompts_version
        self._prompts_module = load_prompts_module(prompts_version)
        # Decisive-Edge C.3: per-repo episodic patterns. Explicit kwarg
        # wins; otherwise we snapshot the current contextvar set by the
        # orchestrator at solve start. Tests that construct an agent
        # directly without going through the orchestrator just see the
        # default empty tuple, matching legacy behaviour.
        if repo_episodes is None:
            episodes_snapshot: tuple[Any, ...] = get_active_repo_episodes()
        else:
            episodes_snapshot = tuple(repo_episodes)
        self.repo_episodes: tuple[Any, ...] = episodes_snapshot

    def _local_doc_tool_enabled(self, *, stage_name: str = "") -> bool:
        return getattr(
            self.agentic_search_config, "enable_local_doc_guidance", False
        ) and agentic_search_guidance_enabled(
            self.agentic_search_config,
            stage_name=stage_name,
        )

    def _external_search_tool_enabled(
        self,
        *,
        stage_name: str = "",
        query_text: str = "",
    ) -> bool:
        return agentic_search_internet_enabled(
            self.agentic_search_config,
            stage_name=stage_name,
            query_text=query_text,
        )

    def _augment_observation(
        self,
        observation: str,
        *,
        stage_name: str = "",
        include_semiformal_editing: bool = False,
    ) -> str:
        return augment_prompt_with_agentic_search_guidance(
            observation,
            self.agentic_search_config,
            repo_root=self.working_dir,
            stage_name=stage_name,
            include_semiformal_editing=include_semiformal_editing,
        )

    def _augment_system_prompt_with_repo_episodes(self, system_prompt: str) -> str:
        """Decisive-Edge C.3: append a ``# Repo conventions`` section.

        Returns the system prompt unchanged when no episodes were loaded
        for the current solve OR when every loaded episode is below the
        confidence floor in :pyfunc:`render_repo_episodes_prompt_block`.
        This explicit no-op is what makes the C.3 test
        ``empty episodes → no Repo conventions section`` pass.
        """
        block = _render_repo_conventions_block(self.repo_episodes)
        if not block:
            return system_prompt
        # The block already starts with ``# Repo conventions`` and ends
        # with the footer note + a trailing newline, so we just sandwich
        # it onto the end of the system prompt with a separator blank
        # line. v2 prompts already render their own ``# Context`` section
        # in the user observation; the repo conventions stay in the
        # SYSTEM prompt so the agent treats them as durable framing
        # rather than per-task observation noise.
        return system_prompt.rstrip() + "\n\n" + block

    def _get_llm(self) -> LLMClient:
        if self.llm is None:
            self.llm = LLMClient(self.llm_config, temperature_override=self.temperature)
        return self.llm

    def _run_loop(
        self,
        system_prompt: str,
        initial_observation: str,
        tools: list[ToolDefinition],
        finish_tool_names: set[str],
        feedback_generator: Optional[Callable[[str], str]] = None,
    ) -> AgentResult:
        llm = self._get_llm()
        llm.reset_trajectory()
        feedback_generator = feedback_generator or self._default_feedback_generator()
        machine = AgentStateMachine(
            llm=llm,
            initial_prompt=system_prompt,
            initial_task=initial_observation,
            feedback_generator=feedback_generator,
            tools=tools,
            tool_executor=self.executor.execute,
            max_iterations=self.max_iterations,
            finish_tool_names=finish_tool_names,
            dynamic_context_provider=self.executor.render_dynamic_context,
        )
        if hasattr(machine, "set_context_config"):
            machine.set_context_config(self.context_config)
        if hasattr(self.executor, "set_agent_runtime"):
            self.executor.set_agent_runtime(machine)
        try:
            result = machine.run()
        finally:
            if hasattr(self.executor, "set_agent_runtime"):
                self.executor.set_agent_runtime(None)
        submission = result.submission
        output = (
            json.dumps(submission.arguments, indent=2)
            if submission is not None
            else result.output or "Agent did not submit a structured result."
        )
        return AgentResult(
            success=submission is not None,
            output=output,
            submission_tool=submission.tool_name if submission else None,
            submission=submission.arguments if submission else {},
            trajectory=llm.get_trajectory(),
            iterations_used=result.iterations,
            tokens_used=llm.total_tokens_used,
            # Decisive-Edge C.1: keep the raw model text alongside the
            # JSON-shaped ``output`` so callers (e.g. LocalizerAgent's
            # top-K parser) can recover free-form blocks the schema
            # can't carry. Defaults to ``""`` so legacy callers ignore
            # it cleanly.
            diagnostics={"raw_output": str(result.output or "")},
        )

    def _configure_test_runtime(self, test_command: Optional[str]) -> None:
        if hasattr(self.executor, "set_test_command"):
            self.executor.set_test_command(test_command)

    def _configure_discovery_scope(
        self,
        issue_plan: IssuePlan,
        rollout_brief: RolloutBrief,
        *,
        stage_name: str,
        reproduction_artifact: Any = None,
        localization_artifact: Any = None,
        patch_artifact: Any = None,
    ) -> None:
        if not hasattr(self.executor, "set_discovery_scope"):
            return
        scope = build_discovery_scope(
            issue_plan,
            rollout_brief,
            stage_name=stage_name,
            reproduction_artifact=reproduction_artifact,
            localization_artifact=localization_artifact,
            patch_artifact=patch_artifact,
        )
        self.executor.set_discovery_scope(**scope.to_dict())

    def _configure_delegation_plan(
        self,
        rollout_brief: RolloutBrief,
        *,
        stage_name: str,
    ) -> None:
        if not hasattr(self.executor, "set_delegation_plan"):
            return
        policy = (
            rollout_brief.delegation_policy
            if isinstance(rollout_brief.delegation_policy, dict)
            else {}
        )
        if not self.enable_delegate_subtasks or not rollout_brief.delegation_enabled(stage_name):
            self.executor.set_delegation_plan([])
            return
        self.executor.set_delegation_plan(
            list(policy.get("subtasks") or []),
            parallelism=int(policy.get("parallelism") or 1),
            max_iterations=(
                int(policy.get("max_iterations"))
                if policy.get("max_iterations") is not None
                else None
            ),
        )

    def _configure_write_scope(
        self,
        issue_plan: IssuePlan,
        rollout_brief: RolloutBrief,
    ) -> None:
        if not hasattr(self.executor, "set_write_scope"):
            return
        planner_metadata = (
            issue_plan.planner_metadata if isinstance(issue_plan.planner_metadata, dict) else {}
        )
        search_policy = (
            rollout_brief.search_policy if isinstance(rollout_brief.search_policy, dict) else {}
        )
        delegated_subtask = bool(
            planner_metadata.get("delegated_subtask")
            or search_policy.get("delegated_subtask")
            or search_policy.get("delegated_subtask_title")
        )
        protected_test_files = protected_test_files_from_context(
            issue_plan,
            exclude_incomplete_test_files=True,
        )
        if not delegated_subtask:
            self.executor.set_write_scope([], protected_test_files, enforce=False)
            return

        forbidden_files = [
            str(path).strip()
            for path in list(
                list(planner_metadata.get("delegated_forbidden_files") or [])
                + list(protected_test_files)
            )
            if str(path).strip()
        ]
        self.executor.set_write_scope(
            [],
            forbidden_files,
            enforce=False,
        )

    def _patch_self_check_message(self, changed_files: list[str]) -> str:
        """Cheap AST-based safety checks that run inside the rollout
        loop before each "submit ready?" feedback.

        Two checks fire today:
        - Public-symbol survival: did the patch delete a top-level public
          def / class / assignment that existed in the baseline?
          (Catches the "agent removes a public symbol needed by
          conftest imports" failure mode.)
        - Stub-residue scanner: are there public functions left with
          ``pass`` / ``return None`` / ``raise NotImplementedError`` /
          ``unimplemented!()`` / etc. bodies?
          (Catches the "agent leaves placeholder methods returning None"
          failure mode.)

        Both are pure-Python and run in tens of milliseconds per file —
        cheap enough to run on every feedback call, unlike a full
        ``pytest --collect-only`` which can take 10-30 seconds per repo.
        """
        if not changed_files:
            return ""
        try:
            from ..core.stub_scanner import (
                scan_files_for_stubs,
                summarize_findings,
            )
            from ..core.symbol_survival import (
                detect_public_symbol_losses,
                summarize_losses,
            )
            from ..core.test_runners import detect_adapter
        except ImportError:
            return ""
        workspace = Path(getattr(self, "repo_path", self.working_dir))
        if not workspace.exists():
            return ""
        adapter = detect_adapter(workspace)
        stub_patterns = adapter.stub_patterns() if adapter is not None else []
        sections: list[str] = []
        try:
            losses = detect_public_symbol_losses(workspace, changed_files)
        except Exception:
            losses = []
        if losses:
            sections.append(summarize_losses(losses))
        try:
            stub_findings = scan_files_for_stubs(
                workspace,
                changed_files,
                adapter_stub_patterns=stub_patterns,
            )
        except Exception:
            stub_findings = []
        if stub_findings:
            sections.append(summarize_findings(stub_findings))
        return "\n\n".join(sections)

    def _default_feedback_generator(self) -> Callable[[str], str]:
        def feedback(_: str) -> str:
            changed_files = self._changed_files()
            if not changed_files:
                return (
                    "No code changes are present yet. Continue using tools to inspect the repository "
                    "or make progress, then submit when ready."
                )
            changed_summary = ", ".join(changed_files[:8])
            if len(changed_files) > 8:
                changed_summary += f", ... (+{len(changed_files) - 8} more)"
            syntax_issue = self._changed_file_syntax_issue(changed_files)
            if syntax_issue:
                return (
                    f"Current changed files: {changed_summary}\n"
                    f"Syntax issue detected:\n{syntax_issue}\n"
                    "Revise the workspace before submitting."
                )
            self_check = self._patch_self_check_message(changed_files)
            if self_check:
                return (
                    f"Current changed files: {changed_summary}\n"
                    f"{self_check}\n"
                    "Address the issues above before submitting; the verifier "
                    "will reject patches that drop baseline public symbols or "
                    "leave stub bodies in place."
                )
            return (
                f"Current changed files: {changed_summary}\n"
                "Continue refining the workspace, run validation if useful, and submit when ready."
            )

        return feedback

    def _build_patch_feedback_generator(
        self,
        test_command: Optional[str],
        issue_plan: IssuePlan,
        rollout_brief: Optional[RolloutBrief] = None,
    ) -> Callable[[str], str]:
        validation_cache: dict[str, Any] = {
            "signature": None,
            "result": None,
            "scope": "",
        }

        def feedback(_: str) -> str:
            changed_files = self._changed_files()
            if not changed_files:
                return "No code changes are present yet. Make the fix in the workspace, then run validation."
            syntax_issue = self._changed_file_syntax_issue(changed_files)
            if syntax_issue:
                return f"Current syntax issue:\n{syntax_issue}\nRevise the workspace."

            self_check = self._patch_self_check_message(changed_files)
            if self_check:
                return (
                    f"Patch self-check:\n{self_check}\n"
                    "Restore deleted baseline symbols and implement remaining "
                    "stub bodies before continuing."
                )

            validation_scope, validation_command = self._select_feedback_validation_command(
                test_command,
                issue_plan,
                rollout_brief=rollout_brief,
            )
            if validation_command:
                signature = self._build_validation_signature(changed_files, validation_command)
                if validation_cache["signature"] != signature:
                    validation_cache["result"] = self._run_validation_command(validation_command)
                    validation_cache["signature"] = signature
                    validation_cache["scope"] = validation_scope
                validation = validation_cache["result"]
                scope = validation_cache["scope"] or "focused validation"
                if validation.returncode == 0:
                    return (
                        f"Latest {scope} passed.\n"
                        f"{validation.output}\n"
                        "Run broader verification manually before submitting if the targeted checks do not cover the full task."
                    ).strip()
                return (
                    f"Latest {scope} failed.\n"
                    f"{validation.output}\n"
                    "Revise the patch, investigate the failure, and continue."
                ).strip()

            return (
                f"Changed files: {', '.join(changed_files)}\n"
                "Run targeted validation if needed, then submit the patch when it is ready. Prefer the visible failing tests or the most relevant test files over the full suite."
            )

        return feedback

    def _changed_files(self) -> list[str]:
        return list_git_changed_files(
            self.working_dir,
            baseline_ref=self.baseline_commit,
        )

    def _changed_file_paths(self, git_status_lines: Optional[list[str]] = None) -> list[str]:
        status_lines = git_status_lines if git_status_lines is not None else self._changed_files()
        raw_paths: list[str] = []
        for line in status_lines:
            stripped = (line or "").strip()
            if not stripped:
                continue
            if re.match(r"^[ MADRCU?!]{2}\s", stripped):
                raw_paths.append(parse_porcelain_path(stripped))
            else:
                raw_paths.append(stripped)
        return expand_changed_paths(self.working_dir, raw_paths)

    def _changed_file_syntax_issue(self, git_status_lines: list[str]) -> str:
        for rel_path in self._changed_file_paths(git_status_lines):
            if not rel_path.endswith(".py"):
                continue
            file_path = self.working_dir / rel_path
            if not file_path.exists():
                continue
            try:
                ast.parse(file_path.read_text(errors="replace"))
            except SyntaxError as exc:
                return f"{rel_path}: SyntaxError at line {exc.lineno}: {exc.msg}"
        return ""

    def _run_validation_command(self, command: str) -> Any:
        normalized = command
        disable_plugin_autoload = False
        if "pytest" in command and "PYTEST_DISABLE_PLUGIN_AUTOLOAD" not in command:
            if should_disable_pytest_plugin_autoload(
                command,
                repo_root=self.working_dir,
            ):
                normalized = f"PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 {command}"
                disable_plugin_autoload = True
        env = {
            **os.environ,
            **{
                str(key): str(value)
                for key, value in dict(
                    getattr(self.aci_config, "runtime_env_overrides", {}) or {}
                ).items()
            },
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if disable_plugin_autoload:
            env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        else:
            env.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
        try:
            result = subprocess.run(
                ["bash", "-lc", normalized],
                cwd=str(self.working_dir),
                capture_output=True,
                text=True,
                timeout=max(getattr(self.aci_config, "bash_timeout", 30), 60),
                env=env,
            )
            output = "\n".join(
                part for part in [result.stdout.strip(), result.stderr.strip()] if part
            ).strip()
            if len(output.splitlines()) > getattr(self.aci_config, "max_output_lines", 220):
                lines = output.splitlines()
                output = "\n".join(
                    lines[:40] + ["... [validation output truncated] ..."] + lines[-20:]
                )
            if result.returncode != 0:
                output = f"[exit_code={result.returncode}]\n{output}".strip()
            return type(
                "ValidationResult", (), {"returncode": result.returncode, "output": output}
            )()
        except subprocess.TimeoutExpired:
            return type(
                "ValidationResult",
                (),
                {"returncode": 124, "output": "Validation command timed out."},
            )()

    def _select_feedback_validation_command(
        self,
        test_command: Optional[str],
        issue_plan: IssuePlan,
        rollout_brief: Optional[RolloutBrief] = None,
    ) -> tuple[str, Optional[str]]:
        if not test_command:
            return "", None

        test_context = issue_plan.test_context
        search_policy = dict(rollout_brief.search_policy or {}) if rollout_brief is not None else {}
        graph_target_tests = [
            str(test_id)
            for test_id in list((search_policy or {}).get("graph_target_test_ids") or [])
            if test_id
        ]
        targets: list[str] = []
        scope = ""
        total_targets = 0

        if graph_target_tests:
            total_targets = len(graph_target_tests)
            targets = list(graph_target_tests[:4])
            scope = "frontier-targeted validation"
        elif test_context.failing_test_ids:
            total_targets = test_context.failing_test_count or len(test_context.failing_test_ids)
            targets = list(test_context.failing_test_ids[:8])
            scope = "focused validation against failing visible tests"
        elif test_context.focus_test_files:
            total_targets = len(test_context.focus_test_files)
            targets = list(test_context.focus_test_files[:4])
            scope = "focused validation against relevant visible test files"
        else:
            return "", None

        command = build_targeted_pytest_command(
            test_command,
            targets,
            force_verbose=True,
            disable_plugin_autoload=should_disable_pytest_plugin_autoload(
                test_command,
                repo_root=self.working_dir,
            ),
        )
        if command is None:
            return "", None
        if total_targets > len(targets):
            scope = f"{scope} ({len(targets)}/{total_targets} targets)"
        return scope, command

    def _build_validation_signature(
        self,
        git_status_lines: list[str],
        command: str,
    ) -> tuple[str, ...]:
        signature = [command]
        for rel_path in sorted(self._changed_file_paths(git_status_lines)):
            file_path = self.working_dir / rel_path
            if not file_path.exists():
                signature.append(f"{rel_path}:missing")
                continue
            stat = file_path.stat()
            signature.append(f"{rel_path}:{stat.st_size}:{stat.st_mtime_ns}")
        return tuple(signature)


class ReproducerAgent(BaseAgent):
    """Focused reproduction agent."""

    def run(
        self,
        issue_description: str,
        issue_plan: IssuePlan,
        rollout_brief: RolloutBrief,
        test_command: Optional[str] = None,
    ) -> AgentResult:
        logger.info("Running Reproducer Agent")
        self._configure_test_runtime(test_command or issue_plan.test_context.command)
        self._configure_write_scope(issue_plan, rollout_brief)
        self._configure_discovery_scope(
            issue_plan,
            rollout_brief,
            stage_name="reproducer",
        )
        self._configure_delegation_plan(
            rollout_brief,
            stage_name="reproducer",
        )
        observation = _build_reproducer_prompt_dispatch(
            self._prompts_module,
            issue_description=issue_description,
            issue_plan=issue_plan,
            rollout_brief=rollout_brief,
            test_command=test_command,
            concise=self.use_concise_prompts,
        )
        observation = self._augment_observation(
            observation,
            stage_name="reproducer",
        )
        tools = build_agent_tool_definitions(
            enable_subagents=self.enable_subagents,
            enable_delegate_subtasks=self.enable_delegate_subtasks,
            enable_project_doc_search=self._local_doc_tool_enabled(stage_name="reproducer"),
            enable_external_search=self._external_search_tool_enabled(
                stage_name="reproducer",
                query_text=observation,
            ),
        ) + [make_submit_reproduction_tool()]
        return self._run_loop(
            system_prompt=self._augment_system_prompt_with_repo_episodes(
                _stage_system_prompt_dispatch(
                    self._prompts_module,
                    self._prompts_module.REPRODUCER_SYSTEM_PROMPT,
                    allow_delegation=self.enable_delegate_subtasks,
                    issue_plan=issue_plan,
                )
            ),
            initial_observation=observation,
            tools=tools,
            finish_tool_names={"submit_reproduction"},
        )


class LocalizerAgent(BaseAgent):
    """Focused fault-localization agent.

    Decisive-Edge C.1 extends the agent with a ``top_k`` parameter
    (default ``1``, back-compat). When ``top_k > 1`` the agent's prompt
    is augmented with a request for *K ranked hypotheses* and the
    returned :class:`AgentResult` carries an extra ``hypotheses`` list
    in ``result.diagnostics["localizer_hypotheses"]`` — one entry per
    requested rank, each shaped like the legacy submission
    (``summary``, ``files``, ``symbols``, ``hypotheses``, ``confidence``).

    The new helper :meth:`run_top_k` returns the parsed list directly
    so the rollout engine can dispatch hypothesis ``i`` to rollout
    ``i mod K`` without re-running the localizer per rollout.
    """

    def run(
        self,
        issue_description: str,
        issue_plan: IssuePlan,
        rollout_brief: RolloutBrief,
        reproduction_artifact: Any = None,
        reproduction_summary: Optional[str] = None,
        top_k: int = 1,
    ) -> AgentResult:
        logger.info("Running Localizer Agent (top_k=%s)", top_k)
        self._configure_test_runtime(issue_plan.test_context.command)
        self._configure_write_scope(issue_plan, rollout_brief)
        artifact = coerce_reproduction_artifact(reproduction_artifact)
        self._configure_discovery_scope(
            issue_plan,
            rollout_brief,
            stage_name="localizer",
            reproduction_artifact=artifact,
        )
        self._configure_delegation_plan(
            rollout_brief,
            stage_name="localizer",
        )
        observation = _build_localizer_prompt_dispatch(
            self._prompts_module,
            issue_description=issue_description,
            issue_plan=issue_plan,
            rollout_brief=rollout_brief,
            reproduction_artifact=artifact,
            reproduction_summary=reproduction_summary,
            concise=self.use_concise_prompts,
        )
        observation = self._augment_observation(
            observation,
            stage_name="localizer",
        )
        # Decisive-Edge C.1: ask the model for K ranked hypotheses when
        # top_k > 1. The single submit_localization call carries the
        # *primary* (rank 0) hypothesis; the ranked tail is returned
        # inline as a YAML/JSON-ish ``alternative_hypotheses`` block in
        # the agent's free-text output and parsed by
        # :func:`extract_localizer_hypotheses`. We deliberately keep the
        # tool surface unchanged so legacy callers still work.
        normalized_top_k = max(1, int(top_k or 1))
        if normalized_top_k > 1:
            observation = augment_localizer_prompt_with_top_k(
                observation,
                top_k=normalized_top_k,
            )
        tools = build_agent_tool_definitions(
            enable_subagents=self.enable_subagents,
            enable_delegate_subtasks=self.enable_delegate_subtasks,
            enable_project_doc_search=self._local_doc_tool_enabled(stage_name="localizer"),
            enable_external_search=self._external_search_tool_enabled(
                stage_name="localizer",
                query_text=observation,
            ),
            # WS3F: the localizer explores read-only — drop the mutation tools.
            read_only=True,
        ) + [make_submit_localization_tool()]
        # WS3F: engage the server-side read-only gate (defence-in-depth beyond the
        # advertised tool set) for the localization/explore phase.
        if hasattr(self.executor, "set_read_only"):
            self.executor.set_read_only(True)
        result = self._run_loop(
            system_prompt=self._augment_system_prompt_with_repo_episodes(
                _stage_system_prompt_dispatch(
                    self._prompts_module,
                    self._prompts_module.LOCALIZER_SYSTEM_PROMPT,
                    allow_delegation=self.enable_delegate_subtasks,
                    issue_plan=issue_plan,
                )
            ),
            initial_observation=observation,
            tools=tools,
            finish_tool_names={"submit_localization"},
        )
        # Always populate ``localizer_hypotheses`` with at least the
        # primary submission so callers can treat the surface
        # uniformly. ``top_k > 1`` enriches the tail with parsed
        # alternatives from the model's free-text output (when present).
        primary = _hypothesis_from_submission(result.submission)
        hypotheses: list[dict[str, Any]] = []
        if primary is not None:
            hypotheses.append(primary)
        if normalized_top_k > 1:
            # When a submission exists, ``result.output`` holds the
            # JSON-encoded submission args (not the raw model text).
            # The raw text is preserved on
            # ``result.diagnostics['raw_output']`` for exactly this
            # case. Try both surfaces so monkey-patched fakes that
            # ignore diagnostics still parse correctly.
            raw_output = ""
            if isinstance(result.diagnostics, dict):
                raw_output = str(result.diagnostics.get("raw_output") or "")
            search_target = raw_output or str(result.output or "")
            extras = extract_localizer_hypotheses(
                search_target,
                limit=normalized_top_k - len(hypotheses),
            )
            hypotheses.extend(extras)
        diagnostics = dict(result.diagnostics or {})
        diagnostics["localizer_hypotheses"] = hypotheses
        diagnostics["localizer_top_k_requested"] = normalized_top_k
        diagnostics["localizer_top_k_returned"] = len(hypotheses)
        result.diagnostics = diagnostics
        return result

    def run_top_k(
        self,
        issue_description: str,
        issue_plan: IssuePlan,
        rollout_brief: RolloutBrief,
        reproduction_artifact: Any = None,
        reproduction_summary: Optional[str] = None,
        top_k: int = 1,
    ) -> list[dict[str, Any]]:
        """Convenience wrapper that returns the ranked hypothesis list.

        ``top_k=1`` returns ``[primary]``. Each hypothesis dict is shaped
        like the legacy ``submit_localization`` schema with an extra
        ``rank`` (0-based) and optional ``confidence`` (0..1) when the
        model emits one.
        """
        result = self.run(
            issue_description=issue_description,
            issue_plan=issue_plan,
            rollout_brief=rollout_brief,
            reproduction_artifact=reproduction_artifact,
            reproduction_summary=reproduction_summary,
            top_k=top_k,
        )
        hypotheses = list((result.diagnostics or {}).get("localizer_hypotheses") or [])
        # Stamp the rank so callers can dispatch by index without
        # tracking insertion order.
        for idx, item in enumerate(hypotheses):
            if isinstance(item, dict):
                item.setdefault("rank", idx)
        return hypotheses


# Decisive-Edge C.1 — top-K helpers exposed module-level so the rollout
# engine and tests can reuse them without instantiating the agent.

_LOCALIZER_TOP_K_PROMPT_SUFFIX = (
    "\n\n# Decisive-Edge C.1 — Top-K Ranked Hypotheses\n"
    "After you call ``submit_localization`` with the single best (rank 0) "
    "hypothesis, ALSO emit a free-text JSON block in your final output of "
    "the form:\n"
    "```json\n"
    "{{\n"
    '  "alternative_hypotheses": [\n'
    '    {{"summary": "...", "files": [...], "symbols": [...], '
    '"hypotheses": [...], "confidence": 0.0..1.0}},\n'
    "    ...\n"
    "  ]\n"
    "}}\n"
    "```\n"
    "Provide up to {n_alternatives} *strategically distinct* alternative "
    "hypotheses ranked by descending confidence. The K rollouts in this "
    "wave will each consume one of these K = 1 + alternatives hypotheses, "
    "so distinct hypotheses (different files / symbols / mechanisms) "
    "give the most lift over duplicates."
)


def augment_localizer_prompt_with_top_k(prompt: str, *, top_k: int) -> str:
    """Append the top-K hypothesis instructions to ``prompt``.

    ``top_k <= 1`` is a no-op so callers can pass it unconditionally.
    """
    if top_k is None or int(top_k) <= 1:
        return prompt
    n_alternatives = max(1, int(top_k) - 1)
    suffix = _LOCALIZER_TOP_K_PROMPT_SUFFIX.format(n_alternatives=n_alternatives)
    return f"{prompt}{suffix}"


def _hypothesis_from_submission(
    submission: dict[str, Any] | None,
) -> Optional[dict[str, Any]]:
    """Build a normalized hypothesis dict from a submit_localization payload."""
    if not isinstance(submission, dict) or not submission:
        return None
    files = [str(item).strip() for item in list(submission.get("files") or []) if str(item).strip()]
    symbols = [
        str(item).strip() for item in list(submission.get("symbols") or []) if str(item).strip()
    ]
    hypotheses = [
        str(item).strip() for item in list(submission.get("hypotheses") or []) if str(item).strip()
    ]
    summary = str(submission.get("summary") or "").strip()
    if not (summary or files or symbols or hypotheses):
        return None
    confidence_raw = submission.get("confidence")
    confidence: Optional[float] = None
    if isinstance(confidence_raw, (int, float)):
        confidence = float(confidence_raw)
    return {
        "rank": 0,
        "summary": summary,
        "files": files,
        "symbols": symbols,
        "hypotheses": hypotheses,
        "confidence": confidence,
    }


_ALT_BLOCK_PATTERN = re.compile(
    r"\{[^{}]*\"alternative_hypotheses\"\s*:\s*\[(.*?)\][^{}]*\}",
    re.DOTALL,
)


def extract_localizer_hypotheses(
    output: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    r"""Parse alternative hypotheses from the agent's free-text output.

    The model is instructed to emit ``{"alternative_hypotheses": [...]}``;
    we look for any JSON object containing that key. Fence markers
    (``\`\`\`json``) and surrounding prose are tolerated. Returns at most
    ``limit`` parsed entries (each shaped like the primary hypothesis,
    with ``rank`` starting at ``1``).

    Returns an empty list on any parse / shape failure so the caller can
    degrade silently to the primary submission.
    """
    if not output or not isinstance(output, str) or limit <= 0:
        return []
    parsed: list[dict[str, Any]] = []
    candidate_blobs: list[str] = []
    # Try fenced JSON first (most common when the agent obeys the
    # instructions verbatim).
    fence_match = re.search(
        r"```json\s*(\{.*?\})\s*```",
        output,
        re.DOTALL | re.IGNORECASE,
    )
    if fence_match is not None:
        candidate_blobs.append(fence_match.group(1))
    # Fallback: any object literal that mentions the key. We use a
    # bounded balanced search (re-parse with json.loads after trimming)
    # so unterminated braces don't blow up.
    for match in re.finditer(r"\{[^{}]*\"alternative_hypotheses\"", output):
        start = match.start()
        depth = 0
        for end in range(start, len(output)):
            char = output[end]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate_blobs.append(output[start : end + 1])
                    break
    seen_blobs: set[str] = set()
    for blob in candidate_blobs:
        normalized = blob.strip()
        if not normalized or normalized in seen_blobs:
            continue
        seen_blobs.add(normalized)
        try:
            data = json.loads(normalized)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        alternatives = data.get("alternative_hypotheses")
        if not isinstance(alternatives, list):
            continue
        for entry in alternatives:
            if not isinstance(entry, dict):
                continue
            files = [
                str(item).strip() for item in list(entry.get("files") or []) if str(item).strip()
            ]
            symbols = [
                str(item).strip() for item in list(entry.get("symbols") or []) if str(item).strip()
            ]
            hypotheses = [
                str(item).strip()
                for item in list(entry.get("hypotheses") or [])
                if str(item).strip()
            ]
            summary = str(entry.get("summary") or "").strip()
            if not (summary or files or symbols or hypotheses):
                continue
            confidence_raw = entry.get("confidence")
            confidence: Optional[float] = None
            if isinstance(confidence_raw, (int, float)):
                confidence = float(confidence_raw)
            parsed.append(
                {
                    "rank": len(parsed) + 1,
                    "summary": summary,
                    "files": files,
                    "symbols": symbols,
                    "hypotheses": hypotheses,
                    "confidence": confidence,
                }
            )
            if len(parsed) >= limit:
                return parsed
        if parsed:
            break
    return parsed


class PatcherAgent(BaseAgent):
    """Main patch-generation agent."""

    def __init__(
        self,
        llm_config: LLMConfig,
        working_dir: str,
        aci_config: Any,
        agentic_search_config: Any = None,
        context_config: Optional[ContextConfig] = None,
        repo_context: Optional[RepoContext] = None,
        rollout_id: Optional[int] = None,
        memory_bus: Any = None,
        execution_tree: Any = None,
        baseline_commit: Optional[str] = None,
        temperature: float = 0.0,
        max_iterations: int = 30,
        strategy: PromptStrategy = PromptStrategy.COMPREHENSIVE,
        use_concise_prompts: bool = True,
        rollout_config: Any = None,
        repo_episodes: Optional[Iterable[Any]] = None,
    ):
        # Decisive-Edge B.8: pull prompts_version off the rollout_config
        # before super().__init__ so the BaseAgent module-resolution
        # picks the right module for THIS agent. Defaults to v1 when
        # rollout_config is None or lacks the field.
        prompts_version = (
            getattr(rollout_config, "prompts_version", None) if rollout_config is not None else None
        )
        super().__init__(
            llm_config,
            working_dir,
            aci_config,
            agentic_search_config=agentic_search_config,
            context_config=context_config,
            repo_context=repo_context,
            rollout_id=rollout_id,
            memory_bus=memory_bus,
            execution_tree=execution_tree,
            baseline_commit=baseline_commit,
            temperature=temperature,
            max_iterations=max_iterations,
            use_concise_prompts=use_concise_prompts,
            prompts_version=prompts_version,
            repo_episodes=repo_episodes,
        )
        self.strategy = strategy
        # Phase 2C 2.7: optional RolloutConfig so the agent can honour
        # ``localizer_enforcement`` / ``localizer_allowlist_files``. The
        # engine's ``_instantiate_agent`` (rollout/engine.py) filters
        # kwargs by signature, so this remains None until the engine is
        # updated to plumb the config through (Phase 3). Tests pass it
        # explicitly.
        self.rollout_config = rollout_config

    def run(
        self,
        issue_description: str,
        issue_plan: IssuePlan,
        rollout_brief: RolloutBrief,
        test_command: Optional[str] = None,
        reproduction_artifact: Any = None,
        localization_artifact: Any = None,
        reproduction_summary: Optional[str] = None,
        localization_summary: Optional[str] = None,
    ) -> AgentResult:
        logger.info("Running Patcher Agent (strategy=%s)", self.strategy.value)
        self._configure_test_runtime(test_command or issue_plan.test_context.command)
        self._configure_write_scope(issue_plan, rollout_brief)
        repro = coerce_reproduction_artifact(reproduction_artifact)
        localization = coerce_localization_artifact(localization_artifact)
        # Phase 2C 2.7: install the localizer constraint on the executor
        # BEFORE the agent loop starts. The constraint stays advisory by
        # default (no behaviour change for legacy callers); enforcement
        # mode is read from the optional ``rollout_config``.
        self._configure_localizer_constraint(localization)
        self._configure_discovery_scope(
            issue_plan,
            rollout_brief,
            stage_name="patcher",
            reproduction_artifact=repro,
            localization_artifact=localization,
        )
        self._configure_delegation_plan(
            rollout_brief,
            stage_name="patcher",
        )
        observation = _build_solver_prompt_dispatch(
            self._prompts_module,
            issue_description=issue_description,
            issue_plan=issue_plan,
            rollout_brief=rollout_brief,
            strategy=self.strategy,
            test_command=test_command,
            reproduction_artifact=repro,
            localization_artifact=localization,
            reproduction_summary=reproduction_summary,
            localization_summary=localization_summary,
            concise=self.use_concise_prompts,
            allow_delegation=self.enable_delegate_subtasks,
        )
        observation = self._augment_observation(
            observation,
            stage_name="patcher",
            include_semiformal_editing=True,
        )
        # WS3F: clear the read-only gate so the patch/write phase can edit, even
        # if this executor instance was previously used for read-only localization.
        if hasattr(self.executor, "set_read_only"):
            self.executor.set_read_only(False)
        tools = build_agent_tool_definitions(
            enable_subagents=self.enable_subagents,
            enable_delegate_subtasks=self.enable_delegate_subtasks,
            enable_project_doc_search=self._local_doc_tool_enabled(stage_name="patcher"),
            enable_external_search=self._external_search_tool_enabled(
                stage_name="patcher",
                query_text=observation,
            ),
        ) + [make_submit_patch_tool()]
        agent_result = self._run_loop(
            system_prompt=self._augment_system_prompt_with_repo_episodes(
                _stage_system_prompt_dispatch(
                    self._prompts_module,
                    self._prompts_module.SOLVER_SYSTEM_PROMPT,
                    allow_delegation=self.enable_delegate_subtasks,
                    issue_plan=issue_plan,
                )
            ),
            initial_observation=observation,
            tools=tools,
            finish_tool_names={"submit_patch"},
            feedback_generator=self._build_patch_feedback_generator(
                test_command,
                issue_plan,
                rollout_brief,
            ),
        )
        # Phase 2C 2.7: post-validate the patch against the localizer
        # constraint. Off-localizer source edits are diagnostics and
        # planning pressure; they are not validity failures by themselves.
        self._enforce_localizer_constraint_on_result(agent_result)
        return agent_result

    def _configure_localizer_constraint(
        self,
        localization: Any,
    ) -> None:
        """Phase 2C 2.7: push the localizer's file scope onto the executor.

        Reads ``RolloutConfig.localizer_enforcement`` (and
        ``localizer_allowlist_files``) from ``self.rollout_config`` when
        present; defaults to ``advisory`` otherwise so legacy callers
        see no behaviour change. Falls back to a no-op when:

            * the executor doesn't expose ``set_localizer_constraint``
            * the localization artifact is absent or has no ``files``
        """
        if not hasattr(self.executor, "set_localizer_constraint"):
            return
        localization_files: list[str] = []
        if localization is not None:
            files_attr = getattr(localization, "files", None)
            if isinstance(files_attr, (list, tuple)):
                localization_files = [str(item) for item in files_attr if str(item).strip()]
        enforcement = "advisory"
        allowlist_entries: list[str] = []
        if self.rollout_config is not None:
            enforcement = (
                str(getattr(self.rollout_config, "localizer_enforcement", "advisory") or "advisory")
                .strip()
                .lower()
            )
            allowlist_attr = getattr(self.rollout_config, "localizer_allowlist_files", None)
            if isinstance(allowlist_attr, (list, tuple)):
                allowlist_entries = [str(item) for item in allowlist_attr if str(item).strip()]
        # Decisive-Edge B.1: split entries that contain glob meta-characters
        # off into ``allowlist_globs`` so ``.github/workflows/*`` and similar
        # patterns are matched via ``fnmatch`` rather than the literal-or-
        # prefix scope match used for plain file paths.
        allowlist_files: list[str] = []
        allowlist_globs: list[str] = []
        for entry in allowlist_entries:
            if any(meta in entry for meta in ("*", "?", "[")):
                allowlist_globs.append(entry)
            else:
                allowlist_files.append(entry)
        # If the localizer never produced files, hard_constraint would
        # block ALL diffs — degrade silently to advisory in that case.
        if not localization_files and enforcement == "hard_constraint":
            logger.info(
                "Localizer constraint requested 'hard_constraint' but the "
                "localizer artifact has no files; degrading to 'advisory'."
            )
            enforcement = "advisory"
        self.executor.set_localizer_constraint(
            files=localization_files,
            enforcement=enforcement,
            allowlist_files=allowlist_files,
            allowlist_globs=allowlist_globs or None,
        )

    def _enforce_localizer_constraint_on_result(
        self,
        agent_result: AgentResult,
    ) -> None:
        """Phase 2C 2.7: post-validate the agent's submission and update
        the result in place.

        Off-localizer diffs are recorded in ``agent_result.diagnostics`` so
        planning and verification can react to them. Localization is not a
        hard validity boundary; protected-file gates and objective
        verification decide whether a candidate is harmful.
        """
        if not hasattr(self.executor, "validate_patch_submission"):
            return
        validation = self.executor.validate_patch_submission()
        diagnostics = self.executor.localizer_diagnostics()
        existing = agent_result.diagnostics if isinstance(agent_result.diagnostics, dict) else {}
        merged = dict(existing)
        merged.update(diagnostics)
        merged["localizer_validation"] = validation
        agent_result.diagnostics = merged
        return


class FullSolverAgent(PatcherAgent):
    """Single-agent free-workflow solver."""

    def run(
        self,
        issue_description: str,
        issue_plan: IssuePlan,
        rollout_brief: RolloutBrief,
        test_command: Optional[str] = None,
    ) -> AgentResult:
        logger.info("Running Full Solver Agent (strategy=%s)", self.strategy.value)
        return super().run(
            issue_description=issue_description,
            issue_plan=issue_plan,
            rollout_brief=rollout_brief,
            test_command=test_command,
            reproduction_summary=None,
            localization_summary=None,
        )


class TestWriterAgent(BaseAgent):
    """Generate a synthetic test portfolio for rollout cross-validation."""

    def run(
        self,
        issue_description: str,
        issue_plan: IssuePlan,
        rollout_brief: RolloutBrief,
        reproduction_artifact: Any = None,
        localization_artifact: Any = None,
        reproduction_summary: Optional[str] = None,
        localization_summary: Optional[str] = None,
        prompt_appendix: str = "",
    ) -> AgentResult:
        logger.info("Running TestWriter Agent")
        self._configure_test_runtime(issue_plan.test_context.command)
        self._configure_write_scope(issue_plan, rollout_brief)
        repro = coerce_reproduction_artifact(reproduction_artifact)
        localization = coerce_localization_artifact(localization_artifact)
        self._configure_discovery_scope(
            issue_plan,
            rollout_brief,
            stage_name="test_writer",
            reproduction_artifact=repro,
            localization_artifact=localization,
        )
        self._configure_delegation_plan(
            rollout_brief,
            stage_name="test_writer",
        )
        observation = _build_test_writer_prompt_dispatch(
            self._prompts_module,
            issue_description=issue_description,
            issue_plan=issue_plan,
            rollout_brief=rollout_brief,
            reproduction_artifact=repro,
            localization_artifact=localization,
            reproduction_summary=reproduction_summary,
            localization_summary=localization_summary,
            allow_delegation=self.enable_delegate_subtasks,
            concise=self.use_concise_prompts,
        )
        observation = self._augment_observation(
            observation,
            stage_name="test_writer",
        )
        if str(prompt_appendix or "").strip():
            observation += "\n\n" + str(prompt_appendix or "").strip()
        tools = build_agent_tool_definitions(
            enable_subagents=self.enable_subagents,
            enable_delegate_subtasks=self.enable_delegate_subtasks,
            enable_project_doc_search=self._local_doc_tool_enabled(stage_name="test_writer"),
            enable_external_search=self._external_search_tool_enabled(
                stage_name="test_writer",
                query_text=observation,
            ),
        ) + [make_submit_test_suite_tool()]
        return self._run_loop(
            system_prompt=self._augment_system_prompt_with_repo_episodes(
                _stage_system_prompt_dispatch(
                    self._prompts_module,
                    self._prompts_module.TEST_WRITER_SYSTEM_PROMPT,
                    allow_delegation=self.enable_delegate_subtasks,
                    issue_plan=issue_plan,
                )
            ),
            initial_observation=observation,
            tools=tools,
            finish_tool_names={"submit_test_suite"},
        )
