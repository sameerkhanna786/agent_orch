"""
Agent-computer interface tools for APEX.
"""

from __future__ import annotations

import ast
import contextvars
import json
import logging
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import Request, urlopen

from ..agentic_search import agentic_search_internet_enabled, collect_local_reference_files
from ..core.cli_backend import redact_host_secrets
from ..core.config import ACIConfig
from ..core.failure_classifier import classify_failure as _classify_failure_core
from ..core.filesystem import copy_tree
from ..core.git_utils import (
    ignored_change_pathspecs,
    normalize_changed_path,
)
from ..core.git_utils import (
    list_changed_files as list_git_changed_files,
)
from ..core.llm import AgentLoop, LLMClient, ToolDefinition
from ..core.pytest_utils import (
    build_ephemeral_pytest_command,
    build_pytest_recovery_commands,
    build_runtime_python_command,
    output_indicates_missing_pytest,
    should_disable_pytest_plugin_autoload,
)
from ..preprocessing.repo_analyzer import RepoContext
from ..rollout.localizer_scope import (
    infer_scope_class,
    is_apex_harness_path,
    is_test_path,
    localizer_severity,
    split_localizer_focus,
)
from ..rollout.patch_sanitizer import build_patch_manifest, filter_solution_paths
from .aci_security import (
    TestCommandRejectedError,
    resolve_bash_invocation,
    validate_test_command,
)

_security_logger = logging.getLogger("apex.security")


class _DuckDuckGoHTMLParser(HTMLParser):
    """Minimal parser for DuckDuckGo HTML result pages."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capture_field: str = ""
        self._snippet_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: (value or "") for key, value in attrs}
        classes = attributes.get("class", "")
        if tag == "a" and "result__a" in classes:
            self._finalize_current()
            self._current = {"url": attributes.get("href", ""), "title": "", "snippet": ""}
            self._capture_field = "title"
            return
        if self._current is None:
            return
        if "result__snippet" in classes:
            self._capture_field = "snippet"
            self._snippet_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._capture_field == "title" and tag == "a":
            self._capture_field = ""
            return
        if self._capture_field == "snippet" and tag in {"a", "div", "span"}:
            self._snippet_depth = max(0, self._snippet_depth - 1)
            if self._snippet_depth == 0:
                self._capture_field = ""

    def handle_data(self, data: str) -> None:
        if self._current is None or not self._capture_field:
            return
        self._current[self._capture_field] = self._current.get(self._capture_field, "") + data

    def close(self) -> None:
        super().close()
        self._finalize_current()

    def _finalize_current(self) -> None:
        if not self._current:
            return
        title = re.sub(r"\s+", " ", self._current.get("title", "")).strip()
        url = self._normalize_result_url(self._current.get("url", ""))
        snippet = re.sub(r"\s+", " ", self._current.get("snippet", "")).strip()
        if title and url:
            self.results.append(
                {
                    "title": unescape(title),
                    "url": unescape(url),
                    "snippet": unescape(snippet),
                }
            )
        self._current = None
        self._capture_field = ""
        self._snippet_depth = 0

    def _normalize_result_url(self, raw_url: str) -> str:
        parsed = urlparse(raw_url)
        query = parse_qs(parsed.query)
        if "uddg" in query and query["uddg"]:
            return query["uddg"][0]
        return raw_url


BASE_TOOL_DEFINITIONS: list[ToolDefinition] = [
    ToolDefinition(
        name="view_file",
        description=(
            "View a file in a bounded line range. Use this instead of raw cat so you "
            "do not overload context."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["path"],
        },
    ),
    ToolDefinition(
        name="search_files",
        description=(
            "Search for a regex pattern and return only matching file paths. Use "
            "file_pattern to narrow to file extensions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "file_pattern": {"type": "string"},
            },
            "required": ["pattern"],
        },
    ),
    ToolDefinition(
        name="search_project_docs",
        description=(
            "Search local documentation and dependency metadata files such as "
            "README, docs/, AGENTS.md, pyproject.toml, setup.cfg, and requirements "
            "files. Use this before broad code exploration when you need repo or "
            "dependency usage guidance."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        },
    ),
    ToolDefinition(
        name="search_web_evidence",
        description=(
            "Search the public web for third-party API contracts, version-specific "
            "library behavior, upstream source references, or accepted community "
            "answers. This is only available in internet-aware mode."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "source": {
                    "type": "string",
                    "enum": ["all", "github", "stackoverflow", "docs"],
                },
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        },
    ),
    ToolDefinition(
        name="find_symbols",
        description=(
            "Search Python files for symbol definitions by name. Returns file paths, "
            "line numbers, and signatures."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["class", "function", "method", "any"],
                },
                "path": {"type": "string"},
            },
            "required": ["name"],
        },
    ),
    ToolDefinition(
        name="lookup_definition",
        description=(
            "Look up a class, function, or method using the repository graph. Returns "
            "its source location, code, and structural relationships."
        ),
        parameters={
            "type": "object",
            "properties": {"symbol_name": {"type": "string"}},
            "required": ["symbol_name"],
        },
    ),
    ToolDefinition(
        name="trace_callers",
        description="Find graph entities that call a given symbol.",
        parameters={
            "type": "object",
            "properties": {"symbol_name": {"type": "string"}},
            "required": ["symbol_name"],
        },
    ),
    ToolDefinition(
        name="trace_callees",
        description="Find graph entities called by a given symbol.",
        parameters={
            "type": "object",
            "properties": {"symbol_name": {"type": "string"}},
            "required": ["symbol_name"],
        },
    ),
    ToolDefinition(
        name="get_entity_context",
        description=(
            "Get the full graph context of an entity including its container, siblings, "
            "callers, and callees."
        ),
        parameters={
            "type": "object",
            "properties": {"entity_name": {"type": "string"}},
            "required": ["entity_name"],
        },
    ),
    ToolDefinition(
        name="edit_file",
        description=(
            "Edit a file by replacing one exact text block. Python edits are syntax "
            "checked before being written."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    ),
    ToolDefinition(
        name="bash",
        description=(
            "Run a shell command. The current working directory is preserved across "
            "calls inside the rollout workspace."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer"},
            },
            "required": ["command"],
        },
    ),
    ToolDefinition(
        name="run_test_on_patch",
        description=(
            "Write a temporary Python test script and run it against the current "
            "workspace patch. Use this for focused behavior checks without editing "
            "repository files permanently."
        ),
        parameters={
            "type": "object",
            "properties": {
                "test_code": {"type": "string"},
            },
            "required": ["test_code"],
        },
    ),
    ToolDefinition(
        name="invoke_debugger",
        description=(
            "Launch a concise debugging subagent for a failing command and return a "
            "runtime-focused summary."
        ),
        parameters={
            "type": "object",
            "properties": {
                "test_command": {"type": "string"},
                "suspect_file": {"type": "string"},
                "suspect_lines": {"type": "array", "items": {"type": "integer"}},
                "hypothesis": {"type": "string"},
            },
            "required": ["test_command", "suspect_file"],
        },
    ),
    ToolDefinition(
        name="investigate",
        description=(
            "Spawn a focused research sub-agent to investigate a specific question. "
            "The sub-agent's intermediate exploration is folded away and only a concise "
            "summary is returned."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "max_iterations": {"type": "integer"},
            },
            "required": ["question"],
        },
    ),
    ToolDefinition(
        name="delegate_subtasks",
        description=(
            "Spawn an APEX-managed child-agent team in isolated workspaces using the "
            "orchestrator-authored delegation plan for this rollout. Reference only the "
            "provided planned task ids or titles; child agents may read, edit, and test, "
            "but they do not invent new delegation trees."
        ),
        parameters={
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string"},
                            "title": {"type": "string"},
                            "goal": {"type": "string"},
                            "owned_files": {"type": "array", "items": {"type": "string"}},
                            "focus_files": {"type": "array", "items": {"type": "string"}},
                            "forbidden_files": {"type": "array", "items": {"type": "string"}},
                            "interface_symbols": {"type": "array", "items": {"type": "string"}},
                            "assumptions": {"type": "array", "items": {"type": "string"}},
                            "escalation_triggers": {"type": "array", "items": {"type": "string"}},
                            "success_criteria": {"type": "array", "items": {"type": "string"}},
                            "hypotheses": {"type": "array", "items": {"type": "string"}},
                            "depends_on": {"type": "array", "items": {"type": "string"}},
                            "max_iterations": {"type": "integer"},
                        },
                        "required": ["title", "goal"],
                    },
                },
                "parallelism": {"type": "integer"},
            },
            "required": ["tasks"],
        },
    ),
    ToolDefinition(
        name="broadcast_discovery",
        description=("Share a concise discovery with other rollouts working on the same issue."),
        parameters={
            "type": "object",
            "properties": {
                "insight_type": {
                    "type": "string",
                    "enum": [
                        "DEAD_END",
                        "ROOT_CAUSE",
                        "RELEVANT_FILE",
                        "TEST_STRATEGY",
                        "PATCH_DIRECTION",
                    ],
                },
                "description": {"type": "string"},
                "confidence": {"type": "number"},
                "file_paths": {"type": "array", "items": {"type": "string"}},
                "symbols": {"type": "array", "items": {"type": "string"}},
                "test_ids": {"type": "array", "items": {"type": "string"}},
                "stage_name": {"type": "string"},
                "negative": {"type": "boolean"},
                # Phase 6.2: when true and a persistent episodic store is
                # attached to the executor, also persist the discovery
                # cross-run (next solve on this task signature can read it).
                "cross_run": {"type": "boolean"},
            },
            "required": ["insight_type", "description"],
        },
    ),
    ToolDefinition(
        name="query_discoveries",
        description="Read discoveries published by other concurrent rollouts.",
        parameters={
            "type": "object",
            "properties": {
                "insight_types": {"type": "array", "items": {"type": "string"}},
                "file_paths": {"type": "array", "items": {"type": "string"}},
                "symbols": {"type": "array", "items": {"type": "string"}},
                "test_ids": {"type": "array", "items": {"type": "string"}},
                "stage_names": {"type": "array", "items": {"type": "string"}},
                "negative_only": {"type": "boolean"},
                "positive_only": {"type": "boolean"},
                "max_items": {"type": "integer"},
                # Phase 6.2: when true, query the persistent cross-run
                # episodic store (if attached) in addition to the per-solve
                # memory bus.
                "cross_run": {"type": "boolean"},
            },
        },
    ),
    ToolDefinition(
        name="update_task_list",
        description=("Persist a concise task list that will remain visible in later iterations."),
        parameters={
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["TODO", "IN_PROGRESS", "DONE", "BLOCKED"],
                            },
                        },
                        "required": ["description", "status"],
                    },
                }
            },
            "required": ["tasks"],
        },
    ),
    ToolDefinition(
        name="checkpoint_state",
        description=(
            "Save the current state of your work as a checkpoint so you can restore it "
            "later if a new approach fails."
        ),
        parameters={
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "action_taken": {"type": "string"},
                "confidence": {"type": "number"},
                "test_pass_count": {"type": "integer"},
                "test_fail_count": {"type": "integer"},
            },
            "required": ["summary", "action_taken"],
        },
    ),
    ToolDefinition(
        name="backtrack_to",
        description=(
            "Restore the codebase to a previous checkpoint when the current approach is "
            "not working."
        ),
        parameters={
            "type": "object",
            "properties": {
                "checkpoint_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["checkpoint_id"],
        },
    ),
    ToolDefinition(
        name="planning",
        description=(
            "Record a concise plan or hypothesis. This has no side effects beyond "
            "storing the note for the trajectory."
        ),
        parameters={
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
            },
            "required": ["thought"],
        },
    ),
]

_SUBAGENT_TOOL_NAMES = frozenset({"investigate", "invoke_debugger", "delegate_subtasks"})
# WS3F: mutation / side-effecting tools denied during a read-only explore phase
# (the localizer should ground itself without editing). Two-layer enforcement:
# (1) omit them from the advertised tool set, (2) hard-deny them in execute().
_READ_ONLY_DENIED_TOOL_NAMES = frozenset(
    {"edit_file", "bash", "run_test_on_patch", "invoke_debugger", "delegate_subtasks"}
)


def build_agent_tool_definitions(
    *,
    enable_subagents: bool = True,
    enable_delegate_subtasks: bool = True,
    enable_project_doc_search: bool = False,
    enable_external_search: bool = False,
    read_only: bool = False,
) -> list[ToolDefinition]:
    tools: list[ToolDefinition] = []
    for tool in BASE_TOOL_DEFINITIONS:
        if read_only and tool.name in _READ_ONLY_DENIED_TOOL_NAMES:
            continue
        if not enable_subagents and tool.name in _SUBAGENT_TOOL_NAMES:
            continue
        if enable_subagents and not enable_delegate_subtasks and tool.name == "delegate_subtasks":
            continue
        if not enable_project_doc_search and tool.name == "search_project_docs":
            continue
        if not enable_external_search and tool.name == "search_web_evidence":
            continue
        tools.append(tool)
    return list(tools)


def make_submit_reproduction_tool() -> ToolDefinition:
    return ToolDefinition(
        name="submit_reproduction",
        description=(
            "Submit the reproduction artifact after you have written and run it. "
            "The command should pass once the bug is fixed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "command": {"type": "string"},
                "script_path": {"type": "string"},
                "script_content": {"type": "string"},
                "observed_output": {"type": "string"},
            },
            "required": ["summary"],
        },
    )


def make_submit_localization_tool() -> ToolDefinition:
    return ToolDefinition(
        name="submit_localization",
        description="Submit the files, symbols, and hypotheses most relevant to the fix.",
        parameters={
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "files": {"type": "array", "items": {"type": "string"}},
                "symbols": {"type": "array", "items": {"type": "string"}},
                "hypotheses": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary"],
        },
    )


def make_submit_patch_tool() -> ToolDefinition:
    return ToolDefinition(
        name="submit_patch",
        description=(
            "Submit the final patch summary after editing files and running verification."
        ),
        parameters={
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "tests_run": {"type": "array", "items": {"type": "string"}},
                "changed_files": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
                "followups": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary"],
        },
    )


def make_submit_test_suite_tool() -> ToolDefinition:
    test_artifact_schema = {
        "type": "object",
        "properties": {
            "artifact_id": {"type": "string"},
            "path": {"type": "string"},
            "content": {"type": "string"},
            "framework": {"type": "string"},
            "language": {"type": "string"},
            "strategy": {"type": "string"},
            "summary": {"type": "string"},
            "test_descriptions": {"type": "array", "items": {"type": "string"}},
            "focus_files": {"type": "array", "items": {"type": "string"}},
            "focus_tests": {"type": "array", "items": {"type": "string"}},
            "contract_sources": {"type": "array", "items": {"type": "string"}},
            "contract_targets": {"type": "array", "items": {"type": "string"}},
            "contract_axes": {"type": "array", "items": {"type": "string"}},
            "justification": {"type": "string"},
            "materialization_mode": {"type": "string"},
            "generator_role": {"type": "string"},
            "generator_vendor": {"type": "string"},
            "adjudicator_vendor": {"type": "string"},
            "reference_targets": {"type": "array", "items": {"type": "string"}},
            "properties": {"type": "array", "items": {"type": "string"}},
            "metamorphic_relations": {"type": "array", "items": {"type": "string"}},
            "fuzz_seeds": {"type": "array", "items": {"type": "string"}},
            "coverage_signal": {"type": "number"},
            "mutation_signal": {"type": "number"},
            "flake_signal": {"type": "number"},
            "patch_overfit_risk": {"type": "number"},
            "milestone_id": {"type": "string"},
            "objective_id": {"type": "string"},
            "objective": {"type": "string"},
            "acceptance_requirements": {"type": "array", "items": {"type": "string"}},
            "interface_specification": {"type": "array", "items": {"type": "string"}},
            "oracle_origin": {"type": "string"},
            "pass_then_invert": {
                "type": "object",
                "properties": {
                    "attempted": {"type": "boolean"},
                    "status": {"type": "string"},
                    "passing_variant_summary": {"type": "string"},
                    "inversion_summary": {"type": "string"},
                    "execution_feedback_summary": {"type": "string"},
                },
            },
            "dual_version_verified": {"type": "boolean"},
            "objective_status": {"type": "string"},
            "promotion_status": {"type": "string"},
        },
        "required": ["path", "content", "strategy"],
    }
    return ToolDefinition(
        name="submit_test_suite",
        description=(
            "Submit a repository-native synthetic test portfolio for cross-validation "
            "and public-signal evaluation."
        ),
        parameters={
            "type": "object",
            "properties": {
                "framework": {"type": "string"},
                "language": {"type": "string"},
                "test_code": {"type": "string"},
                "test_descriptions": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string"},
                "portfolio_summary": {"type": "string"},
                "promotion_summary": {"type": "string"},
                "contract_hypotheses": {"type": "array", "items": {"type": "string"}},
                "reference_targets": {"type": "array", "items": {"type": "string"}},
                "task_contract": {"type": "object"},
                "milestones": {"type": "array", "items": {"type": "object"}},
                "test_objectives": {"type": "array", "items": {"type": "object"}},
                "regression_suite_summary": {"type": "object"},
                "minimization_summary": {"type": "object"},
                "test_artifacts": {"type": "array", "items": test_artifact_schema},
                # Phase G.3: structured adversarial edge predictions.
                # The agent enumerates the bug's likely edge surfaces
                # BEFORE writing tests so it can target them explicitly.
                # The post-iteration feedback loop counts predicted vs.
                # exercised edges and nudges the agent to fill gaps.
                "predicted_edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "edge_type": {
                                "type": "string",
                                "enum": [
                                    "boundary",
                                    "off_by_one",
                                    "null_vs_empty",
                                    "return_type",
                                    "exception_path",
                                    "ordering",
                                    "encoding",
                                    "concurrency",
                                    "other",
                                ],
                            },
                            "location": {"type": "string"},
                            "rationale": {"type": "string"},
                            "test_artifact_paths": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["edge_type", "rationale"],
                    },
                },
            },
            "anyOf": [
                {"required": ["test_code"]},
                {"required": ["test_artifacts"]},
            ],
        },
    )


DELEGATED_WORKER_SYSTEM_PROMPT = (
    "You are a delegated APEX worker operating in an isolated child workspace. "
    "Resolve the assigned subtask directly with the available tools, run focused "
    "verification when useful, and submit a concise factual result through "
    "submit_delegate_result. Do not create new child teams; APEX controls "
    "delegation at the orchestrator level."
)


_SUBMIT_DELEGATED_RESULT_TOOL = ToolDefinition(
    name="submit_delegate_result",
    description="Submit the final result for one delegated child subtask.",
    parameters={
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "tests_run": {"type": "array", "items": {"type": "string"}},
            "changed_files": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
            "followups": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary"],
    },
)


@dataclass
class _DelegatedSubtask:
    task_id: str
    title: str
    goal: str
    focus_files: list[str] = field(default_factory=list)
    forbidden_files: list[str] = field(default_factory=list)
    interface_symbols: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    escalation_triggers: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    max_iterations: Optional[int] = None


@dataclass
class _DelegatedSubtaskResult:
    task_id: str
    title: str
    success: bool
    summary: str
    changed_files: list[str] = field(default_factory=list)
    tests_run: list[str] = field(default_factory=list)
    confidence: float = 0.0
    followups: list[str] = field(default_factory=list)
    workspace_path: str = ""
    patch_path: Optional[str] = None
    patch_preview: str = ""
    lineage: list[str] = field(default_factory=list)
    dependency_notes: list[str] = field(default_factory=list)
    error: str = ""
    tokens_used: int = 0


def _clone_child_llm(parent_llm: Any) -> Any:
    if isinstance(parent_llm, LLMClient):
        return LLMClient(parent_llm.config, temperature_override=parent_llm.temperature)
    return parent_llm


def _truncate_chars(text: str, max_chars: int) -> str:
    content = (text or "").strip()
    if len(content) <= max_chars:
        return content
    suffix = "\n[... truncated ...]"
    return content[: max(0, max_chars - len(suffix))].rstrip() + suffix


class _SymbolFinder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.results: list[tuple[str, str, int, Optional[str]]] = []
        self.class_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.results.append(
            (node.name, "class", node.lineno, self.class_stack[-1] if self.class_stack else None)
        )
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        kind = "method" if self.class_stack else "function"
        self.results.append(
            (node.name, kind, node.lineno, self.class_stack[-1] if self.class_stack else None)
        )
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        kind = "method" if self.class_stack else "function"
        self.results.append(
            (node.name, kind, node.lineno, self.class_stack[-1] if self.class_stack else None)
        )
        self.generic_visit(node)


class ACIToolExecutor:
    """Execute ACI tools inside one isolated rollout workspace."""

    def __init__(
        self,
        working_dir: str,
        config: ACIConfig,
        agentic_search_config: Any = None,
        repo_context: Optional[RepoContext] = None,
        memory_bus: Any = None,
        rollout_id: Optional[int] = None,
        execution_tree: Any = None,
        baseline_commit: Optional[str] = None,
        test_command: Optional[str] = None,
        test_timeout: Optional[int] = None,
        agent_depth: int = 0,
        agent_lineage: Optional[list[str]] = None,
        episodic_store: Any = None,
        task_signature: Optional[str] = None,
    ):
        self.working_dir = Path(working_dir).resolve()
        self.config = config
        self.agentic_search_config = agentic_search_config
        self.repo_context = repo_context
        self.memory_bus = memory_bus
        self.rollout_id = rollout_id
        self.execution_tree = execution_tree
        # Phase 6.2: optional persistent cross-run episodic store. When
        # set together with ``task_signature``, ``broadcast_discovery``
        # and ``query_discoveries`` accept ``cross_run=True`` to route
        # through this store instead of the in-memory ``memory_bus``.
        # Both must be present for cross-run mode to engage; if only
        # one is set we degrade gracefully (warning + per-solve only).
        self.episodic_store = episodic_store
        self.task_signature = task_signature
        self.baseline_commit = baseline_commit
        self.test_command = test_command
        self.test_timeout = max(int(test_timeout or config.bash_timeout or 30), 30)
        self._shell_cwd = self.working_dir
        # Strip host secrets before exposing the LLM bash tool to ambient env.
        # An attacker-controlled repo or prompt could otherwise instruct the
        # LLM to exfiltrate ANTHROPIC_API_KEY / AWS_* / GH_* via the bash tool.
        scrubbed_env, _ = redact_host_secrets(os.environ.copy())
        self._bash_env = scrubbed_env
        runtime_env_overrides = {
            str(key): str(value)
            for key, value in dict(getattr(config, "runtime_env_overrides", {}) or {}).items()
        }
        self._bash_env.update(runtime_env_overrides)
        self._bash_env["HOME"] = str(self.working_dir)
        self._bash_env["PYTHONDONTWRITEBYTECODE"] = "1"
        if self._bash_env.get("APEX_TARGET_TOOL_CONTEXT"):
            self._bash_env["APEX_TARGET_TOOL_WORKDIR"] = str(self.working_dir)
        self._target_tool_context = self._load_target_tool_context(
            self._bash_env.get("APEX_TARGET_TOOL_CONTEXT")
        )
        self._planning_log: list[str] = []
        self._task_list: list[dict[str, str]] = []
        self._task_list_version = 0
        self._last_task_list_injection_version = -1
        self._last_discovery_check = 0.0
        self._agent_runtime: Any = None
        self.agent_depth = max(int(agent_depth), 0)
        self.agent_lineage = list(agent_lineage or ["root"])
        team_dirname = Path(config.agent_team_workspace_dirname or ".apex_agent_teams").name
        self._team_workspace_root = self.working_dir / (team_dirname or ".apex_agent_teams")
        self._discovery_scope: dict[str, Any] = {
            "stage_name": "",
            "file_paths": [],
            "symbols": [],
            "test_ids": [],
        }
        self._delegation_plan_tasks: list[_DelegatedSubtask] = []
        self._delegation_plan_lookup: dict[str, _DelegatedSubtask] = {}
        self._delegation_plan_order: list[str] = []
        self._delegation_plan_aliases: dict[str, Optional[str]] = {}
        self._delegation_plan_parallelism: int = 1
        self._write_scope_enforced: bool = False
        self._write_scope_allowed_paths: set[str] = set()
        self._write_scope_forbidden_paths: set[str] = set()
        # TIER 2 (T2.3): sticky enforced module-group write scope. When set, any
        # changed file NOT in the owned allow-list (and not a protected/harness
        # path) is treated as off-group and reverted by ``_enforce_write_scope``.
        # Sticky so a later non-decomposition ``set_write_scope`` call (e.g. the
        # solver's default advisory scope) cannot silently disarm it.
        self._module_group_write_scope_enforced: bool = False
        self._module_group_owned_paths: set[str] = set()
        self._read_only_explore: bool = False  # WS3F read-only explore gate
        # Phase 2C 2.7: localizer-constraint plumbing. Default
        # ``advisory`` so legacy callers see no behaviour change; agents
        # opt-in via ``set_localizer_constraint``.
        self._localizer_files: list[str] = []
        self._localizer_enforcement: str = "advisory"
        self._localizer_allowlist_files: list[str] = []
        self._localizer_allowlist_globs: list[str] = ["tests/**", "test/**"]
        self._localizer_off_target_patches: int = 0
        self._localizer_off_target_files: list[str] = []
        self._localizer_excluded_artifact_files: list[str] = []
        self._localizer_noneditable_context_files: list[str] = []
        self._localizer_scope_class: str = "unknown"
        self._localizer_severity: str = "none"
        # Phase 5.5: caches store (insert_monotonic_seconds, payload) tuples
        # so we can apply a TTL on read. Cache hits also need to be
        # invalidated whenever the agent edits a file (an edit is implicit
        # evidence the cached doc snippet is stale). See
        # ``_cache_invalidate_on_edit`` and ``_cache_lookup``.
        self._project_doc_cache: dict[tuple[str, int], tuple[float, str]] = {}
        self._external_search_cache: dict[tuple[str, str, int], tuple[float, str]] = {}
        configured_budget = getattr(
            self.agentic_search_config,
            "external_search_budget",
            0,
        )
        try:
            self._external_search_budget_total = max(0, int(configured_budget or 0))
        except (TypeError, ValueError):
            self._external_search_budget_total = 0
        self._external_search_budget_used = 0

    def set_test_command(self, test_command: Optional[str]) -> None:
        self.test_command = test_command

    def set_agent_runtime(self, runtime: Any | None) -> None:
        self._agent_runtime = runtime

    def get_agent_runtime(self) -> Any | None:
        return self._agent_runtime

    @staticmethod
    def _load_target_tool_context(context_path: Optional[str]) -> dict[str, Any]:
        if not context_path:
            return {}
        try:
            payload = json.loads(Path(context_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    def _target_shell_cwd(self) -> str:
        context = self._target_tool_context
        runtime = dict(context.get("runtime") or {})
        if str(runtime.get("kind") or "") != "docker_image":
            return str(self._shell_cwd)
        host_workdir = Path(context.get("workdir") or self.working_dir).expanduser().resolve()
        docker_workdir = (
            str(runtime.get("docker_workdir") or "/workspace").rstrip("/") or "/workspace"
        )
        try:
            relative = self._shell_cwd.resolve().relative_to(host_workdir)
        except ValueError:
            return docker_workdir
        if not relative.parts:
            return docker_workdir
        return posixpath.join(docker_workdir, *relative.parts)

    def _host_shell_cwd_from_target(self, target_cwd: str) -> Path:
        context = self._target_tool_context
        runtime = dict(context.get("runtime") or {})
        if str(runtime.get("kind") or "") == "docker_image":
            host_workdir = Path(context.get("workdir") or self.working_dir).expanduser().resolve()
            docker_workdir = (
                str(runtime.get("docker_workdir") or "/workspace").rstrip("/") or "/workspace"
            )
            normalized = posixpath.normpath(str(target_cwd or docker_workdir))
            if normalized == docker_workdir:
                return host_workdir
            prefix = docker_workdir + "/"
            if normalized.startswith(prefix):
                return (host_workdir / Path(*normalized[len(prefix) :].split("/"))).resolve()
            return host_workdir
        try:
            return Path(target_cwd).expanduser().resolve()
        except OSError:
            return self.working_dir

    def set_discovery_scope(
        self,
        *,
        stage_name: str = "",
        file_paths: Optional[list[str]] = None,
        symbols: Optional[list[str]] = None,
        test_ids: Optional[list[str]] = None,
    ) -> None:
        self._discovery_scope = {
            "stage_name": (stage_name or "").strip(),
            "file_paths": list(
                dict.fromkeys(item.strip() for item in (file_paths or []) if item and item.strip())
            ),
            "symbols": list(
                dict.fromkeys(item.strip() for item in (symbols or []) if item and item.strip())
            ),
            "test_ids": list(
                dict.fromkeys(item.strip() for item in (test_ids or []) if item and item.strip())
            ),
        }

    def set_delegation_plan(
        self,
        tasks: Optional[list[dict[str, Any]]] = None,
        *,
        parallelism: int = 1,
        max_iterations: Optional[int] = None,
    ) -> None:
        normalized_tasks, alias_map = self._canonicalize_planned_delegation_tasks(
            tasks or [],
            default_max_iterations=max_iterations,
        )
        self._delegation_plan_tasks = normalized_tasks
        self._delegation_plan_lookup = {task.task_id: task for task in normalized_tasks}
        self._delegation_plan_order = [task.task_id for task in normalized_tasks]
        self._delegation_plan_aliases = alias_map
        if normalized_tasks:
            try:
                requested_parallelism = int(parallelism or 1)
            except (TypeError, ValueError):
                requested_parallelism = 1
            self._delegation_plan_parallelism = max(
                1,
                min(
                    requested_parallelism,
                    self.config.max_agent_team_parallelism,
                    len(normalized_tasks),
                ),
            )
        else:
            self._delegation_plan_parallelism = 1

    def set_read_only(self, enabled: bool) -> None:
        """WS3F: engage/clear the tool-enforced read-only explore gate. When on,
        execute() hard-denies mutation tools regardless of what was advertised."""
        self._read_only_explore = bool(enabled)

    def set_write_scope(
        self,
        allowed_files: Optional[list[str]] = None,
        forbidden_files: Optional[list[str]] = None,
        *,
        enforce: bool = False,
    ) -> None:
        self._write_scope_enforced = bool(enforce)
        self._write_scope_allowed_paths = {
            normalized
            for item in list(allowed_files or [])
            if (normalized := self._normalize_workspace_rel_path(item))
        }
        self._write_scope_forbidden_paths = {
            normalized
            for item in list(forbidden_files or [])
            if (normalized := self._normalize_workspace_rel_path(item))
        }

    def set_module_group_write_scope(
        self,
        owned_files: Optional[list[str]] = None,
        *,
        enforce: bool = True,
    ) -> None:
        """Install an ENFORCED module-group write scope (TIER 2, T2.3).

        ``owned_files`` is the group's allow-list; any changed file outside it
        (and outside the protected/harness set) is off-group and reverted by
        ``_enforce_write_scope``. This is allow-list (not forbidden-list)
        semantics, so the group's edits are confined without enumerating every
        off-group file. Sticky: it survives later ``set_write_scope`` calls.
        Call with ``enforce=False`` / empty list to disarm.
        """
        owned = {
            normalized
            for item in list(owned_files or [])
            if (normalized := self._normalize_workspace_rel_path(item))
        }
        self._module_group_write_scope_enforced = bool(enforce) and bool(owned)
        self._module_group_owned_paths = owned if self._module_group_write_scope_enforced else set()

    def set_localizer_constraint(
        self,
        *,
        files: Optional[list[str]] = None,
        enforcement: str = "advisory",
        allowlist_files: Optional[list[str]] = None,
        allowlist_globs: Optional[list[str]] = None,
    ) -> None:
        """Phase 2C 2.7: install the localizer's file scope.

        ``files`` is the localizer's submitted file list (the agent's
        own narrowed scope). ``enforcement`` is one of:

          * ``"advisory"``       — record but don't act on it.
          * ``"warning"``        — count off-target patches into
                                   diagnostics, but accept them.
          * ``"hard_constraint"``— legacy spelling for a high-severity
                                   diagnostic. Localization is evidence, not
                                   a validity boundary, so off-scope source
                                   diffs are recorded and left for objective
                                   verification rather than rejected here.

        Stored verbatim; ``validate_patch_submission`` interprets them.
        """
        focus = split_localizer_focus(files or [])
        self._localizer_files = [
            normalize_changed_path(str(item).strip())
            for item in focus.editable_focus_files
            if str(item).strip()
        ]
        self._localizer_noneditable_context_files = list(focus.noneditable_context_files)
        mode = str(enforcement or "").strip().lower()
        if mode not in {"advisory", "warning", "hard_constraint"}:
            mode = "warning"
        self._localizer_enforcement = mode
        self._localizer_allowlist_files = [
            normalize_changed_path(str(item).strip())
            for item in list(allowlist_files or [])
            if str(item).strip()
        ]
        # Globs are stored as raw fnmatch patterns; tests/** is appended
        # as a default so test edits never trip the constraint.
        raw_globs = [str(item).strip() for item in list(allowlist_globs or []) if str(item).strip()]
        if "tests/**" not in raw_globs:
            raw_globs.append("tests/**")
        if "test/**" not in raw_globs:
            raw_globs.append("test/**")
        self._localizer_allowlist_globs = raw_globs
        # Reset the per-rollout off-target counter when the constraint
        # is reinstalled (one rollout = one constraint lifetime).
        self._localizer_off_target_patches = 0
        self._localizer_off_target_files: list[str] = []
        self._localizer_excluded_artifact_files = []
        self._localizer_scope_class = infer_scope_class(
            editable_focus_files=self._localizer_files,
            solution_changed_files=[],
        )
        self._localizer_severity = "none"

    def localizer_diagnostics(self) -> dict[str, Any]:
        """Phase 2C 2.7: snapshot of localizer-enforcement counters.

        Returned by ``PatcherAgent`` and merged into ``AgentResult.diagnostics``
        so the engine / orchestrator can persist
        ``off_target_patches`` and the offending file set.
        """
        return {
            "localizer_enforcement": getattr(self, "_localizer_enforcement", "advisory"),
            "localizer_files": list(getattr(self, "_localizer_files", []) or []),
            "off_target_patches": int(getattr(self, "_localizer_off_target_patches", 0) or 0),
            "off_target_files": list(getattr(self, "_localizer_off_target_files", []) or []),
            "excluded_artifact_files": list(
                getattr(self, "_localizer_excluded_artifact_files", []) or []
            ),
            "noneditable_context_files": list(
                getattr(self, "_localizer_noneditable_context_files", []) or []
            ),
            "scope_class": getattr(self, "_localizer_scope_class", "unknown"),
            "severity": getattr(self, "_localizer_severity", "none"),
        }

    def _path_within_localizer_scope(self, rel_path: str) -> bool:
        """True iff ``rel_path`` is allowed under the configured localizer
        constraint."""
        normalized = normalize_changed_path(rel_path)
        if not normalized:
            return True
        files = getattr(self, "_localizer_files", []) or []
        # Localizer-listed files (exact or directory-prefixed) are in scope.
        for scope in files:
            if not scope:
                continue
            if self._scope_path_matches(normalized, scope):
                return True
        allowlist_files = getattr(self, "_localizer_allowlist_files", []) or []
        for allowed in allowlist_files:
            if not allowed:
                continue
            if self._scope_path_matches(normalized, allowed):
                return True
        allowlist_globs = getattr(self, "_localizer_allowlist_globs", []) or []
        from fnmatch import fnmatch as _fnmatch

        for glob in allowlist_globs:
            if _fnmatch(normalized, glob):
                return True
        return False

    def validate_patch_submission(
        self,
        *,
        changed_files: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Phase 2C 2.7: run the localizer-constraint check on a diff.

        ``changed_files`` may be supplied by the caller (cheap when
        already computed) or left None to trigger a fresh ``git status``
        inside the workspace. Returns:

            {
                "violation": bool,         # off-target files touched
                "off_target_files": list,  # the offending paths
                "rejected": bool,          # reserved; localizer scope alone is non-dropping
                "enforcement": str,        # the active mode
            }

        ``violation=True`` means record the off-scope files and continue.
        Protected-file and harness gates enforce harmful edits elsewhere.
        """
        enforcement = getattr(self, "_localizer_enforcement", "advisory")
        files = getattr(self, "_localizer_files", None)
        # Advisory or no constraint → no-op.
        if enforcement == "advisory" or files is None:
            return {
                "violation": False,
                "off_target_files": [],
                "rejected": False,
                "enforcement": enforcement,
            }
        if changed_files is None:
            try:
                changed_files = list_git_changed_files(self.working_dir)
            except Exception as exc:  # noqa: BLE001 - never fail validation here
                _security_logger.warning(
                    "validate_patch_submission: list_git_changed_files raised "
                    "%s: %s; treating as no off-target diff",
                    type(exc).__name__,
                    exc,
                )
                changed_files = []
        normalized_changed = [normalize_changed_path(p) for p in changed_files if p]
        manifest = build_patch_manifest(normalized_changed)
        self._localizer_excluded_artifact_files = list(manifest.excluded_files)
        solution_changed = filter_solution_paths(normalized_changed)
        off_target = [
            path
            for path in solution_changed
            if path and not self._path_within_localizer_scope(path)
        ]
        scope_class = infer_scope_class(
            editable_focus_files=getattr(self, "_localizer_files", []) or [],
            solution_changed_files=solution_changed,
        )
        severity = localizer_severity(
            scope_class=scope_class,
            out_of_scope_solution_files=off_target,
        )
        self._localizer_scope_class = scope_class
        self._localizer_severity = severity
        if off_target:
            self._localizer_off_target_patches = (
                int(getattr(self, "_localizer_off_target_patches", 0) or 0) + 1
            )
            existing = list(getattr(self, "_localizer_off_target_files", []) or [])
            for path in off_target:
                if path not in existing:
                    existing.append(path)
            self._localizer_off_target_files = existing
        result = {
            "violation": bool(off_target),
            "off_target_files": list(off_target),
            "excluded_artifact_files": list(manifest.excluded_files),
            "solution_files": list(manifest.solution_files),
            "patch_manifest": manifest.to_dict(),
            "editable_focus_files": list(getattr(self, "_localizer_files", []) or []),
            "noneditable_context_files": list(
                getattr(self, "_localizer_noneditable_context_files", []) or []
            ),
            "scope_class": scope_class,
            "severity": severity,
            "rejected": False,
            "enforcement": enforcement,
        }
        if off_target and enforcement == "warning":
            _security_logger.warning(
                "localizer_constraint_warning: patch touched %s out-of-scope "
                "file(s) under enforcement=warning: %s",
                len(off_target),
                off_target,
            )
        elif off_target and enforcement == "hard_constraint":
            _security_logger.warning(
                "localizer_constraint_violation: patch touched %s out-of-scope "
                "file(s) under enforcement=hard_constraint; recording diagnostic. "
                "Files: %s",
                len(off_target),
                off_target,
            )
        return result

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if tool_name.startswith("submit_"):
            return self._submit(tool_name, arguments)

        # WS3F: server-side read-only enforcement (independent of which tools were
        # advertised) — the localizer's explore phase cannot mutate the workspace.
        if getattr(self, "_read_only_explore", False) and tool_name in _READ_ONLY_DENIED_TOOL_NAMES:
            return (
                f"READ-ONLY EXPLORE: the '{tool_name}' tool is disabled during the "
                "localization/explore phase. Use view_file / search_files / find_symbols / "
                "lookup_definition / trace_callers to ground yourself, then submit your findings."
            )

        handlers = {
            "view_file": self._view_file,
            "search_files": self._search_files,
            "search_project_docs": self._search_project_docs,
            "search_web_evidence": self._search_web_evidence,
            "find_symbols": self._find_symbols,
            "lookup_definition": self._lookup_definition,
            "trace_callers": self._trace_callers,
            "trace_callees": self._trace_callees,
            "get_entity_context": self._get_entity_context,
            "edit_file": self._edit_file,
            "bash": self._bash,
            "run_test_on_patch": self._run_test_on_patch,
            "invoke_debugger": self._invoke_debugger,
            "investigate": self._investigate,
            "delegate_subtasks": self._delegate_subtasks,
            "broadcast_discovery": self._broadcast_discovery,
            "query_discoveries": self._query_discoveries,
            "update_task_list": self._update_task_list,
            "checkpoint_state": self._checkpoint_state,
            "backtrack_to": self._backtrack_to,
            "planning": self._planning,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return f"Error: Unknown tool '{tool_name}'."
        try:
            return handler(**arguments)
        except Exception as exc:  # pragma: no cover - defensive
            return f"Error executing {tool_name}: {exc}"

    def render_dynamic_context(self, iteration: int) -> list[str]:
        messages: list[str] = []
        task_text = self.render_task_list()
        should_refresh_task_list = bool(task_text) and (
            iteration == 1
            or iteration % 5 == 0
            or self._task_list_version != self._last_task_list_injection_version
        )
        if should_refresh_task_list:
            messages.append(f"[Task List Refresh {iteration}]\n{task_text}")
            self._last_task_list_injection_version = self._task_list_version

        if (
            self.memory_bus is not None
            and self.rollout_id is not None
            and iteration > 0
            and iteration % 5 == 0
        ):
            discoveries = self.memory_bus.format_for_context(
                exclude_rollout_id=self.rollout_id,
                since_timestamp=self._last_discovery_check,
                stage_name=self._discovery_scope["stage_name"],
                file_paths=self._discovery_scope["file_paths"],
                symbols=self._discovery_scope["symbols"],
                test_ids=self._discovery_scope["test_ids"],
            )
            if discoveries:
                messages.append(discoveries)
                self._last_discovery_check = time.time()

        return messages

    def render_task_list(self) -> str:
        if not self._task_list:
            return ""
        lines = ["## Task List"]
        for item in self._task_list:
            lines.append(f"- [{item['status']}] {item['description']}")
        return "\n".join(lines)

    def _resolve_path(self, path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.working_dir / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.working_dir)
        except ValueError as exc:
            raise ValueError(f"Path '{path}' escapes the rollout workspace.") from exc
        return resolved

    @staticmethod
    def _looks_like_shell_env_assignment(token: str) -> bool:
        return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", str(token or "")))

    def _tokenize_shell_segment(self, segment: str) -> list[str]:
        raw = str(segment or "").strip()
        if not raw:
            return []
        try:
            tokens = shlex.split(raw, posix=True)
        except ValueError:
            return []

        index = 0
        while index < len(tokens) and self._looks_like_shell_env_assignment(tokens[index]):
            index += 1

        if index < len(tokens) and tokens[index] == "env":
            index += 1
            while index < len(tokens):
                token = tokens[index]
                if token == "--":
                    index += 1
                    break
                if self._looks_like_shell_env_assignment(token):
                    index += 1
                    continue
                if token.startswith("-"):
                    index += 1
                    if token == "-u" and index < len(tokens):
                        index += 1
                    continue
                break

        while index < len(tokens):
            command_name = Path(tokens[index]).name
            if command_name == "command":
                index += 1
                continue
            if command_name in {"timeout", "gtimeout"}:
                index += 1
                while index < len(tokens):
                    token = tokens[index]
                    if token == "--":
                        index += 1
                        break
                    if token.startswith("-"):
                        index += 1
                        if token in {"-k", "--kill-after", "-s", "--signal"} and index < len(
                            tokens
                        ):
                            index += 1
                        continue
                    index += 1
                    break
                continue
            break
        return tokens[index:]

    def _resolve_shell_path_token(self, token: str) -> Optional[Path]:
        raw = str(token or "").strip()
        if not raw or raw == "--":
            return None

        expanded = raw
        if raw.startswith("${PWD}"):
            expanded = str(self._shell_cwd) + raw[len("${PWD}") :]
        elif raw.startswith("$PWD"):
            expanded = str(self._shell_cwd) + raw[len("$PWD") :]
        elif raw.startswith("${HOME}"):
            expanded = str(self.working_dir) + raw[len("${HOME}") :]
        elif raw.startswith("$HOME"):
            expanded = str(self.working_dir) + raw[len("$HOME") :]
        elif raw == "~" or raw.startswith("~/"):
            expanded = str(self.working_dir) + raw[1:]
        elif raw == "." or raw.startswith("./") or raw == ".." or raw.startswith("../"):
            expanded = str(self._shell_cwd / raw)
        elif not raw.startswith("/"):
            return None

        try:
            return Path(expanded).resolve()
        except OSError:
            return None

    def _path_escapes_workspace(self, candidate: Path) -> bool:
        try:
            candidate.resolve().relative_to(self.working_dir)
        except ValueError:
            return True
        return False

    def _bash_workspace_escape_error(self, command: str) -> Optional[str]:
        for segment in re.split(r"(?:&&|\|\||;|\n)", str(command or "")):
            tokens = self._tokenize_shell_segment(segment)
            if not tokens:
                continue
            command_name = Path(tokens[0]).name

            if command_name in {"cd", "pushd"}:
                if len(tokens) < 2:
                    continue
                resolved = self._resolve_shell_path_token(tokens[1])
                if resolved is not None and self._path_escapes_workspace(resolved):
                    return (
                        "BASH REJECTED - keep shell exploration inside the rollout workspace. "
                        f"`{command_name} {tokens[1]}` resolves outside the workspace. "
                        "Use relative workspace paths or the structured search tools instead."
                    )
                continue

            if command_name not in {"find", "fd", "fdfind", "tree", "du", "ls"}:
                continue
            for token in tokens[1:]:
                if token == "--" or token.startswith("-"):
                    continue
                resolved = self._resolve_shell_path_token(token)
                if resolved is None:
                    continue
                if self._path_escapes_workspace(resolved):
                    return (
                        "BASH REJECTED - keep repository discovery inside the rollout workspace. "
                        f"`{command_name}` targeted `{token}`, which resolves outside the workspace. "
                        "Use relative workspace paths or the structured search tools instead."
                    )
        return None

    def _normalize_workspace_rel_path(self, path: str | Path) -> str:
        raw = str(path or "").strip()
        if not raw:
            return ""
        resolved = self._resolve_path(raw)
        return normalize_changed_path(str(resolved.relative_to(self.working_dir)))

    def _scope_path_matches(self, rel_path: str, scope_path: str) -> bool:
        return rel_path == scope_path or rel_path.startswith(f"{scope_path}/")

    def _path_is_write_allowed(self, rel_path: str) -> bool:
        normalized = normalize_changed_path(rel_path)
        if not normalized:
            return True
        if any(
            self._scope_path_matches(normalized, forbidden)
            for forbidden in self._write_scope_forbidden_paths
        ):
            return False
        # TIER 2 (T2.3) allow-list semantics: when a module-group scope is
        # enforced, anything outside the owned allow-list is off-group and
        # disallowed — EXCEPT test files and apex harness files, which the
        # agent legitimately edits to demonstrate / score the fix.
        if self._module_group_write_scope_enforced and self._module_group_owned_paths:
            if is_test_path(normalized) or is_apex_harness_path(normalized):
                return True
            if not any(
                self._scope_path_matches(normalized, owned)
                for owned in self._module_group_owned_paths
            ):
                return False
        return True

    def _write_scope_error(self, path: str) -> Optional[str]:
        rel_path = self._normalize_workspace_rel_path(path)
        if self._path_is_write_allowed(rel_path):
            return None
        if any(
            self._scope_path_matches(rel_path, forbidden)
            for forbidden in self._write_scope_forbidden_paths
        ):
            return (
                "WRITE REJECTED - delegated scope forbids edits to "
                f"{rel_path}. Choose an implementation path outside forbidden or "
                "protected files, or report why this hard boundary blocks the fix."
            )
        return None

    def _restore_paths_to_head(self, paths: list[str]) -> list[str]:
        errors: list[str] = []
        normalized_paths = [
            rel_path
            for rel_path in list(
                dict.fromkeys(normalize_changed_path(path) for path in paths if path)
            )
            if rel_path
        ]
        if not normalized_paths:
            return errors

        tracked_paths: set[str] = set()
        # Large module-group repairs can touch thousands of off-scope data files.
        # Batch the tracked-file probe and restore so enforcement is O(chunks),
        # not O(files), while preserving the same tracked-vs-untracked behavior.
        chunk_size = 256
        for index in range(0, len(normalized_paths), chunk_size):
            batch = normalized_paths[index : index + chunk_size]
            tracked_result = subprocess.run(
                ["git", "ls-files", "-z", "--", *batch],
                capture_output=True,
                text=True,
                cwd=str(self.working_dir),
                check=False,
            )
            if tracked_result.returncode != 0:
                errors.append(
                    tracked_result.stderr.strip()
                    or tracked_result.stdout.strip()
                    or "Failed to query tracked files for write-scope restore."
                )
                continue
            tracked_paths.update(
                normalize_changed_path(path)
                for path in tracked_result.stdout.split("\0")
                if normalize_changed_path(path)
            )

        tracked_ordered = [path for path in normalized_paths if path in tracked_paths]
        for index in range(0, len(tracked_ordered), chunk_size):
            batch = tracked_ordered[index : index + chunk_size]
            restore_result = subprocess.run(
                ["git", "checkout", "--", *batch],
                capture_output=True,
                text=True,
                cwd=str(self.working_dir),
                check=False,
            )
            if restore_result.returncode != 0:
                errors.append(
                    restore_result.stderr.strip()
                    or restore_result.stdout.strip()
                    or "Failed to restore tracked write-scope violations."
                )

        for rel_path in normalized_paths:
            if rel_path in tracked_paths:
                continue
            target = self.working_dir / rel_path
            try:
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                elif target.exists() or target.is_symlink():
                    target.unlink()
            except OSError as exc:
                errors.append(f"Failed to remove {rel_path}: {exc}")
        return errors

    def _enforce_write_scope(self) -> tuple[list[str], list[str]]:
        module_group_active = bool(
            self._module_group_write_scope_enforced and self._module_group_owned_paths
        )
        if not self._write_scope_forbidden_paths and not module_group_active:
            return [], []
        changed_files = list_git_changed_files(
            self.working_dir,
            baseline_ref=self.baseline_commit,
        )
        if not changed_files and self.baseline_commit:
            # The baseline diff was empty even though enforcement is active. A
            # stale / post-change baseline_commit makes `git diff <baseline>`
            # return nothing (run #62: a rollout changed 526 files but write-scope
            # reverted 0), and list_changed_files does NOT fall through to status
            # when the diff succeeds-but-empty. The scaffolded patcher's edits are
            # UNCOMMITTED, so the working-tree status (baseline_ref=None) always
            # sees them — use it so a wrong baseline can never silently disable
            # write-scope enforcement. Only triggers on the empty-diff symptom, so
            # the normal (correct-baseline) path is byte-identical.
            changed_files = list_git_changed_files(self.working_dir, baseline_ref=None)
        violations = [
            rel_path for rel_path in changed_files if not self._path_is_write_allowed(rel_path)
        ]
        if not violations:
            return [], []
        restore_errors = self._restore_paths_to_head(violations)
        return sorted(dict.fromkeys(violations)), restore_errors

    # ------------------------------------------------------------------
    # Phase 5.5: cache TTL + invalidation helpers
    # ------------------------------------------------------------------
    def _cache_ttl_seconds(self) -> float:
        try:
            return float(getattr(self.config, "cache_ttl_seconds", 60.0) or 0.0)
        except (TypeError, ValueError):
            return 60.0

    def _cache_lookup(
        self,
        cache: dict[Any, tuple[float, str]],
        key: Any,
    ) -> Optional[str]:
        """Return the cached payload for ``key`` if it exists and is fresh."""
        entry = cache.get(key)
        if entry is None:
            return None
        inserted, payload = entry
        ttl = self._cache_ttl_seconds()
        if ttl > 0 and (time.monotonic() - float(inserted)) > ttl:
            cache.pop(key, None)
            return None
        return payload

    def _cache_store(
        self,
        cache: dict[Any, tuple[float, str]],
        key: Any,
        payload: str,
    ) -> None:
        cache[key] = (time.monotonic(), payload)

    def _cache_invalidate_on_edit(self, edited_rel_path: str) -> None:
        """Invalidate doc-related caches whenever the agent edits a file.

        The project-doc cache scans on-disk markdown / RST files; once the
        agent has touched the workspace, any cached result is suspect. The
        external search cache is kept (web results don't depend on local
        edits) unless a future change ties it to repo state.
        """
        if not edited_rel_path:
            return
        if not bool(getattr(self.config, "cache_invalidate_on_any_edit", True)):
            return
        if self._project_doc_cache:
            self._project_doc_cache.clear()

    def _format_search_failure(
        self,
        *,
        backend: str,
        stderr: str,
        stdout: str,
        returncode: int,
        pattern: str,
    ) -> str:
        """Render a structured ``tool_failure`` message for a search-tool error.

        Phase 5.5: ripgrep / grep returncode>=2 means a real failure (regex
        compile error, IO problem, OOM). Surface stderr to the LLM along
        with the failure_class so callers can distinguish APEX-side
        problems (bad regex) from env-side (binary missing, OOM).
        """
        try:
            verdict = _classify_failure_core(
                stderr=stderr,
                stdout=stdout,
                returncode=int(returncode),
                context={"phase": "test_execution"},
            )
            failure_class = verdict.failure_class.value
        except Exception:  # pragma: no cover - defensive
            failure_class = "unclassified"
        snippet = (stderr or "").strip().splitlines()
        first_line = snippet[0] if snippet else "(no stderr)"
        body = (
            f"tool_failure: search_files via {backend} exited with returncode={returncode}.\n"
            f"failure_class: {failure_class}\n"
            f"pattern: {pattern!r}\n"
            f"stderr (first line): {first_line}"
        )
        return self._truncate_output(body)

    def _view_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: Optional[int] = None,
    ) -> str:
        file_path = self._resolve_path(path)
        if not file_path.exists():
            return f"Error: File '{path}' does not exist."
        if file_path.is_dir():
            return f"Error: '{path}' is a directory."

        lines = file_path.read_text(errors="replace").splitlines()
        total_lines = len(lines)
        start_line = max(start_line, 1)
        if end_line is None:
            end_line = min(start_line + self.config.file_view_lines - 1, total_lines)
        end_line = min(end_line, total_lines)

        output: list[str] = []
        if start_line > 1:
            output.append(f"[{start_line - 1} lines above]")
        for index, line in enumerate(lines[start_line - 1 : end_line], start=start_line):
            output.append(f"{index:>6} | {line}")
        if end_line < total_lines:
            output.append(f"[{total_lines - end_line} lines below]")
        output.append(f"\n({total_lines} lines total in {path})")
        return "\n".join(output)

    def _search_files(
        self,
        pattern: str,
        path: Optional[str] = None,
        file_pattern: Optional[str] = None,
    ) -> str:
        search_dir = self._resolve_path(path) if path else self._shell_cwd
        if not search_dir.exists():
            return f"Error: Directory '{path}' does not exist."

        # Phase 5.5: surface ripgrep/grep stderr as a structured tool_failure
        # rather than absorbing it into a vague "Search failed." string. We
        # also classify the failure so callers can distinguish env (binary
        # missing, OOM) from APEX (bad regex) issues.
        matches: list[str] = []
        backend = "rg" if shutil.which("rg") else "grep"
        if backend == "rg":
            command = ["rg", "--files-with-matches", "--color=never", pattern, str(search_dir)]
            if file_pattern:
                command[1:1] = ["-g", file_pattern]
        else:
            command = ["grep", "-rlE", pattern, str(search_dir)]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=15,
                cwd=str(self._shell_cwd),
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return self._format_search_failure(
                backend=backend,
                stderr=f"{type(exc).__name__}: {exc}",
                stdout="",
                returncode=1,
                pattern=pattern,
            )

        # Both rg and grep use returncode 0 = matches, 1 = no matches.
        # Anything >=2 is a real failure (regex compile error, IO, OOM).
        if result.returncode >= 2:
            return self._format_search_failure(
                backend=backend,
                stderr=result.stderr or "",
                stdout=result.stdout or "",
                returncode=result.returncode,
                pattern=pattern,
            )
        matches = (result.stdout or "").splitlines()

        rel_paths = sorted(
            {
                str(Path(match).resolve().relative_to(self.working_dir))
                for match in matches
                if Path(match).exists()
            }
        )
        if not rel_paths:
            return "No files matched the pattern."

        shown = rel_paths[: self.config.search_max_results]
        message = f"Found matches in {len(rel_paths)} file(s):\n" + "\n".join(shown)
        if len(rel_paths) > len(shown):
            remaining = len(rel_paths) - len(shown)
            message += f"\n... and {remaining} more"
        return message

    def _search_project_docs(
        self,
        query: str,
        max_results: int = 5,
    ) -> str:
        query_text = str(query or "").strip()
        if not query_text:
            return "Error: search_project_docs requires a non-empty query."
        try:
            max_results = max(1, int(max_results))
        except (TypeError, ValueError):
            max_results = 5
        cache_key = (query_text, max_results)
        cached = self._cache_lookup(self._project_doc_cache, cache_key)
        if cached is not None:
            return cached
        doc_paths = [
            self.working_dir / rel_path
            for rel_path in collect_local_reference_files(
                self.working_dir,
                max_files=max(max_results * 3, 8),
            )
        ]
        if not doc_paths:
            return "No local documentation files were found in this workspace."

        query_lower = query_text.lower()
        tokens = [
            token for token in re.split(r"[^a-zA-Z0-9_./:-]+", query_lower) if len(token) >= 3
        ]
        if not tokens and query_lower:
            tokens = [query_lower]

        scored: list[tuple[int, str, list[str]]] = []
        for path in doc_paths:
            rel_path = str(path.relative_to(self.working_dir))
            try:
                lines = path.read_text(errors="replace").splitlines()
            except OSError:
                continue
            score = 0
            matches: list[str] = []
            rel_lower = rel_path.lower()
            if any(token in rel_lower for token in tokens):
                score += 3
            for line_number, line in enumerate(lines, start=1):
                lowered = line.lower()
                hits = sum(1 for token in tokens if token in lowered)
                if hits <= 0:
                    continue
                score += 2 * hits
                snippet = re.sub(r"\s+", " ", line).strip()
                if snippet:
                    matches.append(f"{line_number}: {snippet}")
                if len(matches) >= 2:
                    break
            if score <= 0:
                continue
            scored.append((score, rel_path, matches))

        if not scored:
            available = [
                str(path.relative_to(self.working_dir)) for path in doc_paths[:max_results]
            ]
            response = "No local documentation matches were found.\nAvailable docs:\n" + "\n".join(
                f"- {path}" for path in available
            )
            self._cache_store(self._project_doc_cache, cache_key, response)
            return response

        scored.sort(key=lambda item: (-item[0], item[1]))
        lines = [f"Local documentation matches for `{query_text}`:"]
        for index, (_, rel_path, matches) in enumerate(scored[: max(1, max_results)], start=1):
            lines.append(f"{index}. {rel_path}")
            if matches:
                lines.extend(f"   - {snippet}" for snippet in matches)
        response = self._truncate_output("\n".join(lines))
        self._cache_store(self._project_doc_cache, cache_key, response)
        return response

    def _search_web_evidence(
        self,
        query: str,
        source: str = "all",
        max_results: int = 5,
    ) -> str:
        # The act of invoking this tool with a non-empty query IS positive
        # evidence of external-contract uncertainty (the rollout's agent
        # explicitly believes external information is needed). Pass it
        # through so ``agentic_search_internet_enabled`` gates by mode +
        # stage rather than by absence-of-evidence.
        query_text = str(query or "").strip()
        if not agentic_search_internet_enabled(
            self.agentic_search_config,
            query_text=query_text,
            external_contract_uncertainty=True,
        ):
            return "Error: search_web_evidence is unavailable in air-gapped mode."
        if not query_text:
            return "Error: search_web_evidence requires a non-empty query."
        configured_limit = getattr(
            self.agentic_search_config,
            "external_search_max_results",
            max_results,
        )
        try:
            max_results = max(1, min(int(max_results), int(configured_limit)))
        except (TypeError, ValueError):
            max_results = max(1, int(max_results or 1))

        source_value = str(source or "all").strip().lower() or "all"
        if source_value not in {"all", "github", "stackoverflow", "docs"}:
            return f"Error: Unsupported source '{source}'."
        cache_key = (query_text, source_value, max_results)
        cached = self._cache_lookup(self._external_search_cache, cache_key)
        if cached is not None:
            remaining = max(
                0,
                self._external_search_budget_total - self._external_search_budget_used,
            )
            return self._truncate_output(
                cached
                + "\n"
                + f"Interactive external search budget remaining: {remaining}/{self._external_search_budget_total} (cached result)"
            )

        if self._external_search_budget_total <= 0:
            return "Error: interactive external search budget is 0 for this run."
        if self._external_search_budget_used >= self._external_search_budget_total:
            return (
                "Error: interactive external search budget exhausted. "
                "Reuse existing evidence or continue from repository context."
            )

        search_query = self._build_external_search_query(query_text, source_value)
        try:
            html = self._fetch_url_text(
                "https://duckduckgo.com/html/?q=" + quote_plus(search_query),
                timeout_seconds=self._external_search_timeout_seconds(),
            )
        except Exception as exc:
            return f"External search failed: {exc}"

        parser = _DuckDuckGoHTMLParser()
        parser.feed(html)
        parser.close()
        results = parser.results[: max(1, max_results)]
        if not results:
            return f"No external results found for `{query_text}`."

        lines = [f"External evidence results for `{query_text}` ({source_value}):"]
        for index, item in enumerate(results, start=1):
            lines.append(f"{index}. {item['title']}")
            lines.append(f"   URL: {item['url']}")
            snippet = item.get("snippet", "").strip()
            if snippet:
                lines.append(f"   Snippet: {snippet}")
        self._external_search_budget_used += 1
        remaining = max(
            0,
            self._external_search_budget_total - self._external_search_budget_used,
        )
        body = "\n".join(lines)
        self._cache_store(self._external_search_cache, cache_key, body)
        return self._truncate_output(
            body
            + "\n"
            + f"Interactive external search budget remaining: {remaining}/{self._external_search_budget_total}"
        )

    def _build_external_search_query(self, query: str, source: str) -> str:
        base = str(query or "").strip()
        if source == "github":
            return f"{base} site:github.com"
        if source == "stackoverflow":
            return f"{base} site:stackoverflow.com"
        if source == "docs":
            return f"{base} official documentation"
        return base

    def _external_search_timeout_seconds(self) -> int:
        timeout = getattr(
            self.agentic_search_config,
            "external_search_timeout_seconds",
            12,
        )
        try:
            return max(3, int(timeout))
        except (TypeError, ValueError):
            return 12

    def _fetch_url_text(
        self,
        url: str,
        *,
        timeout_seconds: int,
    ) -> str:
        request = Request(
            url,
            headers={
                "User-Agent": "apex-agentic-search/1.0",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urlopen(request, timeout=timeout_seconds) as response:
            return response.read().decode("utf-8", errors="replace")

    def _find_symbols(
        self,
        name: str,
        kind: str = "any",
        path: Optional[str] = None,
    ) -> str:
        search_dir = self._resolve_path(path) if path else self._shell_cwd
        matches: list[str] = []
        for python_file in sorted(search_dir.rglob("*.py")):
            try:
                tree = ast.parse(python_file.read_text(errors="replace"))
            except SyntaxError:
                continue

            finder = _SymbolFinder()
            finder.visit(tree)
            rel_path = python_file.relative_to(self.working_dir)
            for symbol_name, symbol_kind, line_number, parent_class in finder.results:
                if kind not in ("any", symbol_kind):
                    continue
                if name.lower() not in symbol_name.lower():
                    continue
                prefix = f"{parent_class}." if parent_class else ""
                matches.append(
                    f"- {prefix}{symbol_name} ({symbol_kind}) @ {rel_path}:{line_number}"
                )

        if not matches:
            return f"No symbols matching '{name}' (kind={kind}) found."
        return f"Found {len(matches)} symbol(s):\n" + "\n".join(matches[:50])

    def _lookup_definition(self, symbol_name: str) -> str:
        if self.repo_context is None:
            return "Repository graph is unavailable."
        matches = self.repo_context.lookup_definition(symbol_name)
        if not matches:
            return f"No graph definitions found for '{symbol_name}'."
        blocks = []
        for node in matches[:8]:
            blocks.extend(
                [
                    f"- {node.name} ({node.node_type}) @ {node.file_path}:{node.start_line}-{node.end_line}",
                    node.code.strip() or "<empty>",
                    "",
                ]
            )
        return "\n".join(blocks).strip()

    def _trace_callers(self, symbol_name: str) -> str:
        if self.repo_context is None:
            return "Repository graph is unavailable."
        callers = self.repo_context.trace_callers(symbol_name)
        if not callers:
            return f"No callers found for '{symbol_name}'."
        return "Callers:\n" + "\n".join(
            f"- {node.name} @ {node.file_path}:{node.start_line}" for node in callers[:20]
        )

    def _trace_callees(self, symbol_name: str) -> str:
        if self.repo_context is None:
            return "Repository graph is unavailable."
        callees = self.repo_context.trace_callees(symbol_name)
        if not callees:
            return f"No callees found for '{symbol_name}'."
        return "Callees:\n" + "\n".join(
            f"- {node.name} @ {node.file_path}:{node.start_line}" for node in callees[:20]
        )

    def _get_entity_context(self, entity_name: str) -> str:
        if self.repo_context is None:
            return "Repository graph is unavailable."
        context = self.repo_context.get_entity_context(entity_name)
        if not context:
            return f"No context found for '{entity_name}'."

        lines = []
        entity = context["entity"]
        lines.append(
            f"Entity: {entity['name']} ({entity['node_type']}) @ {entity['file_path']}:{entity['start_line']}"
        )
        if context.get("container"):
            container = context["container"]
            lines.append(f"Container: {container['name']} ({container['node_type']})")
        if context.get("siblings"):
            lines.append("Siblings:")
            lines.extend(f"- {item['name']}" for item in context["siblings"][:10])
        if context.get("callers"):
            lines.append("Callers:")
            lines.extend(f"- {item['name']}" for item in context["callers"][:10])
        if context.get("callees"):
            lines.append("Callees:")
            lines.extend(f"- {item['name']}" for item in context["callees"][:10])
        return "\n".join(lines)

    def _edit_file(self, path: str, old_text: str, new_text: str) -> str:
        write_scope_error = self._write_scope_error(path)
        if write_scope_error:
            return write_scope_error
        file_path = self._resolve_path(path)

        if old_text == "":
            if file_path.exists():
                return (
                    "EDIT FAILED - old_text is empty but the file already exists. "
                    "Use an exact replacement for existing files."
                )
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if self.config.lint_on_edit and file_path.suffix == ".py":
                lint_error = self._lint_python(new_text)
                if lint_error:
                    return f"EDIT REJECTED - Syntax error in new file:\n{lint_error}"
            file_path.write_text(new_text)
            # Phase 5.5: invalidate doc cache on any successful edit.
            self._cache_invalidate_on_edit(path)
            line_count = max(len(new_text.splitlines()), 1)
            return self._render_edit_feedback(path, file_path, 1, line_count, created=True)

        if not file_path.exists():
            return f"Error: File '{path}' does not exist."

        content = file_path.read_text(errors="replace")
        occurrences = content.count(old_text)
        if occurrences == 0:
            return (
                f"EDIT FAILED - old_text was not found in {path}. "
                "Use view_file to grab an exact match."
            )
        if occurrences > 1:
            return (
                f"EDIT FAILED - old_text matched {occurrences} locations in {path}. "
                "Use a more specific block."
            )

        start_offset = content.index(old_text)
        start_line = content[:start_offset].count("\n") + 1
        updated = content.replace(old_text, new_text, 1)
        if self.config.lint_on_edit and file_path.suffix == ".py":
            lint_error = self._lint_python(updated)
            if lint_error:
                return f"EDIT REJECTED - The edit introduces a syntax error:\n{lint_error}"

        file_path.write_text(updated)
        # Phase 5.5: invalidate doc cache on any successful edit.
        self._cache_invalidate_on_edit(path)
        span_lines = max(len(new_text.splitlines()), len(old_text.splitlines()), 1)
        end_line = start_line + span_lines - 1
        return self._render_edit_feedback(path, file_path, start_line, end_line)

    def _lint_python(self, code: str) -> Optional[str]:
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return f"SyntaxError at line {exc.lineno}: {exc.msg}"
        return None

    def _bash(self, command: str, timeout: Optional[int] = None) -> str:
        timeout = timeout or self.config.bash_timeout
        workspace_escape_error = self._bash_workspace_escape_error(command)
        if workspace_escape_error:
            return workspace_escape_error
        cwd_sentinel = f"__APEX_ACI_CWD_{uuid.uuid4().hex}__"

        script = "\n".join(
            [
                f"cd {shlex.quote(self._target_shell_cwd())}",
                command,
                "status=$?",
                f"printf '\\n{cwd_sentinel}%s\\n' \"$PWD\"",
                "exit $status",
            ]
        )

        # Phase 5.6: default to ``bash -c``; ``bash -lc`` is opt-in via
        # ``ACIConfig.allow_login_shell``. See aci_security.resolve_bash_invocation.
        bash_argv = resolve_bash_invocation(
            allow_login_shell=bool(getattr(self.config, "allow_login_shell", False))
        )
        try:
            result = subprocess.run(
                bash_argv + [script],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.working_dir),
                env=self._bash_env,
            )
        except subprocess.TimeoutExpired:
            output = f"Command timed out after {timeout} seconds."
            violations, restore_errors = self._enforce_write_scope()
            if violations:
                violation_message = (
                    "WRITE SCOPE VIOLATION - command modified hard-forbidden or protected "
                    f"files: {', '.join(violations[:8])}. APEX restored those changes."
                )
                if restore_errors:
                    violation_message += " Restore errors: " + "; ".join(restore_errors[:3])
                output = f"{violation_message}\n{output}"
            return self._truncate_output(output)
        stdout_text = result.stdout or ""
        stdout_lines = stdout_text.splitlines()
        new_cwd = ""
        for index in range(len(stdout_lines) - 1, -1, -1):
            line = stdout_lines[index]
            if line.startswith(cwd_sentinel):
                new_cwd = line[len(cwd_sentinel) :].strip()
                del stdout_lines[index]
                break
        if new_cwd:
            resolved_cwd = self._host_shell_cwd_from_target(new_cwd)
            try:
                resolved_cwd.relative_to(self.working_dir)
                self._shell_cwd = resolved_cwd
            except ValueError:
                self._shell_cwd = self.working_dir
        stdout_text = "\n".join(stdout_lines)
        if stdout_text and stdout_text != (result.stdout or "").rstrip("\n"):
            stdout_text += "\n"

        output_parts = []
        if stdout_text:
            output_parts.append(stdout_text.rstrip())
        if result.stderr:
            output_parts.append(result.stderr.rstrip())
        output = "\n".join(part for part in output_parts if part).strip()

        if not output:
            if self.config.explicit_empty_output:
                output = "Your command ran successfully and did not produce any output."
            else:
                output = ""

        if result.returncode != 0:
            output = f"[exit_code={result.returncode}]\n{output}".strip()

        violations, restore_errors = self._enforce_write_scope()
        if violations:
            violation_message = (
                "WRITE SCOPE VIOLATION - command modified hard-forbidden or protected "
                f"files: {', '.join(violations[:8])}. APEX restored those changes."
            )
            if restore_errors:
                violation_message += " Restore errors: " + "; ".join(restore_errors[:3])
            output = (
                f"{violation_message}\nOriginal command output:\n{output}"
                if output
                else violation_message
            )

        return self._truncate_output(output)

    def _invoke_debugger(
        self,
        test_command: str,
        suspect_file: str,
        suspect_lines: Optional[list[int]] = None,
        hypothesis: str = "",
    ) -> str:
        from ..agents.debug_subagent import DebugSubagent

        subagent = DebugSubagent(workspace=str(self.working_dir))
        summary = subagent.run(
            test_command=test_command,
            suspect_file=suspect_file,
            suspect_lines=suspect_lines or [],
            hypothesis=hypothesis,
        )
        parent_llm = getattr(self._agent_runtime, "llm", None)
        if isinstance(parent_llm, LLMClient):
            parent_llm.trajectory.append(
                {
                    "timestamp": time.time(),
                    "debug_subtask": True,
                    "test_command": test_command,
                    "suspect_file": suspect_file,
                    "summary": summary.summary,
                    "usage": {"total_tokens": 0},
                }
            )
        return summary.to_concise_string()

    def _investigate(self, question: str, max_iterations: int = 5) -> str:
        if self._agent_runtime is None or not hasattr(self._agent_runtime, "fold_subtask"):
            return "Error: investigate tool requires an active agent loop reference."
        summary = self._agent_runtime.fold_subtask(
            question,
            max_iterations=max(1, min(int(max_iterations), 10)),
        )
        rendered = self._truncate_output(summary.strip() or "Investigation returned no summary.")
        return f"Investigation result:\n{rendered}"

    def _delegate_subtasks(
        self,
        tasks: list[dict[str, Any]],
        parallelism: int = 1,
    ) -> str:
        if not self.config.enable_agent_teams:
            return "Agent teams are disabled in this ACI configuration."
        if self._agent_runtime is None:
            return "Error: delegate_subtasks requires an active agent loop reference."
        if self.agent_depth >= self.config.max_agent_team_depth:
            return (
                "Error: delegate_subtasks exceeded the configured team depth limit "
                f"({self.config.max_agent_team_depth})."
            )

        try:
            normalized_tasks = self._resolve_planned_delegation_tasks(tasks)
            batches = self._build_delegation_batches(normalized_tasks)
        except ValueError as exc:
            return f"Error: {exc}"

        try:
            requested_parallelism = int(parallelism or 1)
        except (TypeError, ValueError):
            return "Error: delegate_subtasks `parallelism` must be an integer."
        requested_parallelism = max(
            1,
            min(
                requested_parallelism,
                self._delegation_plan_parallelism,
                self.config.max_agent_team_parallelism,
                len(normalized_tasks),
            ),
        )
        group_id = f"team-{uuid.uuid4().hex[:8]}"
        group_dir = self._team_workspace_root / group_id
        group_dir.mkdir(parents=True, exist_ok=True)

        order_map = {task.task_id: index for index, task in enumerate(normalized_tasks)}
        results_by_id: dict[str, _DelegatedSubtaskResult] = {}
        parent_llm = getattr(self._agent_runtime, "llm", None)
        allow_parallel = requested_parallelism > 1 and isinstance(parent_llm, LLMClient)

        for batch in batches:
            for start in range(0, len(batch), requested_parallelism):
                chunk = batch[start : start + requested_parallelism]
                if allow_parallel and len(chunk) > 1:
                    chunk_results: list[_DelegatedSubtaskResult] = []
                    with ThreadPoolExecutor(max_workers=len(chunk)) as executor:
                        futures = {}
                        for task in chunk:
                            child_context = contextvars.copy_context()
                            futures[
                                executor.submit(
                                    child_context.run,
                                    self._run_delegated_subtask,
                                    group_dir,
                                    task,
                                    {
                                        dependency_id: results_by_id[dependency_id]
                                        for dependency_id in task.depends_on
                                        if dependency_id in results_by_id
                                    },
                                )
                            ] = task.task_id
                        for future in as_completed(futures):
                            chunk_results.append(future.result())
                    chunk_results.sort(key=lambda item: order_map[item.task_id])
                else:
                    chunk_results = [
                        self._run_delegated_subtask(
                            group_dir,
                            task,
                            {
                                dependency_id: results_by_id[dependency_id]
                                for dependency_id in task.depends_on
                                if dependency_id in results_by_id
                            },
                        )
                        for task in chunk
                    ]

                for result in chunk_results:
                    results_by_id[result.task_id] = result
                    self._record_delegated_usage(group_id, result)

        return self._truncate_output(
            self._format_delegated_results(group_id, normalized_tasks, results_by_id)
        )

    def _canonicalize_planned_delegation_tasks(
        self,
        tasks: Any,
        *,
        default_max_iterations: Optional[int],
    ) -> tuple[list[_DelegatedSubtask], dict[str, Optional[str]]]:
        if not isinstance(tasks, list) or not tasks:
            return [], {}
        if len(tasks) > self.config.max_agent_team_size:
            tasks = list(tasks)[: self.config.max_agent_team_size]

        normalized_tasks: list[_DelegatedSubtask] = []
        pending_dependencies: dict[str, list[str]] = {}
        task_kinds: dict[str, str] = {}
        alias_map: dict[str, Optional[str]] = {}
        seen_ids: set[str] = set()

        def register_alias(alias: str, task_id: str) -> None:
            normalized_alias = str(alias or "").strip()
            if not normalized_alias:
                return
            existing = alias_map.get(normalized_alias)
            if existing is None and normalized_alias in alias_map:
                return
            if existing is not None and existing != task_id:
                alias_map[normalized_alias] = None
                return
            alias_map[normalized_alias] = task_id

        for index, payload in enumerate(tasks, start=1):
            if not isinstance(payload, dict):
                continue
            title = str(payload.get("title") or "").strip()
            if not title:
                continue
            raw_task_id = str(payload.get("task_id") or "").strip()
            base_task_id = raw_task_id or title or f"task_{index}"
            task_id = self._sanitize_task_id(base_task_id, default=f"task_{index}")
            while task_id in seen_ids:
                task_id = f"{task_id}_{index}"
            seen_ids.add(task_id)

            kind = str(payload.get("kind") or "implementation").strip().lower() or "implementation"
            focus_files = self._normalize_string_list(
                payload.get("owned_files") or payload.get("focus_files")
            )
            forbidden_files = self._normalize_string_list(payload.get("forbidden_files"))
            interface_symbols = self._normalize_string_list(payload.get("interface_symbols"))
            assumptions = self._normalize_string_list(payload.get("assumptions"))
            escalation_triggers = self._normalize_string_list(payload.get("escalation_triggers"))
            validation_targets = self._normalize_string_list(payload.get("validation_targets"))
            hypotheses = self._normalize_string_list(payload.get("hypotheses"))
            success_criteria = self._normalize_string_list(payload.get("success_criteria"))
            deliverable = str(payload.get("deliverable") or "").strip()
            if not success_criteria:
                if validation_targets:
                    success_criteria.append("Validate against: " + ", ".join(validation_targets))
                if deliverable:
                    success_criteria.append(deliverable)
            goal = (
                str(payload.get("goal") or "").strip()
                or str(payload.get("objective") or "").strip()
                or deliverable
                or title
            )

            max_iterations = payload.get("max_iterations")
            if max_iterations is None:
                max_iterations = default_max_iterations
            if max_iterations is not None:
                try:
                    max_iterations = int(max_iterations)
                except (TypeError, ValueError):
                    max_iterations = default_max_iterations
            if max_iterations is not None:
                max_iterations = max(
                    1,
                    min(max_iterations, self.config.max_agent_team_iterations),
                )

            normalized_tasks.append(
                _DelegatedSubtask(
                    task_id=task_id,
                    title=title,
                    goal=goal,
                    focus_files=focus_files,
                    forbidden_files=forbidden_files,
                    interface_symbols=interface_symbols,
                    assumptions=assumptions,
                    escalation_triggers=escalation_triggers,
                    success_criteria=success_criteria,
                    hypotheses=hypotheses,
                    max_iterations=max_iterations,
                )
            )
            pending_dependencies[task_id] = [
                str(item or "").strip()
                for item in list(payload.get("depends_on") or [])
                if str(item or "").strip()
            ]
            task_kinds[task_id] = kind
            for alias in {raw_task_id, base_task_id, task_id, title}:
                if alias:
                    register_alias(alias, task_id)
                    sanitized_alias = self._sanitize_task_id(alias, default="")
                    if sanitized_alias:
                        register_alias(sanitized_alias, task_id)

        by_id = {task.task_id: task for task in normalized_tasks}
        implementation_ids = [
            task.task_id
            for task in normalized_tasks
            if task_kinds.get(task.task_id) != "validation"
        ]
        for task in normalized_tasks:
            normalized_dependencies: list[str] = []
            raw_dependencies = pending_dependencies.get(task.task_id, [])
            if not raw_dependencies and task_kinds.get(task.task_id) == "validation":
                raw_dependencies = list(implementation_ids)
            for dependency in raw_dependencies:
                dependency_key = str(dependency or "").strip()
                sanitized_dependency = self._sanitize_task_id(dependency_key, default="")
                explicit_hit = alias_map[dependency_key] if dependency_key in alias_map else None
                sanitized_hit = (
                    alias_map[sanitized_dependency] if sanitized_dependency in alias_map else None
                )
                if (dependency_key in alias_map and explicit_hit is None) or (
                    sanitized_dependency in alias_map and sanitized_hit is None
                ):
                    continue
                resolved_dependency = explicit_hit or sanitized_hit
                if (
                    not resolved_dependency
                    or resolved_dependency not in by_id
                    or resolved_dependency == task.task_id
                ):
                    continue
                if resolved_dependency not in normalized_dependencies:
                    normalized_dependencies.append(resolved_dependency)
            task.depends_on = normalized_dependencies

        return normalized_tasks, alias_map

    def _resolve_planned_delegation_tasks(
        self,
        tasks: Any,
    ) -> list[_DelegatedSubtask]:
        if not self._delegation_plan_tasks:
            raise ValueError(
                "delegate_subtasks requires an orchestrator-authored delegation plan for this stage."
            )
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("delegate_subtasks requires a non-empty `tasks` array.")

        requested_ids: list[str] = []
        available = ", ".join(self._delegation_plan_order)
        for index, payload in enumerate(tasks, start=1):
            if not isinstance(payload, dict):
                raise ValueError(f"Subtask {index} must be an object.")
            candidates = [
                str(payload.get("task_id") or "").strip(),
                str(payload.get("title") or "").strip(),
            ]
            resolved_id: Optional[str] = None
            for candidate in candidates:
                if not candidate:
                    continue
                sanitized_candidate = self._sanitize_task_id(candidate, default="")
                explicit_hit = (
                    self._delegation_plan_aliases[candidate]
                    if candidate in self._delegation_plan_aliases
                    else None
                )
                sanitized_hit = (
                    self._delegation_plan_aliases[sanitized_candidate]
                    if sanitized_candidate in self._delegation_plan_aliases
                    else None
                )
                if (candidate in self._delegation_plan_aliases and explicit_hit is None) or (
                    sanitized_candidate in self._delegation_plan_aliases and sanitized_hit is None
                ):
                    raise ValueError(
                        f"Requested subtask `{candidate}` is ambiguous in the orchestrator plan. "
                        "Use an exact task_id."
                    )
                resolved_id = explicit_hit or sanitized_hit
                if resolved_id:
                    break
            if not resolved_id:
                identifier = candidates[0] or candidates[1] or f"task_{index}"
                raise ValueError(
                    f"Requested subtask `{identifier}` is not in the orchestrator-authored "
                    f"delegation plan. Available task ids: {available}."
                )
            if resolved_id not in requested_ids:
                requested_ids.append(resolved_id)

        selected_ids: set[str] = set()

        def include_with_dependencies(task_id: str) -> None:
            if task_id in selected_ids:
                return
            selected_ids.add(task_id)
            for dependency in self._delegation_plan_lookup[task_id].depends_on:
                include_with_dependencies(dependency)

        for task_id in requested_ids:
            include_with_dependencies(task_id)

        return [
            self._delegation_plan_lookup[task_id]
            for task_id in self._delegation_plan_order
            if task_id in selected_ids
        ]

    def _normalize_delegated_tasks(
        self,
        tasks: Any,
    ) -> list[_DelegatedSubtask]:
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("delegate_subtasks requires a non-empty `tasks` array.")
        if len(tasks) > self.config.max_agent_team_size:
            raise ValueError(
                "delegate_subtasks received too many subtasks "
                f"({len(tasks)} > {self.config.max_agent_team_size})."
            )

        normalized_tasks: list[_DelegatedSubtask] = []
        dependency_payloads: dict[str, list[str]] = {}
        alias_map: dict[str, Optional[str]] = {}
        seen_ids: set[str] = set()

        def register_alias(alias: str, task_id: str) -> None:
            normalized_alias = str(alias or "").strip()
            if not normalized_alias:
                return
            existing = alias_map.get(normalized_alias)
            if existing is None and normalized_alias in alias_map:
                return
            if existing is not None and existing != task_id:
                alias_map[normalized_alias] = None
                return
            alias_map[normalized_alias] = task_id

        for index, payload in enumerate(tasks, start=1):
            if not isinstance(payload, dict):
                raise ValueError(f"Subtask {index} must be an object.")

            title = str(payload.get("title") or "").strip()
            goal = str(payload.get("goal") or "").strip()
            if not title:
                raise ValueError(f"Subtask {index} is missing `title`.")
            if not goal:
                raise ValueError(f"Subtask {index} is missing `goal`.")

            raw_task_id = str(payload.get("task_id") or "").strip()
            base_task_id = raw_task_id or title or f"task_{index}"
            task_id = self._sanitize_task_id(base_task_id, default=f"task_{index}")
            while task_id in seen_ids:
                task_id = f"{task_id}_{index}"
            seen_ids.add(task_id)

            focus_files = self._normalize_string_list(
                payload.get("owned_files") or payload.get("focus_files")
            )
            forbidden_files = self._normalize_string_list(payload.get("forbidden_files"))
            interface_symbols = self._normalize_string_list(payload.get("interface_symbols"))
            assumptions = self._normalize_string_list(payload.get("assumptions"))
            escalation_triggers = self._normalize_string_list(payload.get("escalation_triggers"))
            success_criteria = self._normalize_string_list(payload.get("success_criteria"))
            hypotheses = self._normalize_string_list(payload.get("hypotheses"))
            max_iterations = payload.get("max_iterations")
            if max_iterations is not None:
                try:
                    max_iterations = int(max_iterations)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Subtask `{task_id}` has an invalid `max_iterations` value."
                    ) from exc
                max_iterations = max(
                    1,
                    min(max_iterations, self.config.max_agent_team_iterations),
                )

            normalized_tasks.append(
                _DelegatedSubtask(
                    task_id=task_id,
                    title=title,
                    goal=goal,
                    focus_files=focus_files,
                    forbidden_files=forbidden_files,
                    interface_symbols=interface_symbols,
                    assumptions=assumptions,
                    escalation_triggers=escalation_triggers,
                    success_criteria=success_criteria,
                    hypotheses=hypotheses,
                    max_iterations=max_iterations,
                )
            )
            dependency_payloads[task_id] = [
                str(item or "").strip()
                for item in list(payload.get("depends_on") or [])
                if str(item or "").strip()
            ]
            for alias in {raw_task_id, base_task_id, task_id}:
                if alias:
                    register_alias(alias, task_id)
                    sanitized_alias = self._sanitize_task_id(alias, default="")
                    if sanitized_alias:
                        register_alias(sanitized_alias, task_id)

        by_id = {task.task_id: task for task in normalized_tasks}
        for task in normalized_tasks:
            normalized_dependencies: list[str] = []
            for dependency in dependency_payloads.get(task.task_id, []):
                dependency_key = str(dependency or "").strip()
                sanitized_dependency = self._sanitize_task_id(dependency_key, default="")
                explicit_hit = alias_map[dependency_key] if dependency_key in alias_map else None
                sanitized_hit = (
                    alias_map[sanitized_dependency] if sanitized_dependency in alias_map else None
                )
                if (dependency_key in alias_map and explicit_hit is None) or (
                    sanitized_dependency in alias_map and sanitized_hit is None
                ):
                    raise ValueError(
                        f"Subtask `{task.task_id}` has an ambiguous dependency reference "
                        f"`{dependency}`. Use unique task_id values."
                    )
                resolved_dependency = explicit_hit or sanitized_hit
                if not resolved_dependency or resolved_dependency not in by_id:
                    raise ValueError(
                        f"Subtask `{task.task_id}` depends on unknown task `{dependency}`."
                    )
                if resolved_dependency == task.task_id:
                    raise ValueError(f"Subtask `{task.task_id}` cannot depend on itself.")
                if resolved_dependency not in normalized_dependencies:
                    normalized_dependencies.append(resolved_dependency)
            task.depends_on = normalized_dependencies

        return normalized_tasks

    def _build_delegation_batches(
        self,
        tasks: list[_DelegatedSubtask],
    ) -> list[list[_DelegatedSubtask]]:
        by_id = {task.task_id: task for task in tasks}
        indegree = {task.task_id: len(task.depends_on) for task in tasks}
        dependents: dict[str, list[str]] = {task.task_id: [] for task in tasks}
        ordered_ids = [task.task_id for task in tasks]
        for task in tasks:
            for dependency in task.depends_on:
                dependents.setdefault(dependency, []).append(task.task_id)

        ready = [task_id for task_id in ordered_ids if indegree[task_id] == 0]
        processed = 0
        batches: list[list[_DelegatedSubtask]] = []

        while ready:
            current_batch = [by_id[task_id] for task_id in ready]
            batches.append(current_batch)
            processed += len(current_batch)

            next_ready_set: set[str] = set()
            for task_id in ready:
                for dependent in dependents.get(task_id, []):
                    indegree[dependent] -= 1
                    if indegree[dependent] == 0:
                        next_ready_set.add(dependent)
            ready = [task_id for task_id in ordered_ids if task_id in next_ready_set]

        if processed != len(tasks):
            raise ValueError("delegate_subtasks detected a dependency cycle.")

        return batches

    def _run_delegated_subtask(
        self,
        group_dir: Path,
        task: _DelegatedSubtask,
        dependency_results: dict[str, _DelegatedSubtaskResult],
    ) -> _DelegatedSubtaskResult:
        child_dir = group_dir / task.task_id
        dependency_items = [
            dependency_results[dependency_id]
            for dependency_id in task.depends_on
            if dependency_id in dependency_results
        ]
        child_llm = _clone_child_llm(getattr(self._agent_runtime, "llm", None))

        try:
            baseline_commit, dependency_notes = self._create_child_workspace(
                child_dir,
                dependency_items,
            )
            child_executor = ACIToolExecutor(
                str(child_dir),
                self.config,
                agentic_search_config=self.agentic_search_config,
                repo_context=self.repo_context,
                memory_bus=self.memory_bus,
                rollout_id=self.rollout_id,
                execution_tree=None,
                baseline_commit=baseline_commit,
                test_command=self.test_command,
                test_timeout=self.test_timeout,
                agent_depth=self.agent_depth + 1,
                agent_lineage=self.agent_lineage + [task.task_id],
            )
            child_executor.set_discovery_scope(**self._discovery_scope)
            child_executor.set_write_scope(
                [],
                self._normalize_string_list(task.forbidden_files),
                enforce=False,
            )
            child_tools = self._build_delegated_tools()
            max_iterations = max(
                1,
                task.max_iterations
                if task.max_iterations is not None
                else self.config.max_agent_team_iterations,
            )
            child_loop = AgentLoop(
                llm=child_llm,
                system_prompt=DELEGATED_WORKER_SYSTEM_PROMPT,
                tools=child_tools,
                tool_executor=child_executor.execute,
                max_iterations=max_iterations,
                finish_tool_names={"submit_delegate_result"},
                dynamic_context_provider=child_executor.render_dynamic_context,
            )
            if hasattr(self._agent_runtime, "context_config"):
                child_loop.set_context_config(self._agent_runtime.context_config)

            submission = child_loop.run(
                self._build_delegated_prompt(
                    task,
                    dependency_items,
                    dependency_notes,
                )
            )

            submitted = dict(submission.arguments) if submission is not None else {}
            changed_files = list_git_changed_files(
                child_dir,
                baseline_ref=baseline_commit,
            )
            changed_files = list(
                dict.fromkeys(
                    changed_files + self._normalize_string_list(submitted.get("changed_files"))
                )
            )
            tests_run = self._normalize_string_list(submitted.get("tests_run"))
            followups = self._normalize_string_list(submitted.get("followups"))
            summary = str(submitted.get("summary") or "").strip()
            if not summary:
                summary = self._extract_last_assistant_message(child_loop)
            confidence = self._normalize_confidence(
                submitted.get("confidence"),
                default=0.65 if submission is not None else 0.0,
            )
            diff_text, patch_path = self._collect_child_patch(
                child_dir,
                baseline_commit,
                group_dir / f"{task.task_id}.patch",
            )

            return _DelegatedSubtaskResult(
                task_id=task.task_id,
                title=task.title,
                success=submission is not None,
                summary=summary or "Delegated child finished without a structured summary.",
                changed_files=changed_files,
                tests_run=tests_run,
                confidence=confidence,
                followups=followups,
                workspace_path=self._relative_to_workspace(child_dir),
                patch_path=patch_path,
                patch_preview=_truncate_chars(
                    diff_text,
                    self.config.agent_team_patch_preview_chars,
                ),
                lineage=child_executor.agent_lineage,
                dependency_notes=dependency_notes,
                tokens_used=int(getattr(child_llm, "total_tokens_used", 0) or 0),
            )
        except Exception as exc:
            return _DelegatedSubtaskResult(
                task_id=task.task_id,
                title=task.title,
                success=False,
                summary=f"Delegated child failed: {exc}",
                workspace_path=self._relative_to_workspace(child_dir),
                lineage=self.agent_lineage + [task.task_id],
                error=str(exc),
                tokens_used=int(getattr(child_llm, "total_tokens_used", 0) or 0),
            )
        finally:
            if not self.config.keep_agent_team_workspaces and child_dir.exists():
                shutil.rmtree(child_dir, ignore_errors=True)

    def _build_delegated_tools(self) -> list[ToolDefinition]:
        runtime_tools = list(getattr(self._agent_runtime, "tools", []) or [])
        finish_tool_names = set(getattr(self._agent_runtime, "finish_tool_names", set()) or set())
        filtered_tools: list[ToolDefinition] = []
        seen_names: set[str] = set()
        for tool in list(BASE_TOOL_DEFINITIONS) + runtime_tools:
            if tool.name in {"approve", "revise", "delegate_subtasks"}:
                continue
            if tool.name == "search_project_docs" and not bool(
                getattr(self.agentic_search_config, "enable_local_doc_guidance", False)
            ):
                continue
            if tool.name == "search_web_evidence" and not agentic_search_internet_enabled(
                self.agentic_search_config,
                # Tool registration must work even before any
                # rollout-side stall has occurred; the per-invocation
                # gate at ``_search_web_evidence`` enforces the
                # local-first policy at call time.
                external_contract_uncertainty=True,
            ):
                continue
            if tool.name in finish_tool_names or tool.name.startswith("submit_"):
                continue
            if tool.name in seen_names:
                continue
            filtered_tools.append(tool)
            seen_names.add(tool.name)
        filtered_tools.append(_SUBMIT_DELEGATED_RESULT_TOOL)
        return filtered_tools

    def _build_delegated_prompt(
        self,
        task: _DelegatedSubtask,
        dependency_items: list[_DelegatedSubtaskResult],
        dependency_notes: list[str],
    ) -> str:
        focus_files = self._normalize_string_list(task.focus_files)
        forbidden_files = self._normalize_string_list(task.forbidden_files)
        interface_symbols = self._normalize_string_list(task.interface_symbols)
        assumptions = self._normalize_string_list(task.assumptions)
        escalation_triggers = self._normalize_string_list(task.escalation_triggers)
        success_criteria = self._normalize_string_list(task.success_criteria)
        hypotheses = self._normalize_string_list(task.hypotheses)
        lines = [
            "# Parent Objective",
            _truncate_chars(
                self._current_runtime_task() or "No parent task description was captured.",
                1400,
            ),
            "",
            "# Delegated Subtask",
            f"Task ID: {task.task_id}",
            f"Title: {task.title}",
            f"Goal: {task.goal}",
            "",
            "# Focus Files",
            "\n".join(f"- {item}" for item in focus_files) or "- infer from the repository",
            "",
            "# Success Criteria",
            "\n".join(f"- {item}" for item in success_criteria)
            or "- produce a concise, evidence-backed result",
            "",
            "# Hypotheses",
            "\n".join(f"- {item}" for item in hypotheses)
            or "- infer the best current hypothesis from the repository",
        ]
        if forbidden_files:
            lines.extend(
                [
                    "",
                    "# Forbidden Files",
                    "\n".join(f"- {item}" for item in forbidden_files),
                ]
            )
        if interface_symbols:
            lines.extend(
                [
                    "",
                    "# Interface Symbols",
                    "\n".join(f"- {item}" for item in interface_symbols),
                ]
            )
        if assumptions:
            lines.extend(
                [
                    "",
                    "# Assumptions",
                    "\n".join(f"- {item}" for item in assumptions),
                ]
            )
        if escalation_triggers:
            lines.extend(
                [
                    "",
                    "# Escalate If",
                    "\n".join(f"- {item}" for item in escalation_triggers),
                ]
            )
        task_list = self.render_task_list()
        if task_list:
            lines.extend(["", task_list])
        if dependency_items:
            lines.extend(["", "# Dependency Handoffs"])
            for item in dependency_items:
                lines.append(f"- {item.task_id}: {item.summary}")
                if item.patch_path:
                    lines.append(f"  patch seed: {item.patch_path}")
        if dependency_notes:
            lines.extend(["", "# Dependency Patch Notes"])
            lines.extend(f"- {note}" for note in dependency_notes[:6])
        lines.extend(
            [
                "",
                "# Constraints",
                (
                    "You are working in an isolated child workspace. Changes here do not "
                    "modify the parent workspace directly; your patch will be saved separately."
                ),
                f"Current lineage: {' > '.join(self.agent_lineage + [task.task_id])}",
                "Do not call delegate_subtasks again; Apex controls team orchestration from the parent rollout.",
                (
                    "Run focused verification when useful, keep the scope narrow, and call "
                    "submit_delegate_result when the subtask is complete."
                ),
                (
                    "Focus files are starting points, not exclusive ownership boundaries. "
                    "If adjacent implementation files are necessary, edit them minimally "
                    "and explain the evidence in your summary or followups."
                ),
                (
                    "Forbidden files remain hard boundaries; protected tests and forbidden "
                    "paths must not be edited unless the subtask explicitly allows them."
                ),
            ]
        )
        return "\n".join(lines)

    def _current_runtime_task(self) -> str:
        runtime = self._agent_runtime
        if runtime is None:
            return ""
        for attribute in ("original_task_description", "initial_task"):
            value = getattr(runtime, attribute, "")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _create_child_workspace(
        self,
        child_dir: Path,
        dependency_items: list[_DelegatedSubtaskResult],
    ) -> tuple[str, list[str]]:
        if child_dir.exists():
            shutil.rmtree(child_dir, ignore_errors=True)
        self._team_workspace_root.mkdir(parents=True, exist_ok=True)
        ignore = shutil.ignore_patterns(
            ".git",
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            ".mypy_cache",
            self._team_workspace_root.name,
        )
        copy_tree(self.working_dir, child_dir, ignore=ignore)
        baseline_commit = self._bootstrap_child_repo(child_dir)

        dependency_notes: list[str] = []
        applied_dependency_patch = False
        for item in dependency_items:
            if not item.patch_path:
                dependency_notes.append(
                    f"{item.task_id}: no patch artifact was available; summary only."
                )
                continue
            patch_source = Path(item.patch_path)
            if not patch_source.is_absolute():
                patch_source = (self.working_dir / patch_source).resolve()
            apply_result = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", str(patch_source)],
                capture_output=True,
                text=True,
                cwd=str(child_dir),
                check=False,
            )
            if apply_result.returncode != 0:
                dependency_notes.append(
                    f"{item.task_id}: "
                    f"{apply_result.stderr.strip() or apply_result.stdout.strip() or 'patch apply failed'}"
                )
                continue
            applied_dependency_patch = True

        if applied_dependency_patch:
            baseline_commit = self._commit_child_repo_state(
                child_dir,
                "APEX child dependency seed",
            )

        return baseline_commit, dependency_notes

    def _bootstrap_child_repo(self, child_dir: Path) -> str:
        init_result = subprocess.run(
            ["git", "init"],
            capture_output=True,
            text=True,
            cwd=str(child_dir),
            check=False,
        )
        if init_result.returncode != 0:
            raise RuntimeError(
                init_result.stderr.strip()
                or init_result.stdout.strip()
                or "Failed to initialize child workspace git repository."
            )
        self._ensure_git_identity(child_dir)
        info_exclude = child_dir / ".git" / "info" / "exclude"
        info_exclude.parent.mkdir(parents=True, exist_ok=True)
        with info_exclude.open("a") as handle:
            handle.write("\n")
            handle.write(self._team_workspace_root.name + "/\n")
            handle.write("__pycache__/\n")
            handle.write(".pytest_cache/\n")
            handle.write(".mypy_cache/\n")
            handle.write("*.pyc\n")
        return self._commit_child_repo_state(child_dir, "APEX child baseline")

    def _commit_child_repo_state(self, repo_dir: Path, message: str) -> str:
        self._ensure_git_identity(repo_dir)
        subprocess.run(
            ["git", "add", "-A"],
            capture_output=True,
            cwd=str(repo_dir),
            check=False,
        )
        commit_result = subprocess.run(
            ["git", "commit", "--allow-empty", "-m", message],
            capture_output=True,
            text=True,
            cwd=str(repo_dir),
            check=False,
        )
        if commit_result.returncode != 0:
            raise RuntimeError(
                commit_result.stderr.strip()
                or commit_result.stdout.strip()
                or "git commit failed in delegated child workspace."
            )
        head_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(repo_dir),
            check=False,
        )
        return head_result.stdout.strip()

    def _ensure_git_identity(self, repo_dir: Path) -> None:
        for key, value in {
            "user.email": "apex@example.com",
            "user.name": "APEX",
        }.items():
            existing = subprocess.run(
                ["git", "config", key],
                capture_output=True,
                text=True,
                cwd=str(repo_dir),
                check=False,
            )
            if existing.returncode == 0 and existing.stdout.strip():
                continue
            subprocess.run(
                ["git", "config", key, value],
                capture_output=True,
                cwd=str(repo_dir),
                check=False,
            )

    def _collect_child_patch(
        self,
        child_dir: Path,
        baseline_commit: str,
        patch_file: Path,
    ) -> tuple[str, Optional[str]]:
        subprocess.run(
            ["git", "add", "-N", "."],
            capture_output=True,
            cwd=str(child_dir),
            check=False,
        )
        diff_result = subprocess.run(
            [
                "git",
                "diff",
                "--binary",
                "--relative",
                baseline_commit,
                "--",
                ".",
                *ignored_change_pathspecs(),
            ],
            capture_output=True,
            text=True,
            cwd=str(child_dir),
            check=False,
        )
        diff_text = diff_result.stdout
        if not diff_text.strip():
            return "", None
        patch_file.parent.mkdir(parents=True, exist_ok=True)
        patch_file.write_text(diff_text)
        return diff_text, self._relative_to_workspace(patch_file)

    def _record_delegated_usage(
        self,
        group_id: str,
        result: _DelegatedSubtaskResult,
    ) -> None:
        parent_llm = getattr(self._agent_runtime, "llm", None)
        if not isinstance(parent_llm, LLMClient):
            return
        parent_llm.total_tokens_used += max(result.tokens_used, 0)
        parent_llm.trajectory.append(
            {
                "timestamp": time.time(),
                "delegated_subtask": True,
                "group_id": group_id,
                "task_id": result.task_id,
                "title": result.title,
                "success": result.success,
                "summary": result.summary,
                "usage": {"total_tokens": result.tokens_used},
                "changed_files": list(result.changed_files),
                "patch_path": result.patch_path,
            }
        )

    def _format_delegated_results(
        self,
        group_id: str,
        tasks: list[_DelegatedSubtask],
        results_by_id: dict[str, _DelegatedSubtaskResult],
    ) -> str:
        ordered_results = [
            results_by_id[task.task_id] for task in tasks if task.task_id in results_by_id
        ]
        succeeded = sum(1 for item in ordered_results if item.success)
        lines = [
            (
                f"Delegation group {group_id} completed with "
                f"{succeeded}/{len(ordered_results)} successful child agents."
            )
        ]
        for result in ordered_results:
            status = "ok" if result.success else "failed"
            lines.extend(
                [
                    "",
                    f"[{status}] {result.task_id}: {result.title}",
                    f"summary: {result.summary}",
                    f"lineage: {' > '.join(result.lineage)}",
                ]
            )
            if result.changed_files:
                lines.append("changed_files: " + ", ".join(result.changed_files[:8]))
            if result.tests_run:
                lines.append("tests_run: " + ", ".join(result.tests_run[:6]))
            if result.patch_path:
                lines.append(f"patch_path: {result.patch_path}")
            if result.workspace_path:
                lines.append(f"workspace: {result.workspace_path}")
            if result.followups:
                lines.append("followups: " + "; ".join(result.followups[:4]))
            if result.dependency_notes:
                lines.append("dependency_notes: " + " | ".join(result.dependency_notes[:3]))
            if result.error and not result.success:
                lines.append(f"error: {result.error}")
            if result.patch_preview:
                lines.append("patch_preview:")
                lines.append(result.patch_preview)
        return "\n".join(lines)

    def _extract_last_assistant_message(self, loop: AgentLoop) -> str:
        for message in reversed(loop.messages):
            if message.role == "assistant" and (message.content or "").strip():
                return message.content.strip()
        return ""

    def _relative_to_workspace(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.working_dir))
        except ValueError:
            return str(path.resolve())

    def _normalize_string_list(self, values: Any) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            return []
        normalized = []
        for value in values:
            text = str(value or "").strip()
            if text:
                normalized.append(text)
        return list(dict.fromkeys(normalized))

    def _normalize_confidence(self, value: Any, *, default: float) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = default
        return max(0.0, min(confidence, 1.0))

    def _sanitize_task_id(self, raw: str, *, default: str) -> str:
        text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(raw or "").strip()).strip("._-")
        return text or default

    def _run_test_on_patch(self, test_code: str) -> str:
        test_path = self.working_dir / "_apex_patch_test.py"
        test_path.write_text(test_code)
        try:
            is_pytest_style = "def test_" in test_code or "import pytest" in test_code
            try:
                # Repo/issue-supplied test commands may legitimately chain
                # `export ... && pytest`, so allow basic chaining. We still
                # always reject command substitution and unbounded redirects.
                validate_test_command(
                    self.test_command,
                    allow_shell_chaining=True,
                    source="run_test_on_patch.test_command",
                )
            except TestCommandRejectedError as exc:
                return (
                    "[exit_code=2]\n"
                    f"Refusing to run test_command containing shell injection: {exc.reason}"
                )
            disable_plugin_autoload = should_disable_pytest_plugin_autoload(
                self.test_command or "python3 -m pytest -q",
                repo_root=self.working_dir,
            )
            if is_pytest_style:
                command = build_ephemeral_pytest_command(
                    self.test_command,
                    str(test_path.name),
                    disable_plugin_autoload=disable_plugin_autoload,
                ) or (
                    f"{'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ' if disable_plugin_autoload else ''}"
                    f"python3 -m pytest {test_path.name} -q --tb=no"
                )
            else:
                command = (
                    build_runtime_python_command(self.test_command, str(test_path.name))
                    or f"python3 {shlex.quote(str(test_path.name))}"
                )
            base_env = {
                **self._bash_env,
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            # Strip host secrets before any test command (LLM-supplied or repo
            # supplied) executes via bash. ApexRunner-style flows that
            # genuinely need a token must add it via cli_env_overrides.
            env, _ = redact_host_secrets(base_env)
            if disable_plugin_autoload:
                env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            else:
                env.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
            # Phase 5.6: default to ``bash -c``; allow opt-in login shell
            # via ``ACIConfig.allow_login_shell``.
            test_bash_argv = resolve_bash_invocation(
                allow_login_shell=bool(getattr(self.config, "allow_login_shell", False))
            )
            result = subprocess.run(
                test_bash_argv + [command],
                cwd=str(self.working_dir),
                capture_output=True,
                text=True,
                timeout=self.test_timeout,
                env=env,
            )
            if (
                result.returncode != 0
                and is_pytest_style
                and output_indicates_missing_pytest(result.stdout + result.stderr)
            ):
                for recovery_command in build_pytest_recovery_commands(
                    command,
                    repo_root=self.working_dir,
                ):
                    if recovery_command.strip() == command.strip():
                        continue
                    result = subprocess.run(
                        test_bash_argv + [recovery_command],
                        cwd=str(self.working_dir),
                        capture_output=True,
                        text=True,
                        timeout=self.test_timeout,
                        env=env,
                    )
                    if result.returncode == 0 or not output_indicates_missing_pytest(
                        result.stdout + result.stderr
                    ):
                        break
            output = "\n".join(
                part for part in [result.stdout.strip(), result.stderr.strip()] if part
            ).strip()
            if result.returncode != 0:
                output = f"[exit_code={result.returncode}]\n{output}".strip()
            return self._truncate_output(output or "Test finished without output.")
        except subprocess.TimeoutExpired:
            return f"[exit_code=124]\nCommand timed out after {self.test_timeout} seconds."
        finally:
            try:
                test_path.unlink()
            except FileNotFoundError:
                pass

    def _broadcast_discovery(
        self,
        insight_type: str,
        description: str,
        confidence: float = 0.7,
        file_paths: Optional[list[str]] = None,
        symbols: Optional[list[str]] = None,
        test_ids: Optional[list[str]] = None,
        stage_name: str = "",
        negative: bool = False,
        cross_run: bool = False,
    ) -> str:
        if self.memory_bus is None or self.rollout_id is None:
            return "Discovery bus is unavailable in this execution."
        self.memory_bus.broadcast(
            rollout_id=self.rollout_id,
            insight_type=insight_type,
            description=description,
            confidence=confidence,
            file_paths=file_paths,
            symbols=symbols,
            test_ids=test_ids,
            stage_name=stage_name,
            negative=negative,
        )
        # Phase 6.2: optionally also persist cross-run. Skips silently
        # (with debug log) when the executor wasn't given a store/sig
        # so callers can pass cross_run=True unconditionally without
        # blowing up older deployments.
        cross_run_status = ""
        if cross_run:
            if self.episodic_store is None or not self.task_signature:
                cross_run_status = " (cross_run skipped: no store attached)"
            else:
                try:
                    payload = {
                        "insight_type": str(insight_type or ""),
                        "description": str(description or ""),
                        "confidence": float(confidence),
                        "file_paths": list(file_paths or []),
                        "symbols": list(symbols or []),
                        "test_ids": list(test_ids or []),
                        "stage_name": str(stage_name or ""),
                        "negative": bool(negative),
                    }
                    self.episodic_store.broadcast(
                        task_signature=self.task_signature,
                        rollout_id=str(self.rollout_id),
                        episode_type=str(insight_type or "DISCOVERY").upper(),
                        payload=payload,
                    )
                    cross_run_status = " (also persisted cross-run)"
                except Exception as exc:  # pragma: no cover - defensive
                    cross_run_status = f" (cross_run persist failed: {exc})"
        return f"Discovery broadcast recorded.{cross_run_status}"

    def _query_discoveries(
        self,
        insight_types: Optional[list[str]] = None,
        file_paths: Optional[list[str]] = None,
        symbols: Optional[list[str]] = None,
        test_ids: Optional[list[str]] = None,
        stage_names: Optional[list[str]] = None,
        negative_only: bool = False,
        positive_only: bool = False,
        max_items: int = 12,
        cross_run: bool = False,
    ) -> str:
        if self.memory_bus is None or self.rollout_id is None:
            return "Discovery bus is unavailable in this execution."
        discoveries = self.memory_bus.query(
            exclude_rollout_id=self.rollout_id,
            insight_types=insight_types,
            file_paths=file_paths or self._discovery_scope["file_paths"],
            symbols=symbols or self._discovery_scope["symbols"],
            test_ids=test_ids or self._discovery_scope["test_ids"],
            stage_names=stage_names,
            negative_only=negative_only,
            positive_only=positive_only,
            max_items=max(1, min(int(max_items), 20)),
        )
        # Phase 6.2: optionally augment with cross-run priors from the
        # persistent episodic store. We render them as a separate block
        # so the agent can tell which beliefs came from THIS solve's
        # parallel rollouts vs. earlier solves on the same task.
        cross_run_lines: list[str] = []
        if cross_run and self.episodic_store is not None and self.task_signature:
            try:
                normalized_types = [str(t).upper() for t in (insight_types or []) if t] or None
                if normalized_types:
                    episodes: list[Any] = []
                    for episode_type in normalized_types:
                        episodes.extend(
                            self.episodic_store.query(
                                task_signature=self.task_signature,
                                episode_type=episode_type,
                                limit=max(1, min(int(max_items), 20)),
                            )
                        )
                else:
                    episodes = self.episodic_store.query(
                        task_signature=self.task_signature,
                        limit=max(1, min(int(max_items), 20)),
                    )
                for ep in episodes[: max(1, min(int(max_items), 20))]:
                    payload = ep.payload or {}
                    description = str(payload.get("description") or "").strip()
                    if not description:
                        continue
                    if positive_only and bool(payload.get("negative")):
                        continue
                    if negative_only and not bool(payload.get("negative")):
                        continue
                    prefix = "RULED_OUT" if payload.get("negative") else ep.episode_type
                    cross_run_lines.append(
                        f"- [PRIOR/{prefix}] rollout {ep.rollout_id}: {description}"
                    )
            except Exception as exc:  # pragma: no cover - defensive
                cross_run_lines.append(f"(cross_run query failed: {exc})")
        if not discoveries and not cross_run_lines:
            return "No discoveries from parallel agents yet."
        lines = ["Discoveries:"]
        for item in discoveries:
            metadata = []
            if item.stage_name:
                metadata.append(f"stage={item.stage_name}")
            if item.file_paths:
                metadata.append("files=" + ", ".join(item.file_paths[:3]))
            if item.symbols:
                metadata.append("symbols=" + ", ".join(item.symbols[:3]))
            if item.test_ids:
                metadata.append("tests=" + ", ".join(item.test_ids[:3]))
            suffix = f" ({'; '.join(metadata)})" if metadata else ""
            prefix = "RULED_OUT" if item.negative else item.insight_type
            lines.append(f"- [{prefix}] rollout {item.rollout_id}: {item.description}{suffix}")
        if cross_run_lines:
            lines.append("Cross-run priors (from earlier solves on this task):")
            lines.extend(cross_run_lines)
        return "\n".join(lines)

    def _update_task_list(self, tasks: list[dict[str, str]]) -> str:
        normalized = []
        for task in tasks:
            description = (task.get("description") or "").strip()
            status = (task.get("status") or "TODO").strip().upper()
            if not description:
                continue
            if status not in {"TODO", "IN_PROGRESS", "DONE", "BLOCKED"}:
                status = "TODO"
            normalized.append({"description": description, "status": status})
        self._task_list = normalized
        self._task_list_version += 1
        return self.render_task_list() or "Task list cleared."

    def _checkpoint_state(
        self,
        summary: str,
        action_taken: str,
        confidence: float = 0.5,
        test_pass_count: int = 0,
        test_fail_count: int = 0,
    ) -> str:
        if self.execution_tree is None:
            return "Execution tree is unavailable in this execution."

        checkpoint_id, created = self.execution_tree.checkpoint(
            context_summary=summary,
            action_taken=action_taken,
            value_score=max(0.0, min(float(confidence), 1.0)),
            test_pass=test_pass_count,
            test_fail=test_fail_count,
        )
        if not created:
            return (
                f"Checkpoint not created; still at {checkpoint_id}. "
                "Depth or branch limits for the current search path were reached."
            )
        return (
            f"Checkpoint saved: {checkpoint_id}\n"
            f"Known checkpoints: {', '.join(self.execution_tree.list_checkpoint_ids())}"
        )

    def _backtrack_to(self, checkpoint_id: str, reason: str = "") -> str:
        if self.execution_tree is None:
            return "Execution tree is unavailable in this execution."
        if not self.execution_tree.backtrack(checkpoint_id):
            return f"Backtrack failed: unknown or unrestorable checkpoint '{checkpoint_id}'."
        self._shell_cwd = self.working_dir
        reason_text = reason.strip() or "no reason provided"
        return f"Backtracked to checkpoint {checkpoint_id}. Reason: {reason_text}."

    def _planning(self, thought: str) -> str:
        self._planning_log.append(thought)
        return "Planning note recorded."

    def _submit(self, tool_name: str, payload: dict[str, Any]) -> str:
        # Phase 2C 2.7: ``submit_patch`` runs the localizer constraint
        # check inline so that:
        #  * In hard_constraint mode the agent sees a high-severity
        #    diagnostic listing the files without dropping useful source
        #    progress solely because it moved beyond a localization prior.
        #  * In warning mode the agent sees a clear warning so it
        #    knows its diff strayed off-target (still recorded).
        #  * In advisory mode the legacy "tool recorded" message is
        #    preserved.
        if tool_name == "submit_patch":
            payload_changed_files = (
                list(payload.get("changed_files") or []) if isinstance(payload, dict) else []
            )
            payload_changed_files = [
                str(item) for item in payload_changed_files if str(item).strip()
            ]
            try:
                validation = self.validate_patch_submission(
                    changed_files=payload_changed_files or None,
                )
            except Exception as exc:  # noqa: BLE001 - never fail submit
                _security_logger.warning(
                    "submit_patch: validate_patch_submission raised %s: %s; recording anyway",
                    type(exc).__name__,
                    exc,
                )
                validation = {"violation": False, "rejected": False}
            if validation.get("violation"):
                files = ", ".join(validation.get("off_target_files") or []) or "<unknown>"
                keys = ", ".join(sorted(payload)) if payload else "no fields"
                severity = str(validation.get("severity") or "unknown")
                return (
                    f"{tool_name} recorded ({keys}). "
                    f"WARNING (localizer enforcement={validation.get('enforcement')} "
                    f"severity={severity}): diff touched "
                    f"out-of-scope files [{files}]; off_target_patches counter "
                    f"incremented in diagnostics for planning and verification."
                )
        keys = ", ".join(sorted(payload)) if payload else "no fields"
        return f"{tool_name} recorded ({keys})."

    def _truncate_output(self, output: str) -> str:
        lines = output.splitlines()
        if len(lines) <= self.config.max_output_lines:
            return output
        head_count = self.config.max_output_lines // 2
        tail_count = self.config.max_output_lines - head_count
        truncated = lines[:head_count]
        truncated.append(f"... ({len(lines) - self.config.max_output_lines} lines omitted) ...")
        truncated.extend(lines[-tail_count:])
        return "\n".join(truncated)

    def _render_edit_feedback(
        self,
        path: str,
        file_path: Path,
        start_line: int,
        end_line: int,
        *,
        created: bool = False,
    ) -> str:
        lines = file_path.read_text(errors="replace").splitlines()
        total_lines = len(lines)
        context = self.config.edit_feedback_context_lines
        window_start = max(start_line - context, 1)
        window_end = min(end_line + context, total_lines)
        rendered = []
        for index, line in enumerate(lines[window_start - 1 : window_end], start=window_start):
            rendered.append(f"{index:>6} | {line}")
        action = "created" if created else "edited"
        return (
            f"Successfully {action} {path}.\n"
            f"Lines {window_start}-{window_end} after edit:\n" + "\n".join(rendered)
        )
