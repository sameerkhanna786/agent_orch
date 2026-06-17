"""
Issue planning and rollout briefing.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ..acceptance import (
    quick_verification_signal_score,
    rollout_has_authoritative_acceptance,
)
from ..controller_models import evaluate_policy_model
from ..controller_policy import (
    EVIDENCE_MODE_EVAL_ONLY_SUITE,
    EVIDENCE_MODE_GOLD_SUITE_VISIBLE,
    EVIDENCE_MODE_NO_SUITE_VISIBLE,
    EVIDENCE_MODE_PARTIAL_SUITE_VISIBLE,
    EvaluationConstraints,
    ShadowPolicyOption,
    TaskRegimePolicy,
    TaskRegimeProfile,
    TestInventory,
    build_shadow_policy_trace,
    canonical_test_inventory,
    default_test_inventory_language,
    derive_test_collection_command,
    infer_evidence_policy,
    infer_test_inventory_framework,
)
from ..controller_schema import (
    ControllerAction,
    coerce_controller_action,
    sync_controller_action_payload,
)
from ..controller_trace import append_controller_decision
from ..core.cli_backend import (
    CLIModelClient,
    extract_total_tokens,
)
from ..core.config import (
    ROLLOUT_PROFILE_STAGE_ORDER,
    AgentMode,
    ApexConfig,
    LLMBackend,
    LLMConfig,
    SearchMode,
)
from ..core.anti_repetition_memory import summarize_failed_rollouts
from ..core.component_ablation import (
    component_ablation_assignment_for_task,
    component_disabled,
)
from ..core.llm import (
    LLMClient,
    Message,
    ToolDefinition,
    _verify_value_against_tool_schema,
)
from ..core.llm_routing import (
    classify_llm_call_failover_failure,
    llm_backend_fingerprint,
    llm_backend_is_available,
    llm_backend_unavailable_reason,
    record_llm_backend_failure,
    resolve_available_llm_config,
)
from ..core.terminal_output import normalize_terminal_output
from ..core.stub_scanner import scan_repo_for_stub_surface
from ..preprocessing.repo_analyzer import RepoContext
from ..rollout.blackboard import blackboard_context_from_issue_plan
from ..rollout.localizer_scope import infer_scope_class, is_apex_harness_path, is_test_path
from ..test_portfolio import (
    extract_issue_contract_targets,
    infer_required_contract_axes_from_texts,
    normalize_test_generation_design_payload,
)

logger = logging.getLogger("apex.planning")


def _strip_residual_followup_text(text: str) -> str:
    marker = "Residual follow-up objective:"
    value = str(text or "")
    index = value.find(marker)
    if index < 0:
        return value.strip()
    return value[:index].strip()


def _issue_plan_expected_test_count(issue_plan: Optional["IssuePlan"]) -> int:
    """Best-effort expected test count for an issue plan (0 when unknown).

    Mirrors ``apex.rollout.engine._issue_plan_expected_test_count`` but lives
    here to avoid a circular import (engine imports this module). Fully
    fail-open: any error returns 0, which makes the size factor collapse to 1
    (today's flat behavior).
    """

    if issue_plan is None:
        return 0
    try:
        inventory = canonical_test_inventory(issue_plan)
    except Exception:  # noqa: BLE001 - size signal is best-effort, never fatal
        return 0
    try:
        count = int(inventory.expected_test_count or 0)
    except (TypeError, ValueError):
        count = 0
    if count <= 0:
        try:
            count = len(list(inventory.expected_test_ids or []))
        except Exception:  # noqa: BLE001 - never fatal
            count = 0
    return max(0, count)


def _rollout_budget_size_factor(
    expected_test_count: int,
    *,
    tests_per_unit: int,
    max_size_factor: int,
) -> int:
    """size_factor = clamp(1, ceil(count / tests_per_unit), max_size_factor).

    Local copy of ``apex.rollout.engine._rollout_budget_size_factor`` (engine
    imports this module, so we cannot import it back). Returns 1 (today's exact
    behavior) whenever the suite is small or inputs are degenerate, so the
    width scaling is provably a no-op off the giant suites. Monotone
    non-decreasing in ``expected_test_count``.
    """

    try:
        count = max(0, int(expected_test_count))
    except (TypeError, ValueError):
        return 1
    unit = max(1, int(tests_per_unit or 0))
    cap = max(1, int(max_size_factor or 1))
    if count <= 0:
        return 1
    factor = -(-count // unit)  # ceil division
    return max(1, min(factor, cap))


@dataclass
class ModuleGroup:
    """One disjoint module group for decomposition-scale repos (T2.2).

    Each group is owned by exactly one rollout. ``owned_files`` are the
    repo-relative source files the rollout may write; ``bridge_files`` are
    shared/interface files it may read but should coordinate around;
    ``interface_symbols`` are the cross-group contract symbols; and
    ``expected_test_ids_subset`` is the per-group slice of the full expected
    test set (populated by the Layer-B adapter, T2.4).
    """

    group_id: int
    owned_files: list[str] = field(default_factory=list)
    bridge_files: list[str] = field(default_factory=list)
    interface_symbols: list[str] = field(default_factory=list)
    expected_test_ids_subset: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "owned_files": list(self.owned_files),
            "bridge_files": list(self.bridge_files),
            "interface_symbols": list(self.interface_symbols),
            "expected_test_ids_subset": list(self.expected_test_ids_subset),
        }


def repo_is_decomposition_scale(
    issue_plan: "IssuePlan",
    repo_context: RepoContext,
    *,
    config: Optional[Any] = None,
    stub_file_count: Optional[int] = None,
) -> bool:
    """Return True iff the repo is large/structured enough to decompose (T2.1).

    SIZE/STRUCTURE triggered ONLY (Layer A, general): a repo qualifies iff

      * ``expected_test_count >= decomposition_min_expected_tests`` (default
        4000), OR
      * repo-wide stub-file count ``>= decomposition_min_stub_files`` (default
        120), OR
      * the inferred scope class is ``library_reconstruction``.

    No repo/language conditional — every threshold is a general measured-size
    knob, so small repos (pytest/jinja-scale) never trip the predicate.
    """
    rollout_cfg = getattr(config, "rollout", None) if config is not None else None
    if rollout_cfg is not None and not bool(
        getattr(rollout_cfg, "enable_decomposition_scale_partitioning", True)
    ):
        return False
    min_expected = int(
        getattr(rollout_cfg, "decomposition_min_expected_tests", 4000)
        if rollout_cfg is not None
        else 4000
    )
    min_stub_files = int(
        getattr(rollout_cfg, "decomposition_min_stub_files", 120)
        if rollout_cfg is not None
        else 120
    )

    test_context = getattr(issue_plan, "test_context", None)
    expected_test_count = int(getattr(test_context, "expected_test_count", 0) or 0)
    if expected_test_count >= min_expected:
        return True

    if stub_file_count is not None and int(stub_file_count) >= min_stub_files:
        return True

    # Structural signal: a library_reconstruction scope class (many changed /
    # focus files) is the qualitative twin of the size thresholds.
    relevant_files = list(getattr(issue_plan, "relevant_files", []) or [])
    source_focus_files = list(
        getattr(test_context, "incomplete_source_files", []) or []
    ) + list(getattr(test_context, "source_focus_files", []) or [])
    scope_class = infer_scope_class(
        editable_focus_files=relevant_files,
        solution_changed_files=relevant_files + source_focus_files,
    )
    if scope_class == "library_reconstruction":
        return True
    return False


def _normalize_verifier_diagnostic_path(path: Any) -> str:
    rel_path = str(path or "").strip().strip("[]").replace("\\", "/")
    if rel_path.startswith("./"):
        rel_path = rel_path[2:]
    # Residual summaries can include paths from copied candidate artifacts
    # (`workspaces/<id>/workspace/<repo path>`); follow-up plans need repo paths.
    rel_path = re.sub(r"^(?:.*?/)?workspaces/_pool/[^/]+/workspace/", "", rel_path)
    rel_path = re.sub(r"^(?:.*?/)?workspaces/[^/]+/workspace/", "", rel_path)
    rel_path = re.sub(r"^(?:.*?/)?workspaces/_pool/[^/]+/", "", rel_path)
    rel_path = re.sub(r"^(?:.*?/)?workspaces/[^/]+/", "", rel_path)
    rel_path = re.sub(r"^workspace/workspaces/_pool/[^/]+/", "", rel_path)
    rel_path = re.sub(r"^workspace/workspaces/[^/]+/", "", rel_path)
    if rel_path.startswith("./"):
        rel_path = rel_path[2:]
    if (
        not rel_path
        or rel_path.startswith(("http://", "https://", "/", "~"))
        or "/.venv/" in rel_path
        or "/site-packages/" in rel_path
        or "/dist-packages/" in rel_path
    ):
        return ""
    parts = Path(rel_path).parts
    if any(part == ".." for part in parts):
        return ""
    return rel_path


def _extract_verifier_diagnostic_locations(
    text: str,
    *,
    limit: int = 24,
) -> list[dict[str, Any]]:
    """Extract concrete ``path:line[:column]`` diagnostics from verifier text."""

    locations: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for match in re.finditer(
        r"(?P<path>[^:\n;\[]+):(?P<line>\d+)(?::(?P<column>\d+))?:\s*(?P<message>[^\n;]+)",
        str(text or ""),
    ):
        rel_path = _normalize_verifier_diagnostic_path(match.group("path"))
        if not rel_path:
            continue
        try:
            line = int(match.group("line") or 0)
            column = int(match.group("column") or 0)
        except ValueError:
            continue
        if line <= 0:
            continue
        key = (rel_path, line, column)
        if key in seen:
            continue
        seen.add(key)
        message = str(match.group("message") or "").strip()
        for marker in (
            " Verifier diagnostic source context",
            " Verifier coverage rejection:",
            " Focus follow-up search",
            " Cross-rollout residual focus files:",
        ):
            marker_index = message.find(marker)
            if marker_index >= 0:
                message = message[:marker_index].strip()
                break
        locations.append(
            {
                "path": rel_path,
                "line": line,
                "column": column,
                "message": _truncate_words(message, max_words=14, max_chars=160),
            }
        )
        if len(locations) >= limit:
            break
    return locations


def _extract_verifier_diagnostic_paths(text: str, *, limit: int = 12) -> list[str]:
    """Extract concrete file paths from verifier ``path:line`` diagnostics."""

    paths: list[str] = []
    for location in _extract_verifier_diagnostic_locations(text, limit=limit * 2):
        rel_path = str(location.get("path") or "")
        if rel_path not in paths:
            paths.append(rel_path)
        if len(paths) >= limit:
            break
    return paths


def _extract_verifier_validity_focus_paths(text: str, *, limit: int = 12) -> list[str]:
    """Extract hard verifier/validity focus paths from a residual summary.

    Line diagnostics are one source of hard validity evidence. Some validators
    also produce file-level findings, such as changed-source stub residue, that
    are summarized in the verifier/validity focus sentence rather than as
    ``path:line`` diagnostics. Preserve both so follow-up repair briefs do not
    silently narrow back to lint-only work.
    """

    paths: list[str] = []

    def add_path(raw: str) -> None:
        normalized = _normalize_verifier_diagnostic_path(raw)
        if normalized and normalized not in paths:
            paths.append(normalized)

    for path in _extract_verifier_diagnostic_paths(text, limit=limit):
        add_path(path)
        if len(paths) >= limit:
            return paths

    content = str(text or "")
    lower_content = content.lower()
    markers = (
        "focus follow-up search on verifier/validity-rejected files:",
        "focus follow-up search on verifier-rejected files:",
    )
    boundaries = (
        "\n",
        " cross-rollout residual focus files:",
        " failure excerpts from the best candidate:",
        " verifier diagnostic source context from the best candidate:",
    )
    for marker in markers:
        search_from = 0
        while True:
            start = lower_content.find(marker, search_from)
            if start < 0:
                break
            list_start = start + len(marker)
            list_end = len(content)
            for boundary in boundaries:
                boundary_index = lower_content.find(boundary, list_start)
                if boundary_index >= 0:
                    list_end = min(list_end, boundary_index)
            path_list = content[list_start:list_end]
            for raw_path in path_list.split(","):
                add_path(raw_path.strip().rstrip("."))
                if len(paths) >= limit:
                    return paths
            search_from = list_end
            if search_from >= len(content):
                break
    return paths


def _verifier_repair_source_focus_paths(paths: list[Any], *, limit: int = 12) -> list[str]:
    """Return editable source paths for hard verifier-repair action targets."""

    cleaned: list[str] = []
    generated_report_names = {"rollout_report.json", "targeted_report.json"}
    for raw_path in paths:
        path = _normalize_verifier_diagnostic_path(raw_path)
        if not path:
            continue
        name = Path(path).name
        if (
            is_test_path(path, repo_relative=False)
            or is_apex_harness_path(path, repo_relative=False)
            or name in generated_report_names
        ):
            continue
        if path not in cleaned:
            cleaned.append(path)
            if len(cleaned) >= limit:
                break
    return cleaned


def _extract_additional_validity_residual_text(
    text: str,
    *,
    max_chars: int = 2200,
) -> str:
    """Return non-line hard validity diagnostics from a residual summary."""

    content = str(text or "")
    markers = (
        "Unimplemented function bodies still in the candidate patch:",
        "Public symbols present in the baseline but missing from the candidate:",
        "Verifier missing expected-test groups:",
        "Sample expected test IDs missing from the final verifier collection:",
        "Expected tests not collected by the best candidate",
        "The missing IDs include parametrized cases;",
    )
    spans: list[tuple[int, int]] = []
    for marker in markers:
        search_from = 0
        while True:
            start = content.find(marker, search_from)
            if start < 0:
                break
            end = len(content)
            for boundary in (
                " Focus follow-up search",
                " Cross-rollout residual focus files:",
                " Failure excerpts from the best candidate:",
                " Verifier diagnostic source context from the best candidate:",
            ):
                boundary_index = content.find(boundary, start + len(marker))
                if boundary_index >= 0:
                    end = min(end, boundary_index)
            spans.append((start, end))
            search_from = start + len(marker)
    merged_spans: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if not merged_spans or start > merged_spans[-1][1]:
            merged_spans.append((start, end))
        else:
            prev_start, prev_end = merged_spans[-1]
            merged_spans[-1] = (prev_start, max(prev_end, end))
    chunks = [content[start:end].strip() for start, end in merged_spans]
    rendered = " ".join(chunk for chunk in chunks if chunk).strip()
    rendered = _normalize_verifier_diagnostic_text_paths(rendered)
    if len(rendered) > max_chars:
        return rendered[:max_chars].rstrip() + "..."
    return rendered


def _normalize_verifier_diagnostic_text_paths(text: str) -> str:
    """Render candidate-workspace diagnostic paths as repo-relative paths."""

    content = str(text or "")
    if not content:
        return ""
    suffix = "|".join(re.escape(ext) for ext in _RESIDUAL_PATH_EXTENSIONS)
    pattern = re.compile(rf"(?P<path>(?:[A-Za-z]:)?/?[^\s;`'\",)]+?\.(?:{suffix}))")

    def replace(match: re.Match[str]) -> str:
        raw_path = str(match.group("path") or "")
        normalized = _normalize_verifier_diagnostic_path(raw_path)
        if normalized and normalized != raw_path:
            return normalized
        return raw_path

    return pattern.sub(replace, content)


def _format_verifier_diagnostic_location(location: dict[str, Any]) -> str:
    path = str(location.get("path") or "").strip()
    if not path:
        return ""
    try:
        line = int(location.get("line") or 0)
        column = int(location.get("column") or 0)
    except (TypeError, ValueError):
        line = 0
        column = 0
    suffix = f":{line}" if line > 0 else ""
    if column > 0:
        suffix += f":{column}"
    message = _truncate_words(location.get("message"), max_words=18, max_chars=180)
    return f"{path}{suffix}: {message}".strip().rstrip(":")


def _verifier_repair_objective_text(
    residual_summary: str,
    diagnostic_locations: list[dict[str, Any]],
) -> str:
    rendered = [
        value
        for value in (
            _format_verifier_diagnostic_location(location)
            for location in diagnostic_locations
        )
        if value
    ]
    if rendered:
        return "Verifier validity diagnostics to repair: " + "; ".join(rendered) + "."
    match = re.search(
        r"(Verifier (?:lint|static validity|prune|coverage) rejection[^.]*\.)",
        str(residual_summary or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return _truncate_words(match.group(1), max_words=80, max_chars=900)
    return "Repair hard verifier validity diagnostics."


def _extract_verifier_diagnostic_source_context(
    text: str,
    *,
    max_chars: int = 4000,
) -> str:
    marker = "Verifier diagnostic source context from the best candidate:"
    content = str(text or "")
    start = content.find(marker)
    if start < 0:
        return ""
    end_candidates = []
    for stop_marker in (
        "Verifier coverage rejection:",
        "Focus follow-up search on",
        "Cross-rollout residual focus files:",
    ):
        index = content.find(stop_marker, start + len(marker))
        if index >= 0:
            end_candidates.append(index)
    end = min(end_candidates) if end_candidates else len(content)
    context = content[start:end].strip()
    if len(context) > max_chars:
        context = context[:max_chars].rstrip() + "\n... [truncated]"
    return context


_BASELINE_COLLECTION_ERROR_BYPASS_REASON = "baseline_collection_errors_with_traceback_focus"
_LOW_PROGRESS_SCORE = 0.18
_MEANINGFUL_PROGRESS_SCORE = 0.25
_BOUNDARY_COLLAPSE_PROGRESS_SCORE = 0.35
_PROFILE_SEARCH_STAGE_NAMES = ("reproducer", "localizer", "test_writer")
_FOCUS_TEST_HIGH_AUTHORITY_PATH_TOKENS = frozenset(
    {
        "acceptance",
        "case",
        "cases",
        "e2e",
        "functional",
        "integration",
        "scenario",
        "scenarios",
        "system",
        "target",
        "targets",
        "task",
        "tasks",
    }
)
_FOCUS_TEST_SUPPORT_PATH_TOKENS = frozenset(
    {
        "_internal",
        "_util",
        "benchmark",
        "benchmarks",
        "coverage",
        "fixture",
        "fixtures",
        "harness",
        "helper",
        "helpers",
        "internal",
        "sanity",
        "support",
        "util",
        "utils",
        "vendor",
    }
)
_FOCUS_TEST_DATA_FILE_SUFFIXES = frozenset(
    {".cfg", ".ini", ".json", ".toml", ".txt", ".yaml", ".yml"}
)
_FOCUS_TEST_CODE_FILE_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".go",
        ".h",
        ".hpp",
        ".java",
        ".js",
        ".jsx",
        ".kt",
        ".kts",
        ".php",
        ".ps1",
        ".psm1",
        ".py",
        ".pyi",
        ".rb",
        ".rs",
        ".sh",
        ".scala",
        ".ts",
        ".tsx",
    }
)
_FOCUS_TEST_GENERIC_SCENARIO_BASENAMES = frozenset(
    {
        "__init__",
        "base",
        "common",
        "default",
        "index",
        "main",
    }
)
_TRACEBACK_LINE_SYMBOL_NOISE = frozenset(
    {
        "and",
        "as",
        "assert",
        "await",
        "break",
        "case",
        "catch",
        "class",
        "continue",
        "def",
        "delete",
        "do",
        "elif",
        "else",
        "except",
        "export",
        "finally",
        "for",
        "from",
        "function",
        "if",
        "import",
        "in",
        "let",
        "match",
        "new",
        "raise",
        "return",
        "switch",
        "throw",
        "try",
        "var",
        "while",
        "with",
        "yield",
    }
)


def _dedupe_preserve(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


_RESIDUAL_PATH_EXTENSIONS = (
    "py",
    "js",
    "jsx",
    "ts",
    "tsx",
    "mjs",
    "cjs",
    "go",
    "rs",
    "java",
    "kt",
    "kts",
    "rb",
    "cs",
    "php",
    "swift",
    "c",
    "cc",
    "cpp",
    "h",
    "hpp",
)
_NON_RESIDUAL_TEST_STATUSES = (" PASSED", " SKIPPED", " XFAIL", " XPASS", " DESELECTED")
_RESIDUAL_TEST_STATUSES = (" FAILED", " ERROR", " XFAILED")
_RESIDUAL_KEYWORD_NOISE = _TRACEBACK_LINE_SYMBOL_NOISE | frozenset(
    {
        "candidate",
        "collected",
        "current",
        "deselected",
        "deterministic",
        "errors",
        "failed",
        "failing",
        "fixture",
        "fixtures",
        "left",
        "line",
        "passed",
        "prior",
        "pytest",
        "reached",
        "residual",
        "right",
        "rollout",
        "selected",
        "source",
        "test",
        "tests",
        "value",
        "values",
    }
)


def _line_has_non_residual_test_status(line: str) -> bool:
    status_line = any(
        status in line for status in _NON_RESIDUAL_TEST_STATUSES + _RESIDUAL_TEST_STATUSES
    )
    return status_line and not any(status in line for status in _RESIDUAL_TEST_STATUSES)


def _residual_relevant_lines(text: str) -> list[str]:
    return [
        line
        for line in str(text or "").splitlines()
        if not _line_has_non_residual_test_status(line)
    ]


def _normalize_residual_path_hint(value: Any) -> str:
    text = str(value or "").strip().strip("`'\"")
    if not text:
        return ""
    text = text.replace("\\", "/")
    text = re.sub(r"^\./+", "", text)
    text = re.sub(r":\d+(?::\d+)?$", "", text)
    if not text or text.startswith(("http://", "https://", "/", "~")):
        return ""
    parts = Path(text.split("::", 1)[0]).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return ""
    return Path(*parts).as_posix()


def _extract_residual_test_ids(text: str, *, limit: int = 16) -> list[str]:
    suffix = "|".join(re.escape(ext) for ext in _RESIDUAL_PATH_EXTENSIONS)
    pattern = re.compile(
        rf"(?P<id>[A-Za-z0-9_.\-/]+\.(?:{suffix})(?:::[^\s,;`'\"\)]+)+)"
    )

    def normalize_identifier(raw: str) -> str:
        identifier = str(raw or "").strip().strip("`'\"")
        identifier = identifier.rstrip(".,;:")
        if identifier[:1] in "[({<" and ".py::" in identifier[1:]:
            opener = identifier[0]
            closer = {"[": "]", "(": ")", "{": "}", "<": ">"}.get(opener)
            if closer and identifier.endswith(closer):
                identifier = identifier[1:-1].strip()
            else:
                identifier = identifier[1:].strip()
        while identifier.endswith(")") and identifier.count("(") < identifier.count(")"):
            identifier = identifier[:-1].rstrip()
        while identifier.endswith("]") and identifier.count("[") < identifier.count("]"):
            identifier = identifier[:-1].rstrip()
        while identifier.endswith("}") and identifier.count("{") < identifier.count("}"):
            identifier = identifier[:-1].rstrip()
        return identifier.rstrip(".,;:")

    section_lines: list[str] = []
    other_lines: list[str] = []
    in_structured_id_section = False
    for line in _residual_relevant_lines(text):
        stripped = line.strip()
        if stripped.startswith("### Residual failing test IDs"):
            in_structured_id_section = True
            continue
        if in_structured_id_section and stripped.startswith("### "):
            in_structured_id_section = False
        if in_structured_id_section:
            section_lines.append(line)
        else:
            other_lines.append(line)

    collected: list[str] = []
    # Prefer the structured failed_tests list from quick verification over prose
    # excerpts; prose clusters often wrap ids in brackets or truncate messages.
    for line in section_lines + other_lines:
        for match in pattern.finditer(line):
            identifier = normalize_identifier(match.group("id"))
            if identifier:
                collected.append(identifier)
    return _dedupe_preserve(collected)[:limit]


def _extract_residual_paths_from_text(text: str, *, limit: int = 24) -> list[str]:
    suffix = "|".join(re.escape(ext) for ext in _RESIDUAL_PATH_EXTENSIONS)
    pattern = re.compile(rf"(?P<path>[A-Za-z0-9_.\-/]+\.(?:{suffix}))(?::\d+)?")
    paths = [
        normalized
        for line in _residual_relevant_lines(text)
        for match in pattern.finditer(line)
        for normalized in [_normalize_residual_path_hint(match.group("path"))]
        if normalized
    ]
    return _dedupe_preserve(paths)[:limit]


def _extract_residual_keyword_hints(
    text: str,
    residual_test_ids: list[str],
    *,
    limit: int = 32,
) -> list[str]:
    keywords: list[str] = []
    for test_id in residual_test_ids:
        node_parts = str(test_id or "").split("::")
        keywords.append(node_parts[0])
        for part in node_parts[1:]:
            symbol = re.sub(r"\[[^\]]*\]", "", part).strip()
            if not symbol:
                continue
            keywords.append(symbol)
            if symbol.startswith("test_"):
                keywords.append(symbol[len("test_") :])
    for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", str(text or "")):
        token = match.group(0)
        lowered = token.lower()
        if lowered in _RESIDUAL_KEYWORD_NOISE:
            continue
        keywords.append(token)
    return _dedupe_preserve(keywords)[:limit]


def _test_file_from_residual_test_id(test_id: str) -> str:
    return _normalize_residual_path_hint(str(test_id or "").partition("::")[0])


def _residual_text_summary(text: str, *, limit: int = 520) -> str:
    cleaned = re.sub(
        r"\s+",
        " ",
        _strip_residual_followup_text("\n".join(_residual_relevant_lines(text))),
    ).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "..."


def _task_state_item_residual_overlap(
    item: Any,
    *,
    residual_files: set[str],
    residual_test_ids: set[str],
    residual_text: str,
) -> bool:
    if not isinstance(item, dict):
        return False

    item_test_ids = {
        str(test_id).strip()
        for test_id in list(item.get("test_ids") or [])
        if str(test_id).strip()
    }
    if item_test_ids and residual_test_ids and item_test_ids & residual_test_ids:
        return True
    description = " ".join(
        str(item.get(key) or "")
        for key in ("description", "summary", "rationale", "hypothesis_description")
    )
    if residual_test_ids and any(test_id in description for test_id in residual_test_ids):
        return True
    if residual_test_ids:
        return False

    item_paths: set[str] = set()
    for key in (
        "file_paths",
        "focus_files",
        "risk_files",
        "relevant_files",
        "action_file_paths",
        "owned_files",
    ):
        for value in list(item.get(key) or []):
            normalized = _normalize_residual_path_hint(value)
            if normalized:
                item_paths.add(normalized)
    for test_id in item_test_ids:
        path = _test_file_from_residual_test_id(test_id)
        if path:
            item_paths.add(path)
    if item_paths and residual_files and item_paths & residual_files:
        return True

    if residual_files and any(path in description for path in residual_files):
        return True
    return bool(description and description in residual_text)


def _filter_task_state_items_to_residual(
    items: Any,
    *,
    residual_files: set[str],
    residual_test_ids: set[str],
    residual_text: str,
    limit: int = 8,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for item in list(items or []):
        if _task_state_item_residual_overlap(
            item,
            residual_files=residual_files,
            residual_test_ids=residual_test_ids,
            residual_text=residual_text,
        ):
            filtered.append(dict(item))
    return filtered[:limit]


def _recenter_task_state_context_on_residual(
    task_state_context: Optional[dict[str, Any]],
    *,
    residual_summary: str,
    residual_focus_files: list[str],
) -> dict[str, Any]:
    """Make current residual evidence outrank stale frontier memory.

    Follow-up plans branch from the best verified partial workspace. Their
    active subgoal must be the latest verification residual, not older
    advisory hypotheses from the initial task graph.
    """

    context = dict(task_state_context) if isinstance(task_state_context, dict) else {}
    residual_text = str(residual_summary or "")
    residual_test_ids = _extract_residual_test_ids(residual_text)
    residual_files = _dedupe_preserve(
        [
            normalized
            for value in list(residual_focus_files or [])
            for normalized in [_normalize_residual_path_hint(value)]
            if normalized
        ]
        + [_test_file_from_residual_test_id(test_id) for test_id in residual_test_ids]
        + _extract_residual_paths_from_text(residual_text)
    )
    if not residual_text.strip() and not residual_files and not residual_test_ids:
        return context

    residual_file_set = set(residual_files)
    residual_test_set = set(residual_test_ids)
    concise_residual = _residual_text_summary(residual_text)

    current_obligations: list[dict[str, Any]] = []
    for test_id in residual_test_ids[:8]:
        path = _test_file_from_residual_test_id(test_id)
        current_obligations.append(
            {
                "description": f"Resolve current residual failing test: {test_id}",
                "source": "residual_followup",
                "priority": 1.0,
                "file_paths": [path] if path else [],
                "test_ids": [test_id],
            }
        )
    if not current_obligations and residual_files:
        current_obligations.append(
            {
                "description": "Repair the current verified residual around "
                + ", ".join(residual_files[:4]),
                "source": "residual_followup",
                "priority": 1.0,
                "file_paths": residual_files[:8],
                "test_ids": [],
            }
        )

    retained_obligations = _filter_task_state_items_to_residual(
        context.get("open_obligations"),
        residual_files=residual_file_set,
        residual_test_ids=residual_test_set,
        residual_text=residual_text,
        limit=6,
    )
    retained_hypotheses = _filter_task_state_items_to_residual(
        context.get("supported_hypotheses"),
        residual_files=residual_file_set,
        residual_test_ids=residual_test_set,
        residual_text=residual_text,
        limit=4,
    )

    frontier_targets: list[dict[str, Any]] = []
    for index, test_id in enumerate(residual_test_ids[:6]):
        path = _test_file_from_residual_test_id(test_id)
        frontier_targets.append(
            {
                "target_id": f"residual_{index}",
                "kind": "residual",
                "description": f"Resolve current residual failing test: {test_id}",
                "obligation_description": f"Resolve current residual failing test: {test_id}",
                "hypothesis_description": (
                    "The latest verification residual is authoritative for this follow-up."
                ),
                "rationale": "Current verified residual supersedes older frontier memory.",
                "frontier_score": 1.0,
                "uncertainty_score": 0.25,
                "file_paths": residual_files[:4] if residual_files else ([path] if path else []),
                "test_ids": [test_id],
                "symbols": [],
            }
        )
    if not frontier_targets and residual_files:
        frontier_targets.append(
            {
                "target_id": "residual_files",
                "kind": "residual",
                "description": "Repair current residual around " + ", ".join(residual_files[:4]),
                "obligation_description": "Repair current verified residual.",
                "hypothesis_description": (
                    "The latest verification residual is authoritative for this follow-up."
                ),
                "rationale": "Current verified residual supersedes older frontier memory.",
                "frontier_score": 1.0,
                "uncertainty_score": 0.25,
                "file_paths": residual_files[:8],
                "test_ids": [],
                "symbols": [],
            }
        )

    recentered = dict(context)
    if concise_residual:
        recentered["summary"] = "Current residual follow-up frontier: " + concise_residual
    recentered["focus_files"] = residual_files[:12]
    recentered["unresolved_test_ids"] = residual_test_ids[:8]
    recentered["open_obligations"] = _dedupe_task_state_dicts(
        current_obligations + retained_obligations,
        key_fields=("description", "source"),
        limit=8,
    )
    recentered["supported_hypotheses"] = retained_hypotheses
    recentered["frontier_targets"] = frontier_targets
    recentered["residual_followup_context_recentered"] = True
    recentered["residual_followup_test_ids"] = residual_test_ids[:12]

    blackboard = recentered.get("blackboard")
    if isinstance(blackboard, dict):
        records = [dict(item) for item in list(blackboard.get("records") or []) if isinstance(item, dict)]
        records.append(
            {
                "record_id": "residual_followup_frontier",
                "record_type": "failure_frontier",
                "description": "Current residual follow-up frontier"
                + (f": {concise_residual}" if concise_residual else "."),
                "provenance": "verified",
                "confidence": 0.92,
                "source": "residual_followup",
                "file_paths": residual_files[:8],
                "symbols": [],
                "test_ids": residual_test_ids[:12],
                "payload": {
                    "residual_followup": True,
                    "failed_tests": residual_test_ids[:12],
                },
            }
        )
        recentered["blackboard"] = {**blackboard, "records": records}

    return recentered


def _dedupe_task_state_dicts(
    items: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
    limit: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for item in items:
        key = tuple(str(item.get(field) or "") for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) >= limit:
            break
    return result


def _repo_language_hints(
    repo_context: RepoContext,
    *,
    max_languages: int = 6,
) -> list[str]:
    languages = [
        str(file_info.language).strip().lower()
        for file_info in list(repo_context.files or [])
        if str(getattr(file_info, "language", "") or "").strip()
    ]
    return _dedupe_preserve(languages)[:max_languages]


def _file_language_hints(
    repo_context: RepoContext,
    paths: list[str],
    *,
    max_languages: int = 6,
) -> list[str]:
    languages: list[str] = []
    for raw_path in list(paths or []):
        path = str(raw_path or "").strip()
        if not path:
            continue
        if "::" in path:
            path = path.split("::", 1)[0]
        file_info = repo_context.get_file_info(path)
        if file_info is None:
            continue
        language = str(getattr(file_info, "language", "") or "").strip().lower()
        if language:
            languages.append(language)
    return _dedupe_preserve(languages)[:max_languages]


def _collapse_whitespace(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _truncate_words(
    text: Any,
    *,
    max_words: int,
    max_chars: Optional[int] = None,
) -> str:
    content = _collapse_whitespace(text)
    if not content:
        return ""
    words = content.split(" ")
    if len(words) > max_words:
        content = " ".join(words[:max_words]).rstrip(" ,;:")
        if content and content[-1] not in ".!?":
            content += "..."
    if isinstance(max_chars, int) and max_chars > 0 and len(content) > max_chars:
        content = content[:max_chars].rstrip(" ,;:")
        if content and content[-1] not in ".!?":
            content += "..."
    return content


def _compact_string_list(
    values: list[Any] | tuple[Any, ...] | None,
    *,
    max_items: int,
    max_words: int,
    max_chars: Optional[int] = None,
) -> list[str]:
    compacted: list[str] = []
    for value in list(values or []):
        item = _truncate_words(
            value,
            max_words=max_words,
            max_chars=max_chars,
        )
        if not item or item in compacted:
            continue
        compacted.append(item)
        if len(compacted) >= max_items:
            break
    return compacted


_DELEGATION_STAGE_ALIASES = {
    "patcher": "patcher",
    "patch": "patcher",
    "repair": "patcher",
    "solver": "patcher",
    "implementation": "patcher",
    "implement": "patcher",
    "editing": "patcher",
    "validation": "patcher",
    "integration": "patcher",
    "integration_validation": "patcher",
    "reproducer": "reproducer",
    "reproduction": "reproducer",
    "localizer": "localizer",
    "localization": "localizer",
    "test_writer": "test_writer",
    "test_generation": "test_writer",
}


def _normalize_delegation_allowed_stages(values: list[Any] | tuple[Any, ...] | None) -> list[str]:
    normalized: list[str] = []
    for value in list(values or ["patcher"]):
        token = str(value).strip().lower()
        if not token:
            continue
        token = token.replace("-", "_").replace(" ", "_")
        mapped = _DELEGATION_STAGE_ALIASES.get(token)
        if mapped:
            normalized.append(mapped)
            continue
        if any(fragment in token for fragment in ("implement", "patch", "repair")):
            normalized.append("patcher")
            continue
        if any(fragment in token for fragment in ("validat", "integrat")):
            normalized.append("patcher")
            continue
        if "repro" in token:
            normalized.append("reproducer")
            continue
        if "local" in token:
            normalized.append("localizer")
            continue
        if "test" in token:
            normalized.append("test_writer")
            continue
        normalized.append(token)
    normalized = _dedupe_preserve(normalized)
    return normalized or ["patcher"]


def _repo_map_files(repo_focus_map: str) -> list[str]:
    files: list[str] = []
    for line in repo_focus_map.splitlines():
        match = re.match(r"^##\s+(.+?)\s+\(", line.strip())
        if match:
            files.append(match.group(1).strip())
    return _dedupe_preserve(files)


def _baseline_values(
    baseline_result: Optional[Any],
    field_name: str,
) -> Any:
    if baseline_result is None:
        return None
    values = getattr(baseline_result, field_name, None)
    if values is None and isinstance(baseline_result, dict):
        values = baseline_result.get(field_name)
    return values


def _baseline_output(
    baseline_result: Optional[Any],
) -> str:
    output = _baseline_values(baseline_result, "output")
    return normalize_terminal_output(output) if isinstance(output, str) else ""


def _looks_like_collection_failure_output(output: str) -> bool:
    lowered = output.lower()
    return any(
        token in lowered
        for token in (
            "importerror while loading conftest",
            "conftestimportfailure",
            "importerror while importing test module",
            "error collecting",
        )
    )


def _baseline_test_ids(
    baseline_result: Optional[Any],
    field_name: str,
) -> list[str]:
    values = _baseline_values(baseline_result, field_name)
    if not values:
        return []
    return [value for value in sorted(values) if value and value != "<full-suite>"]


def _baseline_test_count(
    baseline_result: Optional[Any],
    sample: list[str],
    field_name: str,
) -> int:
    values = _baseline_values(baseline_result, field_name)
    if not values:
        return 0
    if isinstance(values, (set, list, tuple)):
        filtered = [value for value in values if value and value != "<full-suite>"]
        return len(filtered) if filtered else len(values)
    return len(sample)


def _strip_pytest_node_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.split("::", 1)[0].strip()


def _looks_like_collection_error_test_id(test_id: str) -> bool:
    normalized = str(test_id or "").strip()
    if not normalized or not normalized.endswith(".py") or "::" in normalized:
        return False
    lowered_path = normalized.lower()
    name = Path(normalized).name.lower()
    parts = {part.lower() for part in Path(normalized).parts}
    return (
        "test" in parts
        or "tests" in parts
        or "__tests__" in parts
        or name.startswith("test_")
        or name.endswith("_test.py")
        or "/tests/" in lowered_path
    )


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _looks_like_completion_task(issue_description: str) -> bool:
    lowered = issue_description.lower()
    return any(
        pattern in lowered
        for pattern in (
            "intentionally incomplete",
            "implement missing",
            "missing functionality",
            "repository completion",
            "complete the implementation",
            "complete the library",
            "fill in the missing",
            "missing library functionality",
            "long-horizon repository completion",
        )
    )


def _mentions_public_api_contract(issue_description: str) -> bool:
    lowered = issue_description.lower()
    return any(
        pattern in lowered
        for pattern in (
            "public api",
            "api contract",
            "backward compatible",
            "without changing the public api",
            "callers",
            "library behavior",
            "contract",
        )
    )


def _looks_like_timeout_or_stall_error(text: str) -> bool:
    lowered = text.lower()
    return (
        "timed out" in lowered
        or "stalled after" in lowered
        or "without observable progress" in lowered
    )


def _interpolate_rollout_count(minimum: int, maximum: int, ratio: float) -> int:
    if maximum <= minimum:
        return minimum
    bounded_ratio = max(0.0, min(ratio, 1.0))
    return minimum + int(round(bounded_ratio * (maximum - minimum)))


def _infer_rollout_brief_mode(title: str, goal: str) -> str:
    title_goal = f"{title} {goal}".lower()
    if "test" in title_goal:
        return "test_rooted"
    if "dependency" in title_goal or "neighbor" in title_goal:
        return "dependency_trace"
    if any(token in title_goal for token in ("regression", "edge", "invariant")):
        return "invariant_guard"
    if any(token in title_goal for token in ("api", "contract")):
        return "api_contract"
    return "surgical"


def _controller_action_is_default_like(action: Any) -> bool:
    if not isinstance(action, ControllerAction):
        return True
    return action.to_dict() == ControllerAction().to_dict()


class Primitive(str, Enum):
    """Composable orchestration primitives used by the planner/orchestrator."""

    REACT = "react"
    GTR = "generate_test_repair"
    PLAN_EXEC = "plan_execute"
    MCTS = "tree_search"


@dataclass
class RolloutBrief:
    """One focused brief for a rollout."""

    title: str
    goal: str
    focus_files: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    prompt_hint: str = ""
    agent_mode: AgentMode = AgentMode.ADAPTIVE
    search_policy: dict[str, Any] = field(default_factory=dict)
    delegation_policy: dict[str, Any] = field(default_factory=dict)
    controller_action: ControllerAction = field(default_factory=ControllerAction)

    def to_dict(self) -> dict[str, Any]:
        action = self.resolved_controller_action()
        return {
            "title": self.title,
            "goal": self.goal,
            "focus_files": list(self.focus_files),
            "hypotheses": list(self.hypotheses),
            "success_criteria": list(self.success_criteria),
            "prompt_hint": self.prompt_hint,
            "agent_mode": self.agent_mode.value,
            "search_policy": dict(self.search_policy),
            "delegation_policy": dict(self.delegation_policy),
            "controller_action": action.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RolloutBrief":
        payload = dict(data)
        payload["agent_mode"] = AgentMode(payload.get("agent_mode", AgentMode.ADAPTIVE.value))
        payload["controller_action"] = coerce_controller_action(
            payload.get("controller_action"),
            fallback_policy=dict(payload.get("search_policy") or {}),
            default_files=list(payload.get("focus_files") or []),
        )
        valid_keys = {f.name for f in fields(cls)}
        brief = cls(**{k: v for k, v in payload.items() if k in valid_keys})
        brief.set_controller_action(
            brief.controller_action,
            merge_policy=dict(brief.search_policy or {}),
        )
        return brief

    def resolved_controller_action(self) -> ControllerAction:
        policy = self.search_policy if isinstance(self.search_policy, dict) else {}
        if (
            isinstance(policy, dict)
            and not str(policy.get("mode") or "").strip()
            and not isinstance(policy.get("controller_action"), dict)
            and _controller_action_is_default_like(self.controller_action)
        ):
            inferred_mode = _infer_rollout_brief_mode(self.title, self.goal)
            if inferred_mode:
                policy = dict(policy)
                policy["mode"] = inferred_mode
        action, normalized_policy = sync_controller_action_payload(
            policy,
            action=None if policy else self.controller_action,
            default_files=list(self.focus_files or []),
        )
        self.controller_action = action
        self.search_policy = normalized_policy
        return action

    def set_controller_action(
        self,
        action: Any,
        *,
        merge_policy: Optional[dict[str, Any]] = None,
    ) -> ControllerAction:
        typed_action, normalized_policy = sync_controller_action_payload(
            merge_policy if isinstance(merge_policy, dict) else self.search_policy,
            action=action,
            default_files=list(self.focus_files or []),
        )
        self.controller_action = typed_action
        self.search_policy = normalized_policy
        return typed_action

    def delegation_enabled(
        self,
        stage_name: Optional[str] = None,
    ) -> bool:
        policy = self.delegation_policy if isinstance(self.delegation_policy, dict) else {}
        if not bool(policy.get("enabled")):
            return False
        allowed_stages = set(
            _normalize_delegation_allowed_stages(list(policy.get("allowed_stages") or []))
        )
        if stage_name and allowed_stages:
            requested_stage = _normalize_delegation_allowed_stages([stage_name])[0]
            return requested_stage in allowed_stages
        return True


def _normalized_rollout_signature_tokens(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    text = str(value or "").strip()
    if not text:
        return ()
    return tuple(token for token in text.split("|") if token)


def _rollout_brief_profile_key(
    brief: RolloutBrief,
) -> str:
    brief.resolved_controller_action()
    policy = brief.search_policy if isinstance(brief.search_policy, dict) else {}
    signature = _normalized_rollout_signature_tokens(
        policy.get("rollout_profile_signature") or policy.get("rollout_route_signature")
    )
    return "|".join(signature)


def _rollout_brief_allocation_key(
    brief: RolloutBrief,
) -> str:
    action = brief.resolved_controller_action()
    policy = brief.search_policy if isinstance(brief.search_policy, dict) else {}
    explicit = str(action.allocator_arm or policy.get("allocator_arm") or "").strip()
    profile_key = _rollout_brief_profile_key(brief)
    if explicit:
        explicit_profile = ""
        for fragment in explicit.split(" | "):
            if fragment.startswith("profile="):
                explicit_profile = fragment.removeprefix("profile=").strip()
                break
        if not explicit_profile or not profile_key or explicit_profile == profile_key:
            return explicit

    title = str(brief.title or "").strip() or "rollout"
    fragments = [title]
    if profile_key:
        fragments.append(f"profile={profile_key}")
    family_index = policy.get("family_index")
    if isinstance(family_index, int):
        fragments.append(f"family={family_index}")
    variant_index = policy.get("variant_index")
    if isinstance(variant_index, int):
        fragments.append(f"variant={variant_index}")
    mode = str(policy.get("mode") or "").strip().lower()
    if mode:
        fragments.append(f"mode={mode}")
    return " | ".join(fragments)


def _rollout_result_allocation_key(
    rollout_result: Any,
) -> str:
    explicit = str(getattr(rollout_result, "allocation_key", "") or "").strip()
    if explicit:
        return explicit
    return str(getattr(rollout_result, "plan_title", "") or "").strip()


def _llm_route_family(llm_config: Optional[LLMConfig]) -> str:
    if not isinstance(llm_config, LLMConfig):
        return "unknown"
    backend = llm_config.backend
    model = str(llm_config.model or "").strip().lower()
    if backend == LLMBackend.CODEX_CLI:
        return "codex"
    if backend == LLMBackend.CLAUDE_CLI:
        return "claude"
    if backend == LLMBackend.GEMINI_CLI:
        return "gemini"
    if backend in {LLMBackend.OPENCODE_CLI, LLMBackend.METACODE_CLI}:
        if model.startswith("meta/") or "avocado" in model:
            return "meta"
        if model.startswith("openai/"):
            return "openai"
        if model.startswith("anthropic/") or "claude" in model:
            return "anthropic"
        return "opencode"
    if "gpt" in model or "codex" in model:
        return "codex"
    if "claude" in model or model == "opus":
        return "claude"
    if "gemini" in model:
        return "gemini"
    if model.startswith("meta/") or "avocado" in model:
        return "meta"
    return backend.value


@dataclass
class TestContext:
    """Structured view of visible test expectations for prompt construction."""

    command: Optional[str] = None
    summary: str = ""
    planner_invariants: list[str] = field(default_factory=list)
    focus_test_files: list[str] = field(default_factory=list)
    incomplete_test_files: list[str] = field(default_factory=list)
    source_focus_files: list[str] = field(default_factory=list)
    incomplete_source_files: list[str] = field(default_factory=list)
    terminal_source_files: list[str] = field(default_factory=list)
    terminal_reference_symbols: list[str] = field(default_factory=list)
    exception_summaries: list[str] = field(default_factory=list)
    failing_test_ids: list[str] = field(default_factory=list)
    passing_test_ids: list[str] = field(default_factory=list)
    failing_test_count: int = 0
    passing_test_count: int = 0
    expectations: list[str] = field(default_factory=list)
    expected_test_count: int = 0
    expected_test_ids: list[str] = field(default_factory=list)
    test_inventory_framework: str = ""
    test_inventory_language: str = ""
    test_inventory_source: str = ""
    test_collection_command: str = ""
    evidence_mode: str = EVIDENCE_MODE_NO_SUITE_VISIBLE
    evidence_policy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "summary": self.summary,
            "planner_invariants": list(self.planner_invariants),
            "focus_test_files": list(self.focus_test_files),
            "incomplete_test_files": list(self.incomplete_test_files),
            "source_focus_files": list(self.source_focus_files),
            "incomplete_source_files": list(self.incomplete_source_files),
            "terminal_source_files": list(self.terminal_source_files),
            "terminal_reference_symbols": list(self.terminal_reference_symbols),
            "exception_summaries": list(self.exception_summaries),
            "failing_test_ids": list(self.failing_test_ids),
            "passing_test_ids": list(self.passing_test_ids),
            "failing_test_count": self.failing_test_count,
            "passing_test_count": self.passing_test_count,
            "expectations": list(self.expectations),
            "expected_test_count": self.expected_test_count,
            "expected_test_ids": list(self.expected_test_ids),
            "test_inventory_framework": self.test_inventory_framework,
            "test_inventory_language": self.test_inventory_language,
            "test_inventory_source": self.test_inventory_source,
            "test_collection_command": self.test_collection_command,
            "evidence_mode": self.evidence_mode,
            "evidence_policy": dict(self.evidence_policy),
        }

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "summary": self.summary,
            "planner_invariants": list(self.planner_invariants),
            "focus_test_files": list(self.focus_test_files),
            "incomplete_test_files": list(self.incomplete_test_files),
            "source_focus_files": list(self.source_focus_files),
            "incomplete_source_files": list(self.incomplete_source_files),
            "terminal_source_files": list(self.terminal_source_files),
            "terminal_reference_symbols": list(self.terminal_reference_symbols),
            "exception_summaries": list(self.exception_summaries),
            "failing_test_ids": list(self.failing_test_ids),
            "passing_test_ids": list(self.passing_test_ids),
            "failing_test_count": self.failing_test_count,
            "passing_test_count": self.passing_test_count,
            "expectations": list(self.expectations),
            "expected_test_count": 0,
            "expected_test_ids": [],
            "test_inventory_framework": self.test_inventory_framework,
            "test_inventory_language": self.test_inventory_language,
            "test_inventory_source": "",
            "test_collection_command": self.test_collection_command,
            "evidence_mode": self.evidence_mode,
            "evidence_policy": dict(self.evidence_policy),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestContext":
        payload = dict(data)
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in valid_keys})


TestContext.__test__ = False


@dataclass
class _TracebackSignal:
    source_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    terminal_source_files: list[str] = field(default_factory=list)
    referenced_source_files: list[str] = field(default_factory=list)
    referenced_symbols: list[str] = field(default_factory=list)
    exception_summaries: list[str] = field(default_factory=list)


_TracebackSignal.__test__ = False


@dataclass
class IssuePlan:
    """Plan shared across rollouts."""

    summary: str
    keywords: list[str]
    relevant_files: list[str]
    risk_files: list[str]
    success_criteria: list[str]
    rollout_briefs: list[RolloutBrief]
    repo_focus_map: str
    planner_source: str = "heuristic"
    planner_tokens: int = 0
    difficulty_estimate: Optional[float] = None
    recommended_rollouts: Optional[int] = None
    orchestration_primitives: list[str] = field(default_factory=list)
    allocator_features: dict[str, Any] = field(default_factory=dict)
    unsolvable_reason: Optional[str] = None
    test_context: TestContext = field(default_factory=TestContext)
    evaluation_constraints: EvaluationConstraints = field(default_factory=EvaluationConstraints)
    task_regime: TaskRegimeProfile = field(default_factory=TaskRegimeProfile)
    task_state_context: dict[str, Any] = field(default_factory=dict)
    planner_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "keywords": list(self.keywords),
            "relevant_files": list(self.relevant_files),
            "risk_files": list(self.risk_files),
            "success_criteria": list(self.success_criteria),
            "rollout_briefs": [brief.to_dict() for brief in self.rollout_briefs],
            "repo_focus_map": self.repo_focus_map,
            "planner_source": self.planner_source,
            "planner_tokens": self.planner_tokens,
            "difficulty_estimate": self.difficulty_estimate,
            "recommended_rollouts": self.recommended_rollouts,
            "orchestration_primitives": list(self.orchestration_primitives),
            "allocator_features": dict(self.allocator_features),
            "unsolvable_reason": self.unsolvable_reason,
            "test_context": self.test_context.to_dict(),
            "evaluation_constraints": self.evaluation_constraints.to_dict(),
            "task_regime": self.task_regime.to_dict(),
            "task_state_context": dict(self.task_state_context),
            "planner_metadata": dict(self.planner_metadata),
        }

    def to_artifact_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["test_context"] = self.test_context.to_artifact_dict()
        payload["evaluation_constraints"] = self.evaluation_constraints.to_artifact_dict()
        return payload

    def save(self, path: str | Path, *, artifact_safe: bool = False) -> None:
        payload = self.to_artifact_dict() if artifact_safe else self.to_dict()
        from apex.evaluation.checkpointing import atomic_write_json

        atomic_write_json(Path(path), payload)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IssuePlan":
        payload = dict(data)
        payload["rollout_briefs"] = [
            RolloutBrief.from_dict(b) for b in payload.get("rollout_briefs", [])
        ]
        payload["test_context"] = TestContext.from_dict(payload.get("test_context") or {})
        payload["evaluation_constraints"] = EvaluationConstraints.from_dict(
            payload.get("evaluation_constraints") or {}
        )
        payload["task_regime"] = TaskRegimeProfile.from_dict(payload.get("task_regime") or {})
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in valid_keys})

    @classmethod
    def load(cls, path: str | Path) -> "IssuePlan":
        return cls.from_dict(json.loads(Path(path).read_text()))


_PLAN_TOOL = ToolDefinition(
    name="submit_plan",
    description=(
        "Submit a rollout plan using only files that appear in the provided repository "
        "context. Return the requested number of rollout briefs when feasible; for large "
        "rollout budgets, you may return a smaller set of materially different brief families "
        "that the runtime can expand."
    ),
    parameters={
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "relevant_files": {"type": "array", "items": {"type": "string"}},
            "risk_files": {"type": "array", "items": {"type": "string"}},
            "success_criteria": {"type": "array", "items": {"type": "string"}},
            "rollout_briefs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "goal": {"type": "string"},
                        "focus_files": {"type": "array", "items": {"type": "string"}},
                        "hypotheses": {"type": "array", "items": {"type": "string"}},
                        "success_criteria": {"type": "array", "items": {"type": "string"}},
                        "prompt_hint": {"type": "string"},
                        "agent_mode": {
                            "type": "string",
                            "enum": [
                                AgentMode.FULL_SOLVER.value,
                                AgentMode.SCAFFOLDED.value,
                                AgentMode.ADAPTIVE.value,
                            ],
                        },
                        "delegation_policy": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "allowed_stages": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "max_tasks": {"type": "integer"},
                                "parallelism": {"type": "integer"},
                                "reason": {"type": "string"},
                                "subtasks": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "title": {"type": "string"},
                                            "kind": {"type": "string"},
                                            "objective": {"type": "string"},
                                            "owned_files": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "focus_files": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "forbidden_files": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "interface_symbols": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "owned_symbols": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "edit_spans": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "file_path": {"type": "string"},
                                                        "symbol": {"type": "string"},
                                                        "start_line": {"type": "integer"},
                                                        "end_line": {"type": "integer"},
                                                    },
                                                    "required": ["file_path"],
                                                },
                                            },
                                            "assumptions": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "escalation_triggers": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "depends_on": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "validation_targets": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "deliverable": {"type": "string"},
                                        },
                                        "required": ["title"],
                                    },
                                },
                            },
                        },
                    },
                    "required": ["title", "goal"],
                },
            },
        },
        "required": ["summary", "relevant_files", "rollout_briefs"],
    },
)

_CLI_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "relevant_files": {"type": "array", "items": {"type": "string"}},
        "risk_files": {"type": "array", "items": {"type": "string"}},
        "success_criteria": {"type": "array", "items": {"type": "string"}},
        "rollout_briefs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "goal": {"type": "string"},
                    "focus_files": {"type": "array", "items": {"type": "string"}},
                    "hypotheses": {"type": "array", "items": {"type": "string"}},
                    "success_criteria": {"type": "array", "items": {"type": "string"}},
                    "prompt_hint": {"type": "string"},
                    "agent_mode": {
                        "type": "string",
                        "enum": [
                            AgentMode.FULL_SOLVER.value,
                            AgentMode.SCAFFOLDED.value,
                            AgentMode.ADAPTIVE.value,
                        ],
                    },
                    "delegation_policy": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "allowed_stages": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "max_tasks": {"type": "integer"},
                            "parallelism": {"type": "integer"},
                            "reason": {"type": "string"},
                            "subtasks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "kind": {"type": "string"},
                                        "objective": {"type": "string"},
                                        "owned_files": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "forbidden_files": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "owned_symbols": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "edit_spans": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "file_path": {"type": "string"},
                                                    "symbol": {"type": "string"},
                                                    "start_line": {"type": "integer"},
                                                    "end_line": {"type": "integer"},
                                                },
                                                "required": ["file_path"],
                                            },
                                        },
                                        "depends_on": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "deliverable": {"type": "string"},
                                    },
                                    "required": ["title"],
                                },
                            },
                        },
                    },
                },
                "required": ["title", "goal"],
            },
        },
    },
    "required": ["summary", "relevant_files", "rollout_briefs"],
}

_TEST_GENERATION_DESIGN_SCHEMA = {
    "type": "object",
    "properties": {
        "task_contract": {
            "type": "object",
            "properties": {
                "problem_statement": {"type": "string"},
                "acceptance_requirements": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "interface_specification": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        "milestones": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "milestone_id": {"type": "string"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "acceptance_requirements": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "objective_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "validation_level": {
                        "type": "string",
                        "enum": ["core", "iso", "strict"],
                    },
                    "pipeline_stages": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["milestone_id", "title"],
            },
        },
        "test_objectives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "objective_id": {"type": "string"},
                    "milestone_id": {"type": "string"},
                    "objective": {"type": "string"},
                    "acceptance_requirements": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "interface_specification": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "contract_targets": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "contract_axes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "objective_status": {"type": "string"},
                },
                "required": ["objective_id", "milestone_id", "objective"],
            },
        },
        "replan_summary": {"type": "string"},
    },
    "required": ["task_contract", "milestones", "test_objectives"],
}

_TEST_GENERATION_DESIGN_TOOL = ToolDefinition(
    name="submit_test_generation_design",
    description=(
        "Submit a milestone-driven test-generation design with a task contract "
        "and objective mapping."
    ),
    parameters=_TEST_GENERATION_DESIGN_SCHEMA,
)


def _validate_planner_parsed_json(
    parsed_json: Any,
    *,
    schema: Optional[dict[str, Any]] = None,
    required_keys: Optional[list[str]] = None,
) -> Optional[str]:
    """Validate a planner's structured output before accepting it.

    Reuses the in-house schema verifier from ``apex.core.llm`` (no jsonschema
    dependency). When ``schema`` is provided it is checked structurally; otherwise
    a lightweight required-keys check is applied for the ``IssuePlan`` fields the
    caller then reads.

    Returns an error string when the output is structurally invalid, or ``None``
    when it satisfies the constraints.

    FAIL OPEN: if validation itself raises for any reason, this returns ``None``
    (accept the plan), preserving prior behavior.
    """

    try:
        if not isinstance(parsed_json, dict):
            return "planner output must be a JSON object."
        if isinstance(schema, dict):
            error = _verify_value_against_tool_schema(
                parsed_json,
                schema,
                path="plan",
            )
            if error:
                return error
        if required_keys:
            for raw_key in required_keys:
                key = str(raw_key)
                if key not in parsed_json:
                    return f"missing required plan key: {key}."
        return None
    except Exception:  # pragma: no cover - defensive fail-open guard.
        # Validation must never block a plan if the validator itself errors.
        return None


@dataclass
class PlanningDecision:
    """Execution strategy chosen before rollout planning begins."""

    rollout_count: int
    difficulty_estimate: float
    features: dict[str, Any] = field(default_factory=dict)
    primitives: list[Primitive] = field(default_factory=list)
    agent_mode: AgentMode = AgentMode.ADAPTIVE
    unsolvable_reason: Optional[str] = None
    portfolio_profile_count: int = 1
    portfolio_rollout_floor: int = 1
    portfolio_rollout_floor_applied: bool = False
    component_ablation: dict[str, Any] = field(default_factory=dict)
    # WS3B: dispatch one speculative seed first and accept on an authoritative
    # full-scope pass, else fan out the full wave. Set from difficulty when
    # adaptive allocation is off; NEVER lowers rollout_count (no-cost-reduction).
    speculative_first_attempt: bool = False


class BanditRolloutAllocator:
    """Heuristic contextual-bandit style allocator for rollout count."""

    DEFAULT_BUCKETS = (1, 4, 8, 16)
    MAX_DIFFICULTY_SCORE = 11
    _UNSOLVABLE_PATTERNS = {
        "production access": "Issue appears to require production-only access.",
        "prod access": "Issue appears to require production-only access.",
        "requires credentials": "Issue appears to require credentials unavailable to the agent.",
        "manual ui": "Issue appears to require manual UI interaction beyond the current toolset.",
        "cannot run locally": "Issue appears to require an environment that cannot be reproduced locally.",
        "external service account": "Issue appears to depend on external credentials or service accounts.",
        "browser only": "Issue appears to require browser-only validation beyond the current toolset.",
    }

    def __init__(self, regime_policy: Optional[TaskRegimePolicy] = None):
        self.regime_policy = regime_policy

    def extract_features(
        self,
        issue_text: str,
        repo_context: RepoContext,
        *,
        baseline_result: Optional[Any] = None,
    ) -> dict[str, Any]:
        lowered = issue_text.lower()
        keywords = repo_context.extract_issue_keywords(issue_text, max_keywords=10)
        relevant_files = repo_context.get_relevant_files(keywords, max_files=8)
        estimated_files_to_edit = self._estimate_edit_scope(issue_text, relevant_files)
        failing_tests = _baseline_test_ids(baseline_result, "failing_tests")
        passing_tests = _baseline_test_ids(baseline_result, "passing_tests")
        failing_test_count = _baseline_test_count(
            baseline_result,
            failing_tests,
            "failing_tests",
        )
        passing_test_count = _baseline_test_count(
            baseline_result,
            passing_tests,
            "passing_tests",
        )
        regime_profile = (
            self.regime_policy.infer(
                issue_description=issue_text,
                failing_test_ids=failing_tests,
                passing_test_ids=passing_tests,
                relevant_files=relevant_files,
                terminal_source_files=[],
                source_focus_files=[],
                incomplete_source_files=[],
                incomplete_test_files=[],
                exception_summaries=[],
                interface_symbols=[],
                preserve_collected_test_coverage=False,
                relevant_file_languages=_file_language_hints(
                    repo_context,
                    list(relevant_files),
                ),
                repo_languages=_repo_language_hints(repo_context),
                test_command="",
            )
            if isinstance(self.regime_policy, TaskRegimePolicy)
            else TaskRegimeProfile()
        )
        contract_gap = regime_profile.probability("contract_gap")
        interface_risk = regime_profile.probability("high_interface_risk")
        if isinstance(self.regime_policy, TaskRegimePolicy):
            completion_signal = contract_gap >= self.regime_policy.threshold("contract_gap")
            interface_signal = interface_risk >= self.regime_policy.threshold("high_interface_risk")
        else:
            completion_signal = contract_gap >= 0.5 or _looks_like_completion_task(issue_text)
            interface_signal = interface_risk >= 0.45 or _mentions_public_api_contract(issue_text)
        return {
            "issue_length": len(issue_text),
            "has_reproduction_steps": "steps to reproduce" in lowered or "reproduce" in lowered,
            "has_stack_trace": "traceback" in lowered
            or "error:" in lowered
            or "exception" in lowered,
            "repo_size_files": len(repo_context.files),
            "estimated_files_to_edit": estimated_files_to_edit,
            "candidate_files": list(relevant_files),
            "mentions_manual_dependency": any(
                pattern in lowered for pattern in self._UNSOLVABLE_PATTERNS
            ),
            "has_tests": any(self._looks_like_test_path(file.path) for file in repo_context.files),
            "failing_test_count": failing_test_count,
            "passing_test_count": passing_test_count,
            "failing_test_ids": failing_tests[:8],
            "passing_test_ids": passing_tests[:8],
            "task_regime_probabilities": dict(regime_profile.state_probabilities),
            "is_completion_task": completion_signal,
            "mentions_public_api": interface_signal,
        }

    def estimate_difficulty(self, features: dict[str, Any]) -> float:
        score = 0.0
        score += min(features.get("issue_length", 0) / 1200.0, 1.0) * 0.15
        score += 0.2 if not features.get("has_stack_trace") else 0.05
        score += min(features.get("estimated_files_to_edit", 1) / 6.0, 1.0) * 0.35
        score += min(features.get("repo_size_files", 0) / 1500.0, 1.0) * 0.15
        score += 0.1 if not features.get("has_tests") else 0.0
        score += 0.05 if features.get("has_reproduction_steps") else 0.15
        score += min(features.get("failing_test_count", 0) / 20.0, 1.0) * 0.18
        score += 0.1 if features.get("is_completion_task") else 0.0
        score += 0.05 if features.get("mentions_public_api") else 0.0
        if (
            features.get("passing_test_count", 0) > 0
            and features.get("failing_test_count", 0) <= 2
            and features.get("has_stack_trace")
        ):
            score -= 0.05
        return max(0.0, min(score, 1.0))

    def allocate_rollouts(
        self,
        features: dict[str, Any],
        *,
        min_rollouts: int,
        max_rollouts: int,
        buckets: Optional[list[int]] = None,
    ) -> int:
        if buckets is None:
            bucket_candidates = sorted(set(self.DEFAULT_BUCKETS))
        else:
            bucket_candidates = sorted({int(bucket) for bucket in buckets if int(bucket) > 0})
        eligible = [
            bucket for bucket in bucket_candidates if min_rollouts <= bucket <= max_rollouts
        ]
        if not eligible:
            eligible = sorted({min_rollouts, max_rollouts}) if bucket_candidates else []

        difficulty_score = self._difficulty_score(features)

        if not eligible:
            return _interpolate_rollout_count(
                min_rollouts,
                max_rollouts,
                min(1.0, difficulty_score / 7.0),
            )

        if len(eligible) == 1:
            return eligible[0]
        if difficulty_score <= 2:
            return eligible[0]
        if difficulty_score <= 4:
            return eligible[min(1, len(eligible) - 1)]
        if difficulty_score <= 6:
            return eligible[min(2, len(eligible) - 1)]
        return eligible[-1]

    def predict_unsolvable(self, issue_text: str, repo_context: RepoContext) -> Optional[str]:
        lowered = issue_text.lower()
        if not repo_context.files:
            return "Repository appears empty; no local code is available to modify."
        for pattern, reason in self._UNSOLVABLE_PATTERNS.items():
            if pattern in lowered:
                return reason
        return None

    def _estimate_edit_scope(self, issue_text: str, relevant_files: list[str]) -> int:
        lowered = issue_text.lower()
        estimate = max(1, min(len(relevant_files), 6))
        if any(
            token in lowered
            for token in ("refactor", "across", "multiple files", "callers", "all usages")
        ):
            estimate = max(estimate, 4)
        if any(token in lowered for token in ("rename", "migrate", "sweep")):
            estimate = max(estimate, 5)
        return estimate

    def _looks_like_test_path(self, path: str) -> bool:
        normalized = str(path or "").strip().replace("\\", "/")
        name = Path(normalized).name.lower()
        parts = {part.lower() for part in Path(normalized).parts}
        return (
            "test" in parts
            or "tests" in parts
            or "spec" in parts
            or "__tests__" in parts
            or "testdata" in parts
            or name == "conftest.py"
            or name.startswith("test_")
            or name.endswith("_test.py")
            or name.endswith("_test.go")
            or name.endswith("_test.rs")
            or name.endswith("_spec.rb")
            or ".test." in name
            or ".spec." in name
            or re.search(r"(?:^|[._-])test[s]?\.(?:java|kt|kts|scala|cs)$", name) is not None
            or re.search(r"(?:^|[._-])spec\.(?:java|kt|kts|scala|cs)$", name) is not None
        )

    def _difficulty_score(self, features: dict[str, Any]) -> int:
        score = 0
        if not features.get("has_stack_trace"):
            score += 2
        if features.get("estimated_files_to_edit", 1) > 3:
            score += 3
        elif features.get("estimated_files_to_edit", 1) > 1:
            score += 1
        if features.get("repo_size_files", 0) > 1000:
            score += 1
        if not features.get("has_tests"):
            score += 1
        failing_test_count = int(features.get("failing_test_count", 0) or 0)
        if failing_test_count >= 20:
            score += 2
        elif failing_test_count >= 5:
            score += 1
        if features.get("is_completion_task"):
            score += 2
        if features.get("mentions_public_api"):
            score += 1
        if features.get("has_reproduction_steps"):
            score = max(score - 1, 0)
        if (
            features.get("passing_test_count", 0) > 0
            and failing_test_count <= 2
            and features.get("has_stack_trace")
            and features.get("estimated_files_to_edit", 1) <= 2
        ):
            score = max(score - 1, 0)
        return score

    def recommend_followup_rollouts(
        self,
        rollout_results: list[Any],
        *,
        current_rollouts: int,
        min_rollouts: int,
        max_rollouts: int,
        best_candidate: Optional[Any] = None,
    ) -> int:
        """Choose an additional rollout budget after near-miss candidates."""
        remaining_budget = max(0, max_rollouts - current_rollouts)
        if remaining_budget <= 0:
            return 0

        best_score = max([self._rollout_reward(result) for result in rollout_results] or [0.0])
        if best_candidate is not None:
            best_score = max(best_score, self._rollout_reward(best_candidate))

        if best_score >= 0.9:
            target = max(1, min(min_rollouts, 2))
        elif best_score >= 0.7:
            target = max(min_rollouts, min(max(2, current_rollouts // 2), 4))
        else:
            target = max(min_rollouts, min(current_rollouts, 8))

        return max(1, min(remaining_budget, target))

    def allocate_followup_briefs(
        self,
        rollout_briefs: list["RolloutBrief"],
        rollout_results: list[Any],
        *,
        rollout_count: int,
    ) -> list["RolloutBrief"]:
        """Allocate follow-up rollout briefs with a simple within-task UCB policy."""
        if rollout_count <= 0 or not rollout_briefs:
            return []

        prototypes: dict[str, RolloutBrief] = {}
        for brief in rollout_briefs:
            prototypes.setdefault(
                _rollout_brief_allocation_key(brief),
                RolloutBrief.from_dict(brief.to_dict()),
            )

        stats = {allocation_key: {"count": 0, "reward": 0.0} for allocation_key in prototypes}
        total_observations = 0
        for result in rollout_results:
            allocation_key = _rollout_result_allocation_key(result)
            if allocation_key not in stats:
                continue
            stats[allocation_key]["count"] += 1
            stats[allocation_key]["reward"] += self._rollout_reward(result)
            total_observations += 1

        shadow_counts = {
            allocation_key: values["count"] for allocation_key, values in stats.items()
        }
        selections: list[RolloutBrief] = []
        for allocation_index in range(rollout_count):
            total_trials = max(1, total_observations + allocation_index + 1)
            best_key = max(
                prototypes,
                key=lambda allocation_key: self._ucb_score(
                    stats[allocation_key]["reward"],
                    stats[allocation_key]["count"],
                    shadow_counts[allocation_key],
                    total_trials,
                ),
            )
            shadow_counts[best_key] += 1
            cloned = RolloutBrief.from_dict(prototypes[best_key].to_dict())
            policy = dict(cloned.search_policy or {})
            policy["allocator_arm"] = best_key
            action = ControllerAction.from_dict(cloned.resolved_controller_action().to_dict())
            action.allocator_arm = best_key
            cloned.set_controller_action(action, merge_policy=policy)
            selections.append(cloned)
        return selections

    def _ucb_score(
        self,
        reward_sum: float,
        observed_count: int,
        planned_count: int,
        total_trials: int,
    ) -> float:
        if observed_count <= 0:
            return 1_000_000.0 / max(planned_count + 1, 1)
        exploitation = reward_sum / observed_count
        exploration = math.sqrt((2.0 * math.log(max(total_trials, 2))) / max(planned_count, 1))
        return exploitation + exploration

    def _rollout_reward(self, rollout_result: Any) -> float:
        critic_score: Optional[float] = None
        selection_diagnostics = getattr(rollout_result, "selection_diagnostics", None)
        if isinstance(selection_diagnostics, dict):
            critic = selection_diagnostics.get("critic")
            if isinstance(critic, dict):
                raw_score = critic.get("score")
                if isinstance(raw_score, (int, float)):
                    critic_score = max(0.0, min(float(raw_score), 1.0))

        if rollout_has_authoritative_acceptance(rollout_result):
            return 1.0

        quick_verification_score: Optional[float] = None
        quick_verification = getattr(rollout_result, "quick_verification", None)
        if isinstance(quick_verification, dict):
            quick_verification_score = quick_verification_signal_score(quick_verification)
            if quick_verification_score is not None:
                if quick_verification.get("scope") in {"failing_tests", "focus_test_files"}:
                    quick_verification_score = min(1.0, 0.1 + (0.9 * quick_verification_score))

        verification = getattr(rollout_result, "verification", None)
        if isinstance(verification, dict):
            if verification.get("accepted"):
                return 1.0
            score = verification.get("overall_score")
            if isinstance(score, (int, float)):
                verification_score = max(0.0, min(float(score), 1.0))
                if quick_verification_score is not None:
                    verification_score = max(
                        verification_score,
                        min(0.99, (0.7 * verification_score) + (0.3 * quick_verification_score)),
                    )
                if critic_score is None:
                    return verification_score
                return max(
                    verification_score,
                    min(0.98, 0.65 * verification_score + 0.35 * critic_score),
                )
        elif verification is not None:
            accepted = getattr(verification, "accepted", False)
            if accepted:
                return 1.0
            score = getattr(verification, "overall_score", None)
            if isinstance(score, (int, float)):
                verification_score = max(0.0, min(float(score), 1.0))
                if quick_verification_score is not None:
                    verification_score = max(
                        verification_score,
                        min(0.99, (0.7 * verification_score) + (0.3 * quick_verification_score)),
                    )
                if critic_score is None:
                    return verification_score
                return max(
                    verification_score,
                    min(0.98, 0.65 * verification_score + 0.35 * critic_score),
                )
        progress_score = getattr(rollout_result, "progress_score", None)
        if isinstance(progress_score, (int, float)):
            bounded_progress = max(0.0, min(float(progress_score), 0.98))
            if quick_verification_score is not None:
                bounded_progress = max(
                    bounded_progress,
                    min(0.98, (0.65 * bounded_progress) + (0.35 * quick_verification_score)),
                )
            if critic_score is None:
                return bounded_progress
            return max(
                bounded_progress,
                min(0.98, 0.75 * bounded_progress + 0.25 * critic_score),
            )
        if quick_verification_score is not None:
            if critic_score is None:
                return quick_verification_score
            return max(
                quick_verification_score,
                min(0.98, 0.75 * quick_verification_score + 0.25 * critic_score),
            )
        if critic_score is not None and getattr(rollout_result, "success", False):
            return max(0.35, 0.2 + (0.6 * critic_score))
        if getattr(rollout_result, "success", False):
            return 0.35
        return 0.0


class IssuePlanner:
    """Build a shared issue plan, with LLM refinement when available."""

    def __init__(self, config: ApexConfig):
        self.config = config
        self.regime_policy = TaskRegimePolicy(
            config.planning.regime_policy,
            model_library=getattr(config, "controller_models", None),
        )
        self.allocator = BanditRolloutAllocator(regime_policy=self.regime_policy)

    def plan_issue(
        self,
        issue_description: str,
        repo_context: RepoContext,
        rollout_count: Optional[int] = None,
        difficulty: Optional[float] = None,
        baseline_result: Optional[Any] = None,
    ) -> IssuePlan:
        component_ablation = component_ablation_assignment_for_task(
            config=self.config,
            issue_description=issue_description,
            repo_label=str(getattr(repo_context, "repo_path", "") or ""),
        )
        effective_rollout_count, effective_difficulty = self._resolve_plan_rollout_inputs(
            issue_description,
            repo_context,
            rollout_count=rollout_count,
            difficulty=difficulty,
        )
        initial_task_regime = self._infer_task_regime(
            issue_description=issue_description,
            repo_context=repo_context,
            relevant_files=[],
            baseline_result=baseline_result,
            test_context=TestContext(),
            evaluation_constraints=EvaluationConstraints(),
        )
        heuristic = self._build_heuristic_plan(
            issue_description,
            repo_context,
            rollout_count=effective_rollout_count,
            difficulty=effective_difficulty,
            baseline_result=baseline_result,
            task_regime=initial_task_regime,
        )
        heuristic.task_regime = initial_task_regime
        heuristic.planner_metadata = dict(heuristic.planner_metadata or {})
        heuristic.planner_metadata.update(self._task_regime_metadata(initial_task_regime))
        heuristic.planner_metadata["component_ablation"] = dict(component_ablation)

        def finalize_plan(plan: IssuePlan) -> IssuePlan:
            self._apply_collection_error_compatibility_metadata(
                plan,
                repo_context=repo_context,
                baseline_result=baseline_result,
                allow_bypass_reason=(str(plan.planner_source or "").strip().lower() == "heuristic"),
            )
            plan.planner_metadata = dict(plan.planner_metadata or {})
            plan.planner_metadata.update(self._task_regime_metadata(plan.task_regime))
            plan.planner_metadata.setdefault("component_ablation", dict(component_ablation))
            return plan

        if not self.config.planning.enable_manager_planner:
            return finalize_plan(heuristic)
        planner_llm = self.config.get_planner_llm()
        preplanner_llm = self.config.get_preplanner_llm()
        planner_unavailable = self._llm_unavailable(planner_llm)
        preplanner_unavailable = self._llm_unavailable(preplanner_llm)
        if planner_unavailable:
            if self.config.planning.allow_heuristic_fallback:
                return finalize_plan(heuristic)
            raise RuntimeError("Planner LLM is unavailable and heuristic fallback is disabled.")

        coarse_started = time.perf_counter()
        coarse_plan: Optional[IssuePlan] = None
        coarse_error: Optional[str] = None
        preplanner_skip_reason = self._skip_external_preplanner_reason(
            heuristic,
            requested_rollouts=effective_rollout_count,
            preplanner_llm=preplanner_llm,
            planner_llm=planner_llm,
        )
        use_heuristic_seed = bool(
            self.config.planning.enable_coarse_to_fine_planning
            and (preplanner_unavailable or preplanner_skip_reason is not None)
        )
        if (
            self.config.planning.enable_coarse_to_fine_planning
            and not preplanner_unavailable
            and preplanner_skip_reason is None
        ):
            try:
                coarse_plan = self._llm_refine_plan(
                    issue_description,
                    repo_context,
                    heuristic,
                    rollout_count=effective_rollout_count,
                    difficulty=effective_difficulty,
                    hard_timeout_seconds=self._phase_hard_timeout_seconds(
                        preplanner_llm,
                        self.config.planning.preplanner_timeout_seconds,
                    ),
                    llm_config_override=preplanner_llm,
                    baseline_result=baseline_result,
                    planning_mode="coarse",
                )
                coarse_plan.planner_metadata = dict(coarse_plan.planner_metadata)
                coarse_plan.planner_metadata.update(
                    {
                        "planner_pipeline": "coarse_first",
                        "time_to_first_plan_seconds": round(
                            time.perf_counter() - coarse_started,
                            3,
                        ),
                        "coarse_planner_model": preplanner_llm.model,
                        "coarse_planner_backend": preplanner_llm.backend.value,
                        "coarse_planner_tokens": coarse_plan.planner_tokens,
                        "coarse_planner_timeout_seconds": self._phase_hard_timeout_seconds(
                            preplanner_llm,
                            self.config.planning.preplanner_timeout_seconds,
                        ),
                        "plan_portfolio_enabled": bool(self.config.planning.enable_plan_portfolio),
                    }
                )
            except Exception as exc:
                coarse_error = str(exc) or exc.__class__.__name__
                logger.warning("Coarse planner stage failed; continuing to refiner: %s", exc)

        try:
            if coarse_plan is not None and not self._should_refine_coarse_plan(
                issue_description,
                heuristic=heuristic,
                coarse_plan=coarse_plan,
                requested_rollouts=effective_rollout_count,
                preplanner_llm=preplanner_llm,
                planner_llm=planner_llm,
            ):
                coarse_plan.planner_metadata = dict(coarse_plan.planner_metadata)
                coarse_plan.planner_metadata.setdefault(
                    "planner_total_duration_seconds",
                    round(time.perf_counter() - coarse_started, 3),
                )
                if coarse_error:
                    coarse_plan.planner_metadata.setdefault(
                        "coarse_planner_error",
                        coarse_error,
                    )
                if preplanner_skip_reason:
                    coarse_plan.planner_metadata.setdefault(
                        "preplanner_skip_reason",
                        preplanner_skip_reason,
                    )
                return finalize_plan(coarse_plan)

            planner_timeout_seconds = self._planner_hard_timeout_seconds(
                issue_description,
                repo_context,
                heuristic,
                planner_llm,
            )
            refined_started = time.perf_counter()
            refined_plan = self._llm_refine_plan(
                issue_description,
                repo_context,
                coarse_plan or heuristic,
                rollout_count=effective_rollout_count,
                difficulty=effective_difficulty,
                hard_timeout_seconds=(
                    self._phase_hard_timeout_seconds(
                        planner_llm,
                        self.config.planning.refinement_timeout_seconds,
                    )
                    or planner_timeout_seconds
                ),
                llm_config_override=planner_llm,
                seed_plan=coarse_plan or (heuristic if use_heuristic_seed else None),
                baseline_result=baseline_result,
                planning_mode=(
                    "refine"
                    if coarse_plan is not None
                    else "seed_refine"
                    if use_heuristic_seed
                    else "direct"
                ),
            )
            refined_plan.planner_metadata = dict(refined_plan.planner_metadata)
            if coarse_plan is not None:
                refined_plan.planner_tokens += coarse_plan.planner_tokens
            refined_plan.planner_metadata.update(
                {
                    "planner_pipeline": (
                        "coarse_to_fine"
                        if coarse_plan is not None
                        else "heuristic_to_refine"
                        if use_heuristic_seed
                        else "direct_refine"
                    ),
                    "time_to_first_plan_seconds": round(
                        (coarse_plan.planner_metadata or {}).get("time_to_first_plan_seconds")
                        if coarse_plan is not None
                        else 0.0
                        if use_heuristic_seed
                        else (time.perf_counter() - refined_started),
                        3,
                    ),
                    "coarse_planner_model": (
                        preplanner_llm.model if coarse_plan is not None else None
                    ),
                    "coarse_planner_backend": (
                        preplanner_llm.backend.value if coarse_plan is not None else None
                    ),
                    "coarse_planner_tokens": (
                        coarse_plan.planner_tokens if coarse_plan is not None else 0
                    ),
                    "preplanner_skip_reason": preplanner_skip_reason,
                    "refinement_attempted": True,
                    "refinement_model": planner_llm.model,
                    "refinement_backend": planner_llm.backend.value,
                    "refinement_duration_seconds": round(
                        time.perf_counter() - refined_started,
                        3,
                    ),
                    "planner_total_duration_seconds": round(
                        time.perf_counter() - coarse_started,
                        3,
                    ),
                    "plan_portfolio_enabled": bool(self.config.planning.enable_plan_portfolio),
                }
            )
            if coarse_error:
                refined_plan.planner_metadata.setdefault(
                    "coarse_planner_error",
                    coarse_error,
                )
            if preplanner_skip_reason:
                refined_plan.planner_metadata.setdefault(
                    "preplanner_skip_reason",
                    preplanner_skip_reason,
                )
            return finalize_plan(refined_plan)
        except Exception as exc:
            if coarse_plan is not None:
                coarse_plan.planner_metadata = dict(coarse_plan.planner_metadata)
                coarse_plan.planner_metadata.setdefault(
                    "planner_refinement_failed_reason",
                    str(exc) or exc.__class__.__name__,
                )
                coarse_plan.planner_metadata.setdefault(
                    "planner_total_duration_seconds",
                    round(time.perf_counter() - coarse_started, 3),
                )
                if coarse_error:
                    coarse_plan.planner_metadata.setdefault(
                        "coarse_planner_error",
                        coarse_error,
                    )
                logger.warning(
                    "Planner refinement failed, falling back to coarse plan: %s",
                    exc,
                )
                return finalize_plan(coarse_plan)
            if self.config.planning.allow_heuristic_fallback:
                heuristic.planner_metadata = dict(heuristic.planner_metadata)
                heuristic.planner_metadata.setdefault(
                    "planner_fallback_reason",
                    str(exc) or exc.__class__.__name__,
                )
                planner_timeout_seconds = self._planner_hard_timeout_seconds(
                    issue_description,
                    repo_context,
                    heuristic,
                    planner_llm,
                )
                if planner_timeout_seconds is not None:
                    heuristic.planner_metadata.setdefault(
                        "planner_timeout_seconds",
                        planner_timeout_seconds,
                    )
                logger.warning("Planner LLM unavailable, falling back to heuristics: %s", exc)
                return finalize_plan(heuristic)
            raise RuntimeError("Planner LLM failed and heuristic fallback is disabled.") from exc

    def _llm_unavailable(self, llm_config: LLMConfig) -> bool:
        return not llm_backend_is_available(llm_config)

    def _task_regime_probability(
        self,
        task_regime: Optional[TaskRegimeProfile],
        state: str,
    ) -> float:
        if not isinstance(task_regime, TaskRegimeProfile):
            return 0.0
        return task_regime.probability(state)

    def _task_regime_metadata(
        self,
        task_regime: Optional[TaskRegimeProfile],
    ) -> dict[str, Any]:
        profile = task_regime if isinstance(task_regime, TaskRegimeProfile) else TaskRegimeProfile()
        return {
            "task_regime_summary": str(profile.summary or "").strip(),
            "task_regime_probabilities": {
                state: round(float(probability), 4)
                for state, probability in dict(profile.state_probabilities or {}).items()
            },
            "task_regime_active_states": list(profile.active_states()),
            "task_regime_evidence_count": len(profile.evidence or []),
        }

    def _collection_error_focus_reason(
        self,
        *,
        repo_context: RepoContext,
        baseline_result: Optional[Any],
        task_regime: Optional[TaskRegimeProfile] = None,
        test_context: Optional[TestContext] = None,
    ) -> Optional[str]:
        output = _baseline_output(baseline_result)
        traceback_signal = self._extract_traceback_signal(repo_context, output)
        context = test_context if isinstance(test_context, TestContext) else TestContext()

        failing_test_ids = list(context.failing_test_ids or []) or _baseline_test_ids(
            baseline_result,
            "failing_tests",
        )
        passing_test_ids = list(context.passing_test_ids or []) or _baseline_test_ids(
            baseline_result,
            "passing_tests",
        )
        failing_test_count = max(
            int(context.failing_test_count or 0),
            _baseline_test_count(baseline_result, failing_test_ids, "failing_tests"),
        )
        passing_test_count = max(
            int(context.passing_test_count or 0),
            _baseline_test_count(baseline_result, passing_test_ids, "passing_tests"),
        )
        if (
            not failing_test_ids
            and failing_test_count > 0
            and passing_test_count == 0
            and traceback_signal.test_files
        ):
            failing_test_ids = list(traceback_signal.test_files[:4])

        collection_like = bool(_looks_like_collection_failure_output(output)) or any(
            _looks_like_collection_error_test_id(test_id) for test_id in failing_test_ids
        )
        if (
            not collection_like
            and self._task_regime_probability(task_regime, "importability_blocker")
            >= self.regime_policy.threshold("importability_blocker")
            and failing_test_count > 0
        ):
            collection_like = True

        localized = bool(
            traceback_signal.terminal_source_files
            or traceback_signal.source_files
            or traceback_signal.exception_summaries
            or context.terminal_source_files
            or context.source_focus_files
            or context.exception_summaries
        )
        if collection_like and localized and failing_test_count > 0 and passing_test_count == 0:
            return _BASELINE_COLLECTION_ERROR_BYPASS_REASON
        return None

    def _apply_collection_error_compatibility_metadata(
        self,
        issue_plan: IssuePlan,
        *,
        repo_context: RepoContext,
        baseline_result: Optional[Any],
        allow_bypass_reason: bool,
    ) -> None:
        metadata = dict(issue_plan.planner_metadata or {})
        metadata.update(self._build_baseline_signal_metadata(repo_context, baseline_result))
        reason = (
            self._collection_error_focus_reason(
                repo_context=repo_context,
                baseline_result=baseline_result,
                task_regime=issue_plan.task_regime,
                test_context=issue_plan.test_context,
            )
            or str(metadata.get("collection_error_focus_reason") or "").strip()
        )
        if reason:
            metadata["collection_error_focus_reason"] = reason
        else:
            metadata.pop("collection_error_focus_reason", None)
        if (
            allow_bypass_reason
            and self.config.planning.enable_collection_error_planner_bypass
            and reason
        ):
            metadata["planner_bypass_reason"] = reason
        else:
            metadata.pop("planner_bypass_reason", None)
        issue_plan.planner_metadata = metadata

    def _existing_test_generation_design_payload(
        self,
        issue_plan: IssuePlan,
    ) -> dict[str, Any]:
        planner_metadata = (
            dict(issue_plan.planner_metadata or {})
            if isinstance(issue_plan.planner_metadata, dict)
            else {}
        )
        authored = planner_metadata.get("planner_test_generation_design")
        if isinstance(authored, dict):
            return dict(authored)
        design_payload = planner_metadata.get("test_generation_design")
        if isinstance(design_payload, dict):
            return dict(design_payload)
        if not any(
            planner_metadata.get(key)
            for key in (
                "test_generation_contract",
                "test_generation_milestones",
                "test_generation_objectives",
            )
        ):
            return {}
        return {
            "task_contract": dict(planner_metadata.get("test_generation_contract") or {}),
            "milestones": [
                dict(item)
                for item in list(planner_metadata.get("test_generation_milestones") or [])
                if isinstance(item, dict)
            ],
            "test_objectives": [
                dict(item)
                for item in list(planner_metadata.get("test_generation_objectives") or [])
                if isinstance(item, dict)
            ],
        }

    def _build_test_generation_design_prompt(
        self,
        issue_plan: IssuePlan,
        *,
        issue_description: str,
        repo_context: RepoContext,
        interface_targets: list[str],
        behavioral_obligations: list[str],
        required_axes: list[str],
        existing_ledger: dict[str, Any],
        boundary_context: Optional[dict[str, Any]] = None,
    ) -> str:
        design_seed = self._existing_test_generation_design_payload(issue_plan)
        seed_milestones = [
            dict(item)
            for item in list(
                existing_ledger.get("milestones") or design_seed.get("milestones") or []
            )
            if isinstance(item, dict)
        ]
        seed_objectives = [
            dict(item)
            for item in list(
                existing_ledger.get("test_objectives") or design_seed.get("test_objectives") or []
            )
            if isinstance(item, dict)
        ]
        milestone_lines = [
            (
                f"- {item.get('milestone_id')}: {item.get('title') or item.get('summary') or item.get('milestone_id')} "
                f"| objectives={', '.join(list(item.get('objective_ids') or [])) or 'none'} "
                f"| validation={item.get('validation_level') or 'strict'}"
            ).strip()
            for item in seed_milestones
            if str(item.get("milestone_id") or "").strip()
        ]
        objective_lines = [
            (
                f"- {item.get('objective_id')}: {item.get('objective') or item.get('summary') or item.get('objective_id')} "
                f"| milestone={item.get('milestone_id')} "
                f"| axes={', '.join(list(item.get('contract_axes') or [])) or 'none'}"
            ).strip()
            for item in seed_objectives
            if str(item.get("objective_id") or "").strip()
        ]
        relevant_files = _dedupe_preserve(
            list(issue_plan.relevant_files or [])
            + list(issue_plan.risk_files or [])
            + list(issue_plan.test_context.focus_test_files or [])
        )
        repo_map = repo_context.build_context_pack(
            relevant_files[: max(6, min(self.config.planning.max_repo_map_files, 14))],
            max_symbols_per_file=6,
            seed_symbols=issue_plan.keywords
            + list(issue_plan.test_context.terminal_reference_symbols or []),
        )
        lines = [
            "# Issue",
            self._truncate_block(
                _truncate_words(issue_description, max_words=220, max_chars=1600),
                max_lines=18,
            ),
            "",
            "# Issue Summary",
            _truncate_words(str(issue_plan.summary or "").strip(), max_words=40, max_chars=320),
            "",
            "# Success Criteria",
            "\n".join(
                f"- {value}"
                for value in _compact_string_list(
                    list(issue_plan.success_criteria or [])
                    + list(issue_plan.test_context.expectations or []),
                    max_items=8,
                    max_words=18,
                    max_chars=180,
                )
            )
            or "- derive from the issue and repository evidence",
            "",
            "# Relevant Files",
            "\n".join(f"- {path}" for path in relevant_files[:12]) or "- infer from the repo",
            "",
            "# Focus Repo Map",
            self._truncate_block(repo_map, max_lines=60),
            "",
            "# Interface Targets",
            "\n".join(f"- {value}" for value in interface_targets[:8])
            or "- infer the public surfaces",
            "",
            "# Behavioral Obligations",
            "\n".join(f"- {value}" for value in behavioral_obligations[:12])
            or "- derive the observable obligations from the issue and nearby tests",
            "",
            "# Required Contract Axes",
            "\n".join(f"- {value}" for value in required_axes[:8]) or "- positive_path",
        ]
        if issue_plan.test_context.summary:
            lines.extend(
                [
                    "",
                    "# Test Context",
                    self._truncate_block(str(issue_plan.test_context.summary or ""), max_lines=12),
                ]
            )
        if issue_plan.test_context.focus_test_files:
            lines.extend(
                [
                    "",
                    "# Focus Test Files",
                    "\n".join(
                        f"- {path}"
                        for path in list(issue_plan.test_context.focus_test_files or [])[:8]
                    ),
                ]
            )
        if milestone_lines:
            lines.extend(
                [
                    "",
                    "# Existing Milestones",
                    "\n".join(milestone_lines),
                ]
            )
        if objective_lines:
            lines.extend(
                [
                    "",
                    "# Existing Test Objectives",
                    "\n".join(objective_lines),
                ]
            )
        if boundary_context:
            lines.extend(
                [
                    "",
                    "# Boundary Replan Context",
                    f"Reason: {str(boundary_context.get('replan_reason') or 'milestone_boundary').strip()}",
                    f"Active milestone after checkpoint: {str(boundary_context.get('current_milestone_id') or '').strip() or 'none'}",
                    "Completed STRICT-ready milestones:",
                    "\n".join(
                        f"- {value}"
                        for value in list(
                            boundary_context.get("completed_strict_ready_milestone_ids") or []
                        )[:12]
                    )
                    or "- none",
                    "Future milestones still open to revision:",
                    "\n".join(
                        f"- {value}"
                        for value in list(boundary_context.get("future_milestone_ids") or [])[:12]
                    )
                    or "- none",
                    "Current cumulative regression artifacts:",
                    "\n".join(
                        f"- {value}"
                        for value in list(boundary_context.get("regression_suite_artifacts") or [])[
                            :12
                        ]
                    )
                    or "- none",
                ]
            )
            boundary_requested_files = [
                str(value).strip()
                for value in list(boundary_context.get("boundary_requested_files") or [])[:12]
                if str(value).strip()
            ]
            if boundary_requested_files:
                lines.extend(
                    [
                        "Executor-requested adjacent files:",
                        "\n".join(f"- {value}" for value in boundary_requested_files),
                    ]
                )
            boundary_interface_symbols = [
                str(value).strip()
                for value in list(boundary_context.get("boundary_interface_symbols") or [])[:12]
                if str(value).strip()
            ]
            if boundary_interface_symbols:
                lines.extend(
                    [
                        "Boundary interface symbols:",
                        "\n".join(f"- {value}" for value in boundary_interface_symbols),
                    ]
                )
            missing_contract_axes = [
                str(value).strip()
                for value in list(boundary_context.get("missing_contract_axes") or [])[:12]
                if str(value).strip()
            ]
            if missing_contract_axes:
                lines.extend(
                    [
                        "Contract axes still missing after execution:",
                        "\n".join(f"- {value}" for value in missing_contract_axes),
                    ]
                )
            missing_issue_surface_targets = [
                str(value).strip()
                for value in list(boundary_context.get("missing_issue_surface_targets") or [])[:12]
                if str(value).strip()
            ]
            if missing_issue_surface_targets:
                lines.extend(
                    [
                        "Issue-declared targets still missing from the generated coverage:",
                        "\n".join(f"- {value}" for value in missing_issue_surface_targets),
                    ]
                )
            if bool(boundary_context.get("coverage_deferral_detected")):
                lines.append(
                    "Execution detected that the previous portfolio deferred required behavioral coverage instead of encoding it directly in generated tests."
                )
            if bool(boundary_context.get("preferred_test_anchor_gap_detected")):
                lines.append(
                    "Execution detected that the previous portfolio drifted away from the preferred authoritative test anchors."
                )
            if bool(boundary_context.get("observable_behavior_gap_detected")):
                lines.append(
                    "Execution detected that observable public-surface behavior is still under-specified by the current milestone artifacts."
                )
            if bool(boundary_context.get("schema_constraint_drift_detected")):
                lines.append(
                    "Execution detected schema or assertion drift away from the authoritative issue surface."
                )
            if bool(boundary_context.get("field_path_negative_shape_detected")):
                lines.append(
                    "Execution detected missing negative malformed field-path coverage that the current milestone still needs to explain."
                )
            active_blocking_objectives = [
                str(value).strip()
                for value in list(
                    boundary_context.get("active_milestone_blocking_objectives") or []
                )[:12]
                if str(value).strip()
            ]
            if active_blocking_objectives:
                lines.extend(
                    [
                        "Current milestone blocking objectives:",
                        "\n".join(f"- {value}" for value in active_blocking_objectives),
                    ]
                )
            design_missing_objective_ids = [
                str(value).strip()
                for value in list(boundary_context.get("design_missing_objective_ids") or [])[:12]
                if str(value).strip()
            ]
            if design_missing_objective_ids:
                lines.extend(
                    [
                        "Missing design objective IDs seen during execution:",
                        "\n".join(f"- {value}" for value in design_missing_objective_ids),
                    ]
                )
            design_missing_milestone_ids = [
                str(value).strip()
                for value in list(boundary_context.get("design_missing_milestone_ids") or [])[:12]
                if str(value).strip()
            ]
            if design_missing_milestone_ids:
                lines.extend(
                    [
                        "Missing design milestone IDs seen during execution:",
                        "\n".join(f"- {value}" for value in design_missing_milestone_ids),
                    ]
                )
            design_artifact_metadata_gaps = [
                dict(value)
                for value in list(boundary_context.get("design_artifact_metadata_gaps") or [])[:12]
                if isinstance(value, dict)
            ]
            if design_artifact_metadata_gaps:
                lines.extend(
                    [
                        "Artifact metadata gaps seen during execution:",
                        "\n".join(
                            "- "
                            + (
                                f"{str(item.get('path') or '').strip()}: "
                                if str(item.get("path") or "").strip()
                                else ""
                            )
                            + ", ".join(
                                str(field).strip()
                                for field in list(item.get("missing_fields") or [])
                                if str(field).strip()
                            )
                            for item in design_artifact_metadata_gaps
                        ),
                    ]
                )
            boundary_followups = [
                str(value).strip()
                for value in list(boundary_context.get("boundary_followups") or [])[:12]
                if str(value).strip()
            ]
            if boundary_followups:
                lines.extend(
                    [
                        "Executor follow-up notes:",
                        "\n".join(f"- {value}" for value in boundary_followups),
                    ]
                )
        lines.extend(
            [
                "",
                "Design a milestone-driven test-generation plan for the executor.",
                "Return a `task_contract` with `problem_statement`, `acceptance_requirements`, and `interface_specification`.",
                "Return ordered `milestones` and `test_objectives` for the test generator, not rollout briefs.",
                "Each objective must map to exactly one milestone and at least one acceptance requirement.",
                "Use milestone boundaries to separate materially different public behaviors, contract axes, or regression checkpoints. Do not collapse everything into a fixed two-bucket heuristic unless the task is genuinely trivial.",
                "Assume the executor will run one milestone at a time, validate CORE/ISO/STRICT at each checkpoint, and can use multiple vendors or agents within a milestone.",
                "Keep `validation_level` at `strict` unless a weaker boundary is clearly justified.",
                "Set every milestone `pipeline_stages` to the five-stage tactical flow: context_hypothesis, pass_then_invert, execution_feedback, mutation_discrimination, dual_version_verification.",
                "Use `contract_axes` such as positive_path, missing_boundary, negative_malformed, and multi_ordering when they fit the obligation.",
                "Prefer stable, reusable IDs. If the existing design already has milestone or objective IDs that still apply, preserve them instead of inventing new ones.",
                (
                    "Treat completed STRICT-ready milestones as a fixed prefix. You may revise only the active or future milestones while keeping the completed prefix stable."
                    if boundary_context
                    else "Produce the strongest initial milestone decomposition you can from the available repository evidence."
                ),
                "Respond with JSON only. No prose, markdown, or code fences.",
            ]
        )
        return "\n".join(lines)

    def _run_test_generation_design_prompt(
        self,
        llm_config: LLMConfig,
        prompt: str,
        working_dir: str,
        *,
        hard_timeout_seconds: Optional[int] = None,
    ) -> tuple[dict[str, Any], int]:
        attempted_fingerprints: set[tuple[str, str, str, str]] = set()
        last_exc: Optional[Exception] = None
        while True:
            resolved_llm_config, routing = resolve_available_llm_config(
                llm_config,
                self.config.llm_configs,
                exclude_fingerprints=attempted_fingerprints,
                purpose="test_generation_design",
            )
            resolved_fingerprint = llm_backend_fingerprint(resolved_llm_config)
            current_reason = llm_backend_unavailable_reason(resolved_llm_config)
            if current_reason:
                if last_exc is not None:
                    raise last_exc
                raise RuntimeError(current_reason)
            if resolved_fingerprint in attempted_fingerprints:
                break
            if routing.get("fallback_applied"):
                logger.info(
                    "Test-generation design planner rerouted from %s/%s to %s/%s (%s)",
                    routing.get("requested_backend") or "unknown",
                    routing.get("requested_model") or "unknown",
                    routing.get("resolved_backend") or "unknown",
                    routing.get("resolved_model") or "unknown",
                    routing.get("fallback_kind") or "fallback",
                )
            attempted_fingerprints.add(resolved_fingerprint)
            try:
                return self._run_test_generation_design_prompt_once(
                    resolved_llm_config,
                    prompt,
                    working_dir,
                    hard_timeout_seconds=hard_timeout_seconds,
                )
            except Exception as exc:
                reason = record_llm_backend_failure(resolved_llm_config, exc)
                invocation_failover_reason = (
                    "" if reason else classify_llm_call_failover_failure(exc)
                )
                if not reason and not invocation_failover_reason:
                    raise
                last_exc = exc
                if reason:
                    logger.warning(
                        "Test-generation design backend %s/%s became unavailable (%s); retrying alternate backend if configured.",
                        routing.get("resolved_backend") or "unknown",
                        routing.get("resolved_model") or "unknown",
                        reason,
                    )
                else:
                    logger.warning(
                        "Test-generation design backend %s/%s hit invocation-local failure (%s); retrying alternate backend if configured without globally marking the backend unavailable.",
                        routing.get("resolved_backend") or "unknown",
                        routing.get("resolved_model") or "unknown",
                        invocation_failover_reason,
                    )
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(
            llm_backend_unavailable_reason(llm_config)
            or "Test-generation design planner is unavailable."
        )

    def _run_test_generation_design_prompt_once(
        self,
        llm_config: LLMConfig,
        prompt: str,
        working_dir: str,
        *,
        hard_timeout_seconds: Optional[int] = None,
    ) -> tuple[dict[str, Any], int]:
        system_prompt = (
            "You are a strategic planner authoring milestone-driven test-generation designs "
            "for a software engineering agent."
        )
        configured_timeout_seconds = self.config.planning.refinement_timeout_seconds
        effective_hard_timeout_seconds = (
            hard_timeout_seconds
            if isinstance(hard_timeout_seconds, int) and hard_timeout_seconds > 0
            else configured_timeout_seconds
        )
        effective_hard_timeout_seconds = (
            effective_hard_timeout_seconds
            if llm_config.is_cli_backend
            and isinstance(effective_hard_timeout_seconds, int)
            and effective_hard_timeout_seconds > 0
            else None
        )

        if llm_config.is_cli_backend:
            use_inline_schema = llm_config.backend == LLMBackend.CODEX_CLI
            schema_text = json.dumps(_TEST_GENERATION_DESIGN_SCHEMA, indent=2, sort_keys=True)
            cli_prompt = (
                prompt
                + "\n\nReturn a single JSON object matching the test-generation design schema."
                + "\nKeep text fields terse. Omit optional keys when they are empty."
            )
            if use_inline_schema:
                cli_prompt += "\nSchema:\n" + schema_text
            result = CLIModelClient(llm_config).run_structured_prompt(
                prompt=cli_prompt,
                working_dir=working_dir,
                schema=None if use_inline_schema else _TEST_GENERATION_DESIGN_SCHEMA,
                system_prompt=system_prompt,
                allow_edits=False,
                hard_timeout_seconds=effective_hard_timeout_seconds,
            )
            if result.success and result.parsed_json:
                validation_error = (
                    _validate_planner_parsed_json(
                        result.parsed_json,
                        schema=_TEST_GENERATION_DESIGN_SCHEMA,
                        required_keys=["task_contract", "milestones", "test_objectives"],
                    )
                    if self.config.planning.enable_planner_output_validation
                    else None
                )
                if not validation_error:
                    return result.parsed_json, extract_total_tokens(result.usage)
                logger.warning(
                    "Test-generation design attempt 1 produced schema-invalid output (%s); "
                    "retrying rather than accepting the degraded design",
                    validation_error,
                )
            if result.error and _looks_like_timeout_or_stall_error(result.error):
                raise RuntimeError(result.error)
            retry_prompt = (
                cli_prompt
                + "\nThe response must be valid JSON with top-level keys: task_contract, milestones, test_objectives."
            )
            retry_result = CLIModelClient(llm_config).run_structured_prompt(
                prompt=retry_prompt,
                working_dir=working_dir,
                schema=None if use_inline_schema else _TEST_GENERATION_DESIGN_SCHEMA,
                system_prompt=system_prompt,
                allow_edits=False,
                hard_timeout_seconds=effective_hard_timeout_seconds,
            )
            if retry_result.success and retry_result.parsed_json:
                retry_validation_error = (
                    _validate_planner_parsed_json(
                        retry_result.parsed_json,
                        schema=_TEST_GENERATION_DESIGN_SCHEMA,
                        required_keys=["task_contract", "milestones", "test_objectives"],
                    )
                    if self.config.planning.enable_planner_output_validation
                    else None
                )
                if not retry_validation_error:
                    return retry_result.parsed_json, extract_total_tokens(retry_result.usage)
                logger.warning(
                    "Test-generation design attempt 2 produced schema-invalid output (%s); "
                    "falling through to heuristic design",
                    retry_validation_error,
                )
                raise RuntimeError(
                    "Test-generation design returned schema-invalid JSON after retry: "
                    + retry_validation_error
                )
            if retry_result.error and _looks_like_timeout_or_stall_error(retry_result.error):
                raise RuntimeError(retry_result.error)
            raise RuntimeError(
                retry_result.error
                or result.error
                or "Test-generation design planner did not return structured JSON."
            )

        llm = LLMClient(llm_config, temperature_override=0.0)
        response = llm.chat(
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=prompt),
            ],
            tools=[_TEST_GENERATION_DESIGN_TOOL],
            temperature=0.0,
        )
        if (
            not response.tool_calls
            or response.tool_calls[0].name != "submit_test_generation_design"
        ):
            raise RuntimeError("Test-generation design planner did not return a structured design.")
        return response.tool_calls[0].arguments, llm.total_tokens_used

    def _author_test_generation_design(
        self,
        issue_plan: IssuePlan,
        *,
        issue_description: str,
        repo_context: RepoContext,
        interface_targets: list[str],
        behavioral_obligations: list[str],
        required_axes: list[str],
        existing_ledger: dict[str, Any],
        force_replan: bool = False,
        boundary_context: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], int, dict[str, Any]]:
        llm_config = self.config.get_planner_llm()
        if self._llm_unavailable(llm_config):
            return (
                {},
                0,
                {
                    "source": "heuristic",
                    "error": llm_backend_unavailable_reason(llm_config) or "planner_unavailable",
                    "replanned": bool(force_replan),
                },
            )
        prompt = self._build_test_generation_design_prompt(
            issue_plan,
            issue_description=issue_description,
            repo_context=repo_context,
            interface_targets=interface_targets,
            behavioral_obligations=behavioral_obligations,
            required_axes=required_axes,
            existing_ledger=existing_ledger,
            boundary_context=boundary_context,
        )
        try:
            payload, tokens_used = self._run_test_generation_design_prompt(
                llm_config,
                prompt,
                working_dir=repo_context.repo_path,
                hard_timeout_seconds=self._planner_hard_timeout_seconds(
                    issue_description,
                    repo_context,
                    issue_plan,
                    llm_config,
                ),
            )
            normalized = normalize_test_generation_design_payload(
                payload,
                issue_description=issue_description,
                issue_summary=str(issue_plan.summary or "").strip(),
                success_criteria=_dedupe_preserve(
                    list(issue_plan.success_criteria) + list(issue_plan.test_context.expectations)
                ),
                behavioral_obligations=behavioral_obligations,
                interface_targets=interface_targets,
                required_axes=required_axes,
            )
            return (
                normalized,
                tokens_used,
                {
                    "source": "planner_llm",
                    "model": str(llm_config.model or ""),
                    "backend": llm_config.backend.value,
                    "replanned": bool(force_replan),
                    "boundary_replan": bool(boundary_context),
                    "error": "",
                },
            )
        except Exception as exc:
            logger.warning(
                "Test-generation design planning failed; falling back to heuristic design: %s", exc
            )
            return (
                {},
                0,
                {
                    "source": "heuristic",
                    "error": str(exc) or exc.__class__.__name__,
                    "replanned": bool(force_replan),
                    "boundary_replan": bool(boundary_context),
                },
            )

    def _attach_test_generation_design_metadata(
        self,
        issue_plan: IssuePlan,
        *,
        issue_description: str,
        repo_context: RepoContext,
        force_replan: bool = False,
        boundary_context: Optional[dict[str, Any]] = None,
    ) -> None:
        existing_task_state_context = (
            dict(issue_plan.task_state_context or {})
            if isinstance(issue_plan.task_state_context, dict)
            else {}
        )
        existing_ledger = (
            dict(existing_task_state_context.get("test_generation_ledger") or {})
            if isinstance(existing_task_state_context.get("test_generation_ledger"), dict)
            else {}
        )
        interface_targets = _dedupe_preserve(extract_issue_contract_targets(issue_description))
        behavioral_obligations = _dedupe_preserve(
            list(issue_plan.test_context.expectations) + list(issue_plan.success_criteria)
        )
        required_axes = infer_required_contract_axes_from_texts(
            [
                issue_description,
                str(issue_plan.summary or ""),
                *behavioral_obligations,
            ]
        )
        existing_authored_design = self._existing_test_generation_design_payload(issue_plan)
        design_plan = dict(existing_authored_design)
        design_metadata = {
            "source": "cached" if existing_authored_design else "heuristic",
            "error": "",
            "replanned": False,
            "boundary_replan": False,
        }
        design_tokens = 0
        if force_replan or not design_plan:
            design_plan, design_tokens, design_metadata = self._author_test_generation_design(
                issue_plan,
                issue_description=issue_description,
                repo_context=repo_context,
                interface_targets=interface_targets,
                behavioral_obligations=behavioral_obligations,
                required_axes=required_axes,
                existing_ledger=existing_ledger,
                force_replan=force_replan,
                boundary_context=boundary_context,
            )
            if not design_plan and existing_authored_design:
                design_plan = dict(existing_authored_design)
                design_metadata = {
                    **design_metadata,
                    "source": "cached_fallback",
                }
            if design_tokens > 0:
                issue_plan.planner_tokens += design_tokens
        design_plan = normalize_test_generation_design_payload(
            design_plan,
            issue_description=issue_description,
            issue_summary=str(issue_plan.summary or "").strip(),
            success_criteria=_dedupe_preserve(
                list(issue_plan.success_criteria) + list(issue_plan.test_context.expectations)
            ),
            behavioral_obligations=behavioral_obligations,
            interface_targets=interface_targets,
            required_axes=required_axes,
        )
        task_contract = dict(design_plan.get("task_contract") or {})
        existing_milestones_by_id = {
            str(item.get("milestone_id") or "").strip(): dict(item)
            for item in list(existing_ledger.get("milestones") or [])
            if isinstance(item, dict) and str(item.get("milestone_id") or "").strip()
        }
        existing_objectives_by_id = {
            str(item.get("objective_id") or "").strip(): dict(item)
            for item in list(existing_ledger.get("test_objectives") or [])
            if isinstance(item, dict) and str(item.get("objective_id") or "").strip()
        }
        milestones = []
        for raw_milestone in list(design_plan.get("milestones") or []):
            if not isinstance(raw_milestone, dict):
                continue
            milestone = dict(raw_milestone)
            milestone_id = str(milestone.get("milestone_id") or "").strip()
            existing = dict(existing_milestones_by_id.get(milestone_id) or {})
            if existing:
                for field_name in (
                    "objective_status",
                    "strict_validation_ready",
                    "core_ready",
                    "iso_ready",
                    "strict_ready",
                    "ready_for_required_level",
                ):
                    if field_name in existing:
                        milestone[field_name] = existing.get(field_name)
            milestones.append(milestone)
        test_objectives = []
        for raw_objective in list(design_plan.get("test_objectives") or []):
            if not isinstance(raw_objective, dict):
                continue
            objective = dict(raw_objective)
            objective_id = str(objective.get("objective_id") or "").strip()
            existing = dict(existing_objectives_by_id.get(objective_id) or {})
            if existing:
                for field_name in (
                    "objective_status",
                    "baseline_preserved",
                    "dual_version_verified",
                    "mutation_discrimination_passed",
                    "pass_then_invert_complete",
                    "artifact_ids",
                ):
                    if field_name in existing:
                        objective[field_name] = existing.get(field_name)
            test_objectives.append(objective)
        ordered_milestone_ids = [
            str(item.get("milestone_id") or "").strip()
            for item in milestones
            if str(item.get("milestone_id") or "").strip()
        ]
        current_milestone_id = str(existing_ledger.get("current_milestone_id") or "").strip()
        if current_milestone_id and current_milestone_id not in ordered_milestone_ids:
            current_milestone_id = ""
        if not current_milestone_id:
            current_milestone_id = str(
                dict(milestones[0]).get("milestone_id") if milestones else ""
            ).strip()

        issue_plan.planner_metadata = dict(issue_plan.planner_metadata or {})
        issue_plan.planner_metadata.update(
            {
                "planner_test_generation_design": {
                    "task_contract": task_contract,
                    "milestones": milestones,
                    "test_objectives": test_objectives,
                },
                "test_generation_design": {
                    "task_contract": task_contract,
                    "milestones": milestones,
                    "test_objectives": test_objectives,
                },
                "test_generation_contract": task_contract,
                "test_generation_milestones": milestones,
                "test_generation_objectives": test_objectives,
                "test_generation_required_axes": list(required_axes),
                "test_generation_design_source": str(
                    design_metadata.get("source")
                    or issue_plan.planner_metadata.get("test_generation_design_source")
                    or "heuristic"
                ),
                "test_generation_design_model": str(
                    design_metadata.get("model")
                    or issue_plan.planner_metadata.get("test_generation_design_model")
                    or ""
                ),
                "test_generation_design_backend": str(
                    design_metadata.get("backend")
                    or issue_plan.planner_metadata.get("test_generation_design_backend")
                    or ""
                ),
                "test_generation_design_tokens": int(
                    issue_plan.planner_metadata.get("test_generation_design_tokens") or 0
                )
                + int(design_tokens or 0),
                "test_generation_design_replanned": bool(design_metadata.get("replanned")),
                "test_generation_design_boundary_replan": bool(
                    design_metadata.get("boundary_replan")
                ),
                "test_generation_design_error": str(design_metadata.get("error") or "").strip(),
            }
        )
        existing_task_state_context["test_generation_ledger"] = {
            "task_contract": task_contract,
            "milestones": milestones,
            "test_objectives": test_objectives,
            "required_contract_axes": _dedupe_preserve(
                list(existing_ledger.get("required_contract_axes") or []) + list(required_axes)
            ),
            "current_milestone_id": current_milestone_id,
            "regression_suite_artifacts": _dedupe_preserve(
                list(existing_ledger.get("regression_suite_artifacts") or [])
            ),
            "strict_ready": bool(existing_ledger.get("strict_ready")),
            "executor_requested_replans": {
                str(key).strip(): int(value or 0)
                for key, value in dict(
                    existing_ledger.get("executor_requested_replans") or {}
                ).items()
                if str(key).strip()
            },
            "executor_replan_history": [
                dict(item)
                for item in list(existing_ledger.get("executor_replan_history") or [])
                if isinstance(item, dict)
            ][-12:],
            "last_executor_replan_reason": str(
                existing_ledger.get("last_executor_replan_reason") or ""
            ).strip(),
        }
        issue_plan.task_state_context = existing_task_state_context

    def _append_shadow_policy_trace(
        self,
        metadata: dict[str, Any],
        trace: dict[str, Any],
    ) -> None:
        if not trace:
            return
        traces = list(metadata.get("shadow_policy_log") or [])
        traces.append(dict(trace))
        metadata["shadow_policy_log"] = traces[-12:]

    def _shadow_policy_limit(self) -> int:
        policy = getattr(self.config.planning, "shadow_policy", None)
        return max(
            1,
            int(getattr(policy, "max_logged_options", 3) or 3),
        )

    def _delegation_policy_config(self) -> Any:
        return getattr(self.config.planning, "delegation_policy", None)

    def _delegation_boundary_pressure_threshold(self) -> int:
        configured = int(
            getattr(self.config.planning, "delegation_boundary_pressure_threshold", 0) or 0
        )
        policy = self._delegation_policy_config()
        policy_value = int(getattr(policy, "boundary_pressure_threshold", 0) or 0)
        return max(1, configured or policy_value or 1)

    def _completion_like(self, issue_plan: IssuePlan) -> bool:
        return (
            self._task_regime_probability(issue_plan.task_regime, "contract_gap")
            >= self.regime_policy.threshold("contract_gap")
        ) or bool(
            issue_plan.test_context.incomplete_source_files
            or issue_plan.test_context.incomplete_test_files
        )

    def _collect_interface_symbols(
        self,
        repo_context: RepoContext,
        candidate_files: list[str],
    ) -> list[str]:
        counts: Counter[str] = Counter()
        interface_symbols: list[str] = []
        for path in _dedupe_preserve(candidate_files)[:8]:
            file_info = repo_context.get_file_info(path)
            if file_info is None:
                continue
            for symbol in list(file_info.symbols or [])[:8]:
                name = str(symbol.name or "").strip()
                if not name:
                    continue
                counts[name] += 1
                callers = [
                    node
                    for node in repo_context.trace_callers(name)[:4]
                    if str(getattr(node, "file_path", "") or "").strip() != path
                ]
                callees = [
                    node
                    for node in repo_context.trace_callees(name)[:4]
                    if str(getattr(node, "file_path", "") or "").strip() != path
                ]
                if callers or callees:
                    interface_symbols.append(name)
        interface_symbols.extend(name for name, count in counts.items() if count > 1)
        return _dedupe_preserve(interface_symbols)[:8]

    def _infer_task_regime(
        self,
        *,
        issue_description: str,
        repo_context: RepoContext,
        relevant_files: list[str],
        baseline_result: Optional[Any],
        test_context: TestContext,
        evaluation_constraints: EvaluationConstraints,
    ) -> TaskRegimeProfile:
        traceback_signal = self._extract_traceback_signal(
            repo_context,
            _baseline_output(baseline_result),
        )
        failing_test_ids = list(test_context.failing_test_ids or []) or _baseline_test_ids(
            baseline_result,
            "failing_tests",
        )
        passing_test_ids = list(test_context.passing_test_ids or []) or _baseline_test_ids(
            baseline_result,
            "passing_tests",
        )
        if not failing_test_ids and int(test_context.failing_test_count or 0) > 0:
            failing_test_ids = _dedupe_preserve(
                list(test_context.focus_test_files or []) + list(traceback_signal.test_files)
            )[: max(1, min(int(test_context.failing_test_count or 0), 4))]
        terminal_source_files = list(test_context.terminal_source_files or []) or list(
            traceback_signal.terminal_source_files
        )
        source_focus_files = list(test_context.source_focus_files or []) or _dedupe_preserve(
            list(traceback_signal.referenced_source_files) + list(traceback_signal.source_files)
        )
        incomplete_source_files = list(test_context.incomplete_source_files or [])
        incomplete_test_files = list(test_context.incomplete_test_files or [])
        exception_summaries = list(test_context.exception_summaries or []) or list(
            traceback_signal.exception_summaries
        )
        interface_symbols = self._collect_interface_symbols(
            repo_context,
            _dedupe_preserve(
                list(relevant_files)
                + terminal_source_files
                + source_focus_files
                + incomplete_source_files
                + list(traceback_signal.referenced_source_files)
            )[:8],
        )
        return self.regime_policy.infer(
            issue_description=issue_description,
            failing_test_ids=failing_test_ids,
            passing_test_ids=passing_test_ids,
            terminal_source_files=terminal_source_files,
            source_focus_files=source_focus_files,
            incomplete_source_files=incomplete_source_files,
            incomplete_test_files=incomplete_test_files,
            relevant_files=relevant_files,
            exception_summaries=exception_summaries,
            interface_symbols=interface_symbols,
            preserve_collected_test_coverage=bool(
                evaluation_constraints.preserve_collected_test_coverage
            ),
            relevant_file_languages=_file_language_hints(
                repo_context,
                _dedupe_preserve(
                    list(relevant_files)
                    + terminal_source_files
                    + source_focus_files
                    + incomplete_source_files
                    + incomplete_test_files
                ),
            ),
            repo_languages=_repo_language_hints(repo_context),
            test_command=test_context.command,
        )

    def _owned_symbol_names(
        self,
        repo_context: RepoContext,
        files: list[str],
    ) -> list[str]:
        symbols: list[str] = []
        for path in _dedupe_preserve(files):
            file_info = repo_context.get_file_info(path)
            if file_info is None:
                continue
            for symbol in list(file_info.symbols or [])[:8]:
                name = str(symbol.name or "").strip()
                if name:
                    symbols.append(name)
        return _dedupe_preserve(symbols)[:8]

    def _owned_edit_spans(
        self,
        repo_context: RepoContext,
        files: list[str],
        symbols: list[str],
    ) -> list[dict[str, Any]]:
        spans: list[dict[str, Any]] = []
        file_set = set(files)
        for symbol_name in _dedupe_preserve(symbols)[:6]:
            for node in repo_context.lookup_definition(symbol_name)[:4]:
                file_path = str(getattr(node, "file_path", "") or "").strip()
                if file_path not in file_set:
                    continue
                spans.append(
                    {
                        "file_path": file_path,
                        "symbol": symbol_name,
                        "start_line": int(getattr(node, "start_line", 0) or 0),
                        "end_line": int(getattr(node, "end_line", 0) or 0),
                    }
                )
        return spans[:8]

    def _same_llm_identity(
        self,
        left: LLMConfig,
        right: LLMConfig,
    ) -> bool:
        return (
            left.backend == right.backend
            and left.model == right.model
            and left.resolved_cli_command == right.resolved_cli_command
        )

    def _phase_hard_timeout_seconds(
        self,
        llm_config: LLMConfig,
        configured_seconds: Optional[int],
    ) -> Optional[int]:
        if (
            not llm_config.is_cli_backend
            or not isinstance(configured_seconds, int)
            or configured_seconds <= 0
        ):
            return None
        return configured_seconds

    def _heuristic_seed_plan_is_rich(
        self,
        heuristic: IssuePlan,
        *,
        requested_rollouts: int,
    ) -> bool:
        briefs = list(heuristic.rollout_briefs or [])
        if len(briefs) < 2:
            return False
        search_modes = {
            str((brief.search_policy or {}).get("mode") or "").strip().lower()
            for brief in briefs
            if isinstance(brief.search_policy, dict)
            and str((brief.search_policy or {}).get("mode") or "").strip()
        }
        focus_clusters = {
            tuple(list(brief.focus_files or [])[:4])
            for brief in briefs
            if list(brief.focus_files or [])
        }
        family_target = min(
            max(2, requested_rollouts), max(3, self.config.planning.max_rollout_brief_families)
        )
        return (
            len(search_modes) >= min(3, family_target)
            and len(focus_clusters) >= min(3, family_target)
            and self._plan_has_explicit_single_agent_family(briefs)
        )

    def _skip_external_preplanner_reason(
        self,
        heuristic: IssuePlan,
        *,
        requested_rollouts: int,
        preplanner_llm: LLMConfig,
        planner_llm: LLMConfig,
    ) -> Optional[str]:
        if not self.config.planning.enable_coarse_to_fine_planning:
            return None
        if self._same_llm_identity(preplanner_llm, planner_llm):
            return "same_planner_identity"
        if not self.config.planning.allow_preplanner_skip_on_rich_heuristic_seed:
            return None
        if (
            preplanner_llm.is_cli_backend
            and self.config.use_concise_prompts
            and self._heuristic_seed_plan_is_rich(
                heuristic,
                requested_rollouts=requested_rollouts,
            )
        ):
            return "rich_heuristic_seed"
        return None

    def _should_refine_coarse_plan(
        self,
        issue_description: str,
        *,
        heuristic: IssuePlan,
        coarse_plan: IssuePlan,
        requested_rollouts: int,
        preplanner_llm: LLMConfig,
        planner_llm: LLMConfig,
    ) -> bool:
        if not self.config.planning.enable_coarse_to_fine_planning:
            return False
        if self._same_llm_identity(preplanner_llm, planner_llm):
            return False
        if requested_rollouts >= max(8, len(coarse_plan.rollout_briefs)):
            return True
        if self._task_regime_probability(heuristic.task_regime, "contract_gap") >= 0.5:
            return True
        if len(heuristic.relevant_files) >= 10 or len(heuristic.risk_files) >= 4:
            return True
        if (
            self.config.planning.enable_plan_portfolio
            and not self._plan_has_explicit_single_agent_family(coarse_plan.rollout_briefs)
        ):
            return True
        return False

    def _plan_has_explicit_single_agent_family(
        self,
        rollout_briefs: list[RolloutBrief],
    ) -> bool:
        for brief in rollout_briefs:
            policy = brief.delegation_policy if isinstance(brief.delegation_policy, dict) else {}
            if not bool(policy.get("enabled")):
                return True
        return False

    def _plan_has_agentless_pipeline_family(
        self,
        rollout_briefs: list[RolloutBrief],
    ) -> bool:
        for brief in rollout_briefs:
            policy = brief.search_policy if isinstance(brief.search_policy, dict) else {}
            mode = str(policy.get("mode") or "").strip().lower()
            if mode == "agentless_pipeline":
                return True
        return False

    def _build_heuristic_plan(
        self,
        issue_description: str,
        repo_context: RepoContext,
        rollout_count: Optional[int] = None,
        difficulty: Optional[float] = None,
        baseline_result: Optional[Any] = None,
        task_regime: Optional[TaskRegimeProfile] = None,
    ) -> IssuePlan:
        effective_rollout_count = rollout_count or self.config.rollout.num_rollouts
        keywords = repo_context.extract_issue_keywords(
            issue_description,
            max_keywords=self.config.planning.max_keywords,
        )
        relevant_files = repo_context.get_relevant_files(
            keywords,
            max_files=self.config.planning.max_relevant_files,
        )
        if not isinstance(task_regime, TaskRegimeProfile):
            task_regime = self._infer_task_regime(
                issue_description=issue_description,
                repo_context=repo_context,
                relevant_files=relevant_files,
                baseline_result=baseline_result,
                test_context=TestContext(),
                evaluation_constraints=EvaluationConstraints(),
            )
        else:
            task_regime = task_regime or TaskRegimeProfile()
        completion_like = self._task_regime_probability(task_regime, "contract_gap") >= 0.5
        source_focus_files = self._select_source_focus_files(
            issue_description=issue_description,
            repo_context=repo_context,
            keywords=keywords,
            relevant_files=relevant_files,
            task_regime=task_regime,
        )
        baseline_focus_files = self._extract_baseline_focus_files(
            repo_context=repo_context,
            baseline_result=baseline_result,
        )
        baseline_source_files = [
            path for path in baseline_focus_files if not self._looks_like_test_path(path)
        ]
        baseline_test_files = [
            path for path in baseline_focus_files if self._looks_like_test_path(path)
        ]
        if not relevant_files and not source_focus_files and not baseline_focus_files:
            relevant_files = self._fallback_relevant_files(repo_context)
        prioritized_files = (
            baseline_source_files + source_focus_files + baseline_test_files + relevant_files
            if completion_like
            else baseline_source_files + baseline_test_files + source_focus_files + relevant_files
        )
        relevant_files = self._normalize_repo_file_hints(
            repo_context,
            prioritized_files,
        )
        if not relevant_files:
            relevant_files = list(self._fallback_relevant_files(repo_context))
        neighbor_files = repo_context.get_dependency_neighbors(
            relevant_files[: min(4, len(relevant_files))],
            max_neighbors=self.config.planning.include_dependency_neighbors,
        )
        combined_files = list(dict.fromkeys(relevant_files + neighbor_files))
        repo_focus_map = repo_context.build_context_pack(
            combined_files[: self.config.planning.max_repo_map_files],
            max_symbols_per_file=8,
            seed_symbols=keywords,
        )
        success_criteria = self._build_success_criteria(issue_description)
        rollout_briefs = self._build_rollout_briefs(
            issue_description=issue_description,
            repo_context=repo_context,
            relevant_files=combined_files,
            success_criteria=success_criteria,
            rollout_count=effective_rollout_count,
            task_regime=task_regime,
        )

        summary = self._extract_issue_summary(issue_description)
        return IssuePlan(
            summary=summary,
            keywords=keywords,
            relevant_files=combined_files,
            risk_files=combined_files[: min(3, len(combined_files))],
            success_criteria=success_criteria,
            rollout_briefs=rollout_briefs,
            repo_focus_map=repo_focus_map,
            planner_source="heuristic",
            difficulty_estimate=difficulty,
            recommended_rollouts=effective_rollout_count,
            task_regime=task_regime,
            planner_metadata={
                "requested_rollouts": effective_rollout_count,
                "brief_family_count": min(
                    len({brief.title for brief in rollout_briefs}),
                    effective_rollout_count,
                ),
                "family_cap": self.config.planning.max_rollout_brief_families,
                "expansion_mode": "heuristic_diversified",
                **self._task_regime_metadata(task_regime),
            },
        )

    def enrich_issue_plan(
        self,
        issue_plan: IssuePlan,
        *,
        issue_description: str,
        repo_context: RepoContext,
        test_command: Optional[str] = None,
        baseline_result: Optional[Any] = None,
        benchmark_metadata: Optional[dict[str, Any]] = None,
        force_test_generation_design_replan: bool = False,
        test_generation_boundary_context: Optional[dict[str, Any]] = None,
    ) -> IssuePlan:
        """Attach structured task and visible-test context to an existing plan."""
        issue_plan.summary = self._extract_issue_summary(issue_description)
        issue_plan.evaluation_constraints = self._build_evaluation_constraints(
            benchmark_metadata=benchmark_metadata,
            test_command=test_command,
        )
        issue_plan.test_context = self._build_test_context(
            issue_description=issue_description,
            repo_context=repo_context,
            relevant_files=issue_plan.relevant_files,
            keywords=issue_plan.keywords,
            test_command=test_command,
            baseline_result=baseline_result,
            evaluation_constraints=issue_plan.evaluation_constraints,
        )
        issue_plan.task_regime = self._infer_task_regime(
            issue_description=issue_description,
            repo_context=repo_context,
            relevant_files=issue_plan.relevant_files,
            baseline_result=baseline_result,
            test_context=issue_plan.test_context,
            evaluation_constraints=issue_plan.evaluation_constraints,
        )
        issue_plan.test_context.planner_invariants = _dedupe_preserve(
            list(issue_plan.test_context.planner_invariants)
            + list(issue_plan.evaluation_constraints.planner_invariants())
            + list(issue_plan.task_regime.planner_invariants)
        )
        preserve_bypass_reason = bool(
            (issue_plan.planner_metadata or {}).get("planner_bypass_reason")
        )
        issue_plan.planner_metadata = dict(issue_plan.planner_metadata or {})
        self._apply_collection_error_compatibility_metadata(
            issue_plan,
            repo_context=repo_context,
            baseline_result=baseline_result,
            allow_bypass_reason=preserve_bypass_reason,
        )
        issue_plan.planner_metadata.update(self._task_regime_metadata(issue_plan.task_regime))
        task_state_context = (
            dict(issue_plan.task_state_context)
            if isinstance(issue_plan.task_state_context, dict)
            else {}
        )
        task_state_context.update(blackboard_context_from_issue_plan(issue_plan))
        issue_plan.task_state_context = task_state_context
        self._attach_test_generation_design_metadata(
            issue_plan,
            issue_description=issue_description,
            repo_context=repo_context,
            force_replan=force_test_generation_design_replan,
            boundary_context=test_generation_boundary_context,
        )
        # TIER 2 decomposition (T2.3): for decomposition-scale repos, replace the
        # strategy-family round-robin briefs with one enforced-write-scope brief
        # per disjoint module group so each rollout owns a tractable slice and
        # the disjoint partials union (T2.5) instead of every rollout attacking
        # the whole repo and passing ~0. No-op for small/non-decomposition repos.
        self._maybe_apply_decomposition_scale_briefs(
            issue_plan,
            repo_context,
            benchmark_metadata=benchmark_metadata,
        )
        if not bool(
            (issue_plan.planner_metadata or {}).get("decomposition_scale_partitioned")
        ):
            self._retune_rollout_briefs_with_test_context(
                issue_plan,
                repo_context,
                issue_description=issue_description,
            )
            self._prune_redundant_overlap_sensitive_rollout_variants(issue_plan)
        return issue_plan

    def _maybe_apply_decomposition_scale_briefs(
        self,
        issue_plan: "IssuePlan",
        repo_context: RepoContext,
        *,
        benchmark_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Swap in module-group briefs when the repo is decomposition-scale (T2.3)."""
        num_rollouts = int(
            (issue_plan.recommended_rollouts or 0)
            or len(issue_plan.rollout_briefs)
            or self.config.rollout.num_rollouts
        )
        expected_id_mapper = None
        expected_id_partitioner = None
        if isinstance(benchmark_metadata, dict):
            candidate = benchmark_metadata.get("module_group_expected_id_mapper")
            if callable(candidate):
                expected_id_mapper = candidate
            partitioner = benchmark_metadata.get("module_group_expected_id_partitioner")
            if callable(partitioner):
                expected_id_partitioner = partitioner
        try:
            briefs, groups = self._build_module_group_rollout_briefs(
                issue_plan,
                repo_context,
                success_criteria=list(issue_plan.success_criteria or []),
                num_rollouts=num_rollouts,
                expected_id_mapper=expected_id_mapper,
                expected_id_partitioner=expected_id_partitioner,
            )
        except Exception as exc:  # noqa: BLE001 - fail open to today's behavior
            logger.warning("Decomposition-scale brief build failed; keeping defaults: %s", exc)
            return
        if len(briefs) < 2:
            return
        issue_plan.rollout_briefs = briefs
        issue_plan.planner_metadata = dict(issue_plan.planner_metadata or {})
        issue_plan.planner_metadata["decomposition_scale_partitioned"] = True
        issue_plan.planner_metadata["module_group_count"] = len(groups)
        issue_plan.planner_metadata["module_groups"] = [group.to_dict() for group in groups]
        logger.info(
            "Decomposition-scale repo: generated %d module-group briefs (one per rollout).",
            len(briefs),
        )

    def extract_difficulty_features(
        self,
        issue: str,
        repo_context: RepoContext,
        *,
        baseline_result: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Extract deterministic features for rollout allocation."""
        return self.allocator.extract_features(
            issue,
            repo_context,
            baseline_result=baseline_result,
        )

    def estimate_difficulty(
        self,
        issue: str,
        repo_context: RepoContext,
        *,
        baseline_result: Optional[Any] = None,
    ) -> float:
        """Estimate issue difficulty on a 0.0-1.0 scale."""
        return self.allocator.estimate_difficulty(
            self.extract_difficulty_features(
                issue,
                repo_context,
                baseline_result=baseline_result,
            )
        )

    def compute_rollout_count(
        self,
        difficulty: float,
        min_rollouts: Optional[int] = None,
        max_rollouts: Optional[int] = None,
    ) -> int:
        """Map a difficulty score to a configured rollout count."""
        minimum = min_rollouts or self.config.rollout.min_rollouts
        maximum = max_rollouts or self.config.rollout.max_rollouts
        eligible = self._eligible_rollout_buckets(minimum, maximum)
        if not eligible:
            return _interpolate_rollout_count(minimum, maximum, difficulty)
        if difficulty <= 0.25:
            return eligible[0]
        if difficulty <= 0.55:
            return eligible[min(1, len(eligible) - 1)]
        if difficulty <= 0.8:
            return eligible[min(2, len(eligible) - 1)]
        return eligible[-1]

    def _portfolio_rollout_floor(
        self,
        features: dict[str, Any],
        *,
        difficulty: float,
        rollout_cap: int,
        explicit_rollout_count: Optional[int],
    ) -> tuple[int, int]:
        normalized_cap = max(1, int(rollout_cap))
        profile_window = max(1, int(self.config.rollout.max_rollouts))
        profile_count = max(
            1,
            self.config.count_distinct_rollout_profiles(
                profile_window,
                include_prompt_strategy=bool(
                    self.config.rollout.portfolio_diversity_include_prompt_strategy
                ),
                include_temperature=bool(
                    self.config.rollout.portfolio_diversity_include_temperature
                ),
            ),
        )
        configured_seed_profiles = int(self.config.rollout.portfolio_seed_profile_count or 0)
        if configured_seed_profiles > 0:
            profile_budget = min(profile_count, configured_seed_profiles)
        else:
            profile_budget = profile_count
        if explicit_rollout_count is not None or normalized_cap <= 1 or profile_budget <= 1:
            return 1, profile_count

        estimated_files = int(features.get("estimated_files_to_edit", 1) or 1)
        failing_test_count = int(features.get("failing_test_count", 0) or 0)
        hard_task = bool(
            features.get("is_completion_task")
            or features.get("mentions_public_api")
            or self.config.search.mode != SearchMode.OFF
            or difficulty >= 0.55
        )
        if hard_task:
            heuristic_floor = min(normalized_cap, profile_budget)
        else:
            medium_task = (
                difficulty >= 0.35
                or estimated_files > 2
                or failing_test_count >= 5
                or not bool(features.get("has_stack_trace"))
            )
            if medium_task:
                heuristic_floor = min(normalized_cap, max(2, min(profile_budget, 3)))
            else:
                heuristic_floor = min(normalized_cap, max(1, min(profile_budget, 2)))

        evaluation = evaluate_policy_model(
            getattr(self.config, "controller_models", None),
            model_name="planning.portfolio_rollout_floor",
            features={
                **dict(features or {}),
                "difficulty_estimate": float(difficulty),
                "profile_budget": float(profile_budget),
                "rollout_cap": float(normalized_cap),
                "heuristic_score": float(heuristic_floor),
            },
            baseline_value=float(heuristic_floor),
            lower=1.0,
            upper=float(normalized_cap),
        )
        calibrated_floor = int(round(float(evaluation.value or heuristic_floor)))
        return max(1, min(normalized_cap, calibrated_floor)), profile_count

    def build_execution_strategy(
        self,
        issue_description: str,
        repo_context: RepoContext,
        rollout_count: Optional[int] = None,
        baseline_result: Optional[Any] = None,
    ) -> PlanningDecision:
        """Choose rollout count, scaffold primitives, and solvability status."""
        features = self.extract_difficulty_features(
            issue_description,
            repo_context,
            baseline_result=baseline_result,
        )
        difficulty = self.allocator.estimate_difficulty(features)

        requested_rollouts = self._requested_rollout_budget(rollout_count)
        rollout_cap = max(
            self.config.rollout.min_rollouts,
            min(self.config.rollout.max_rollouts, requested_rollouts),
        )
        if self.config.rollout.enable_adaptive_allocation:
            selected_rollouts = self.compute_rollout_count(
                difficulty,
                min_rollouts=self.config.rollout.min_rollouts,
                max_rollouts=rollout_cap,
            )
            selected_rollouts_eval = evaluate_policy_model(
                getattr(self.config, "controller_models", None),
                model_name="planning.rollout_count",
                features={
                    **dict(features or {}),
                    "difficulty_estimate": float(difficulty),
                    "rollout_cap": float(rollout_cap),
                    "heuristic_score": float(selected_rollouts),
                },
                baseline_value=float(selected_rollouts),
                lower=float(self.config.rollout.min_rollouts),
                upper=float(rollout_cap),
            )
            selected_rollouts = self._clamp_rollout_bucket(
                int(round(float(selected_rollouts_eval.value or selected_rollouts)))
            )
            unsolvable_reason = self.allocator.predict_unsolvable(issue_description, repo_context)
        else:
            selected_rollouts = rollout_cap
            unsolvable_reason = None

        portfolio_rollout_floor, portfolio_profile_count = self._portfolio_rollout_floor(
            features,
            difficulty=difficulty,
            rollout_cap=rollout_cap,
            explicit_rollout_count=rollout_count,
        )
        portfolio_rollout_floor_applied = selected_rollouts < portfolio_rollout_floor
        if portfolio_rollout_floor_applied:
            selected_rollouts = portfolio_rollout_floor

        component_ablation = component_ablation_assignment_for_task(
            config=self.config,
            issue_description=issue_description,
            repo_label=str(getattr(repo_context, "repo_path", "") or ""),
        )
        if component_disabled(component_ablation, "multi_rollout"):
            selected_rollouts = 1
            portfolio_rollout_floor_applied = False

        primitives = self.select_orchestration_primitives(features, selected_rollouts)
        agent_mode = self._agent_mode_for_primitives(primitives)
        if self.config.rollout.agent_mode != AgentMode.ADAPTIVE:
            agent_mode = self.config.rollout.agent_mode

        append_controller_decision(
            self.config,
            stage="planning",
            decision_type="execution_strategy",
            chosen_option=f"rollouts:{selected_rollouts}",
            feature_view={
                **dict(features or {}),
                "difficulty_estimate": float(difficulty),
                "portfolio_rollout_floor": float(portfolio_rollout_floor),
                "portfolio_profile_count": float(portfolio_profile_count),
                "selected_rollouts": float(selected_rollouts),
            },
            options=[
                {
                    "option_id": f"rollouts:{selected_rollouts}",
                    "score": float(selected_rollouts),
                    "selected": True,
                    "category": "execution_strategy",
                    "metadata": {
                        "agent_mode": agent_mode.value,
                        "primitives": [primitive.value for primitive in primitives],
                        "portfolio_rollout_floor_applied": bool(portfolio_rollout_floor_applied),
                        "component_ablation": dict(component_ablation),
                    },
                }
            ],
            metadata={
                "unsolvable_reason": unsolvable_reason,
            },
        )
        # WS3B: compute the speculative-first-attempt flag from difficulty. Only
        # consulted when adaptive allocation is OFF (adaptive already stages via
        # difficulty buckets); never touches selected_rollouts (no-cost-reduction).
        speculative_first_attempt = bool(
            getattr(self.config.rollout, "enable_speculative_first_attempt", False)
            and not self.config.rollout.enable_adaptive_allocation
            and difficulty
            <= float(getattr(self.config.rollout, "speculative_first_attempt_max_difficulty", 0.25))
        )
        return PlanningDecision(
            rollout_count=selected_rollouts,
            difficulty_estimate=difficulty,
            features=features,
            primitives=primitives,
            agent_mode=agent_mode,
            unsolvable_reason=unsolvable_reason,
            portfolio_profile_count=portfolio_profile_count,
            portfolio_rollout_floor=portfolio_rollout_floor,
            portfolio_rollout_floor_applied=portfolio_rollout_floor_applied,
            component_ablation=dict(component_ablation),
            speculative_first_attempt=speculative_first_attempt,
        )

    def escalate_execution_strategy(
        self,
        decision: PlanningDecision,
    ) -> Optional[PlanningDecision]:
        """Escalate the COP scaffold when a simpler runtime gets stuck."""
        if (
            not self.config.rollout.enable_dynamic_cop_transitions
            or self.config.rollout.agent_mode != AgentMode.ADAPTIVE
            or decision.unsolvable_reason
        ):
            return None

        next_primitives: Optional[list[Primitive]] = None
        next_rollouts = decision.rollout_count

        if decision.primitives == [Primitive.REACT]:
            next_primitives = [Primitive.PLAN_EXEC, Primitive.GTR]
            next_rollouts = self._clamp_rollout_bucket(max(decision.rollout_count, 4))
        elif Primitive.MCTS not in decision.primitives and self.config.execution_tree.enabled:
            next_primitives = list(decision.primitives) + [Primitive.MCTS]
            next_rollouts = self._clamp_rollout_bucket(max(decision.rollout_count, 8))

        if next_primitives is None:
            return None

        return PlanningDecision(
            rollout_count=next_rollouts,
            difficulty_estimate=max(decision.difficulty_estimate, 0.6),
            features=dict(decision.features),
            primitives=next_primitives,
            agent_mode=self._agent_mode_for_primitives(next_primitives),
            unsolvable_reason=None,
        )

    def select_orchestration_primitives(
        self,
        features: dict[str, Any],
        rollout_count: int,
    ) -> list[Primitive]:
        """Choose a lightweight COP scaffold that fits the current runtime."""
        estimated_files = int(features.get("estimated_files_to_edit", 1))
        if rollout_count <= 1 and estimated_files <= 3 and features.get("has_stack_trace"):
            return [Primitive.REACT]

        primitives = [Primitive.PLAN_EXEC, Primitive.GTR]
        if rollout_count >= 8 and self.config.execution_tree.enabled:
            primitives.append(Primitive.MCTS)
        return primitives

    def apply_execution_strategy(
        self,
        issue_plan: IssuePlan,
        decision: PlanningDecision,
    ) -> IssuePlan:
        """Persist allocator/scaffold metadata into the issue plan."""
        issue_plan.difficulty_estimate = decision.difficulty_estimate
        issue_plan.recommended_rollouts = decision.rollout_count
        issue_plan.orchestration_primitives = [primitive.value for primitive in decision.primitives]
        issue_plan.allocator_features = dict(decision.features)
        issue_plan.unsolvable_reason = decision.unsolvable_reason
        planner_metadata = dict(issue_plan.planner_metadata or {})
        # WS3B: surface the speculative flag for the orchestrator (which receives
        # issue_plan, not the raw decision).
        planner_metadata["execution_strategy_rollout_count"] = int(decision.rollout_count)
        planner_metadata["speculative_first_attempt"] = bool(decision.speculative_first_attempt)
        planner_metadata["portfolio_profile_count"] = int(decision.portfolio_profile_count)
        planner_metadata["portfolio_rollout_floor"] = int(decision.portfolio_rollout_floor)
        planner_metadata["portfolio_rollout_floor_applied"] = bool(
            decision.portfolio_rollout_floor_applied
        )
        planner_metadata["portfolio_seed_profile_count"] = int(
            self.config.rollout.portfolio_seed_profile_count or 0
        )
        planner_metadata["portfolio_diversity_include_prompt_strategy"] = bool(
            self.config.rollout.portfolio_diversity_include_prompt_strategy
        )
        planner_metadata["portfolio_diversity_include_temperature"] = bool(
            self.config.rollout.portfolio_diversity_include_temperature
        )
        issue_plan.planner_metadata = planner_metadata
        gtr_enabled = Primitive.GTR in decision.primitives
        for brief in issue_plan.rollout_briefs:
            search_policy = brief.search_policy if isinstance(brief.search_policy, dict) else {}
            if not bool(search_policy.get("preserve_agent_mode")):
                brief.agent_mode = decision.agent_mode
            brief.prompt_hint = self._compose_prompt_hint(brief.prompt_hint, decision.primitives)
            if not gtr_enabled:
                continue
            policy = (
                dict(brief.delegation_policy) if isinstance(brief.delegation_policy, dict) else {}
            )
            if not bool(policy.get("enabled")):
                continue
            allowed_stages = _normalize_delegation_allowed_stages(
                list(policy.get("allowed_stages") or [])
            )
            if "test_writer" in allowed_stages:
                continue
            policy["allowed_stages"] = _dedupe_preserve([*allowed_stages, "test_writer"])
            brief.delegation_policy = policy
        return issue_plan

    def recommend_followup_rollouts(
        self,
        issue_plan: IssuePlan,
        rollout_results: list[Any],
        *,
        best_candidate: Optional[Any] = None,
        current_total_rollouts: Optional[int] = None,
    ) -> int:
        current_rollouts = current_total_rollouts or len(issue_plan.rollout_briefs)
        max_rollouts = int(self.config.rollout.max_rollouts)
        if self._unbounded_followup_budget_enabled() and current_rollouts >= max_rollouts:
            # Max-quality runs use the follow-up loop caps as the residual-repair
            # guard; the initial exploration ceiling must not starve a near-miss
            # candidate after all baseline rollout budget has been spent.
            per_round_extra = max(1, min(8, max_rollouts or current_rollouts or 1))
            max_rollouts = current_rollouts + max(per_round_extra, self.config.rollout.min_rollouts)
        return self.allocator.recommend_followup_rollouts(
            rollout_results,
            current_rollouts=current_rollouts,
            min_rollouts=self.config.rollout.min_rollouts,
            max_rollouts=max_rollouts,
            best_candidate=best_candidate,
        )

    def _unbounded_followup_budget_enabled(self) -> bool:
        benchmark_config = getattr(self.config, "benchmark", None)
        power_mode = (
            str(
                getattr(benchmark_config, "evaluation_power_mode", "")
                or getattr(benchmark_config, "power_mode", "")
                or ""
            )
            .strip()
            .lower()
        )
        return bool(
            getattr(benchmark_config, "unbounded_followup_budget", False)
            or power_mode in {"max", "maximum", "max_quality", "unlimited", "full_max"}
        )

    def score_rollout_progress(self, rollout_result: Any) -> float:
        """Score how much useful signal a rollout produced, even before verification."""
        return self.allocator._rollout_reward(rollout_result)

    def should_use_progressive_rollout_allocation(self, issue_plan: IssuePlan) -> bool:
        """Return whether large rollout budgets should be executed in staged waves."""
        if not self.config.rollout.enable_progressive_rollout_allocation:
            return False
        total_rollouts = len(issue_plan.rollout_briefs)
        if total_rollouts <= 2:
            return False
        family_count = issue_plan.planner_metadata.get("portfolio_brief_family_count")
        if not isinstance(family_count, int) or family_count <= 0:
            family_count = issue_plan.planner_metadata.get("brief_family_count")
        if not isinstance(family_count, int) or family_count <= 0:
            family_count = len(
                {self._progressive_brief_family_key(brief) for brief in issue_plan.rollout_briefs}
            )
        return total_rollouts > max(1, family_count)

    def select_progressive_seed_briefs(self, issue_plan: IssuePlan) -> list[RolloutBrief]:
        """Seed a large rollout budget with one brief per family, plus one extra when possible."""
        briefs = issue_plan.rollout_briefs
        if not briefs:
            return []

        family_count = issue_plan.planner_metadata.get("portfolio_brief_family_count")
        if not isinstance(family_count, int) or family_count <= 0:
            family_count = issue_plan.planner_metadata.get("brief_family_count")
        if not isinstance(family_count, int) or family_count <= 0:
            family_count = len({self._progressive_brief_family_key(brief) for brief in briefs})
        target = min(len(briefs), max(1, family_count))
        if len(briefs) > 1:
            target = min(len(briefs), max(2, target))
        profile_target = self.config.count_distinct_rollout_profiles(
            len(briefs),
            include_prompt_strategy=bool(
                self.config.rollout.portfolio_diversity_include_prompt_strategy
            ),
            include_temperature=bool(self.config.rollout.portfolio_diversity_include_temperature),
        )
        configured_seed_profiles = int(self.config.rollout.portfolio_seed_profile_count or 0)
        if configured_seed_profiles > 0:
            profile_target = min(profile_target, configured_seed_profiles)
        if profile_target > 1:
            # SPEED LEVER (RANK-3A: size-aware seed-wave WIDTH). The diversity
            # ``profile_target`` is the full historical seed width. For SMALL
            # suites we narrow the first wave toward the diverse-strategy family
            # floor (>= 2, best-of-2 + >=1 diverse strategy preserved) so tiny
            # repos do not pin ~8 long rollouts; GIANTS keep the full
            # ``profile_target`` byte-identical (size_factor >= max => the scaled
            # width == profile_target, so this collapses to today's
            # ``max(target, profile_target)``). The deferred briefs are NEVER
            # dropped from ``issue_plan.rollout_briefs`` — only the SEED count is
            # narrowed, so wave-continuation / residual deepening / best-of-N
            # over-run still draw from the full deque. Fully fail-open.
            full_profile_target = min(len(briefs), max(target, profile_target))
            target = self._size_aware_seed_width(
                issue_plan,
                family_floor=target,
                profile_target=full_profile_target,
            )

        seeds: list[RolloutBrief] = []
        selected_indices: set[int] = set()
        seen_families: set[Any] = set()
        seen_profiles: set[str] = set()
        seen_allocation_arms: set[str] = set()
        for index, brief in enumerate(briefs):
            family_key = self._progressive_brief_family_key(brief)
            if family_key in seen_families:
                continue
            seen_families.add(family_key)
            selected_indices.add(index)
            profile_key = _rollout_brief_profile_key(brief)
            if profile_key:
                seen_profiles.add(profile_key)
            seen_allocation_arms.add(_rollout_brief_allocation_key(brief))
            seeds.append(RolloutBrief.from_dict(brief.to_dict()))
            if len(seeds) >= target:
                return seeds

        for index, brief in enumerate(briefs):
            if len(seeds) >= target:
                break
            if index in selected_indices:
                continue
            profile_key = _rollout_brief_profile_key(brief)
            if not profile_key or profile_key in seen_profiles:
                continue
            selected_indices.add(index)
            seen_profiles.add(profile_key)
            seen_allocation_arms.add(_rollout_brief_allocation_key(brief))
            seeds.append(RolloutBrief.from_dict(brief.to_dict()))

        for index, brief in enumerate(briefs):
            if len(seeds) >= target:
                break
            if index in selected_indices:
                continue
            allocation_key = _rollout_brief_allocation_key(brief)
            if allocation_key in seen_allocation_arms:
                continue
            selected_indices.add(index)
            seen_allocation_arms.add(allocation_key)
            seeds.append(RolloutBrief.from_dict(brief.to_dict()))

        for index, brief in enumerate(briefs):
            if len(seeds) >= target:
                break
            if index in selected_indices:
                continue
            seeds.append(RolloutBrief.from_dict(brief.to_dict()))
        return seeds

    def _size_aware_seed_width(
        self,
        issue_plan: IssuePlan,
        *,
        family_floor: int,
        profile_target: int,
    ) -> int:
        """RANK-3A: size-aware first-wave seed width.

        ``seed_target = clamp(family_floor, ceil(profile_target * size_factor /
        max_size_factor), profile_target)`` where ``size_factor`` is the EXISTING
        ``_rollout_budget_size_factor`` signal (1 for small/unknown suites,
        saturating at ``rollout_budget_max_size_factor`` for giants).

        Invariants:
          * ``family_floor`` is the diverse-strategy floor already computed by
            the caller (>= 2 for multi-brief plans), so best-of-2 + >= 1 diverse
            strategy is always preserved.
          * GIANTS (size_factor >= max(2, max_size_factor)) return
            ``profile_target`` BYTE-IDENTICAL to today (the scaled width equals
            profile_target and the clamp is a no-op).
          * size_factor == 1 narrows toward ``family_floor`` (tightening small
            suites only; never widens past ``profile_target``).
          * Monotone non-decreasing in the suite size.
          * Fail-open: any error returns ``profile_target`` (today's behavior).
        """

        try:
            floor = max(1, int(family_floor))
            full = max(floor, int(profile_target))
            rollout_cfg = getattr(self.config, "rollout", None)
            max_size_factor = int(
                getattr(rollout_cfg, "rollout_budget_max_size_factor", 6) or 6
            )
            tests_per_unit = int(
                getattr(rollout_cfg, "rollout_budget_tests_per_unit", 2000) or 2000
            )
            size_factor = _rollout_budget_size_factor(
                _issue_plan_expected_test_count(issue_plan),
                tests_per_unit=tests_per_unit,
                max_size_factor=max_size_factor,
            )
            # GIANTS keep the full width, byte-identical to today.
            if size_factor >= max(2, max_size_factor):
                return full
            denom = max(1, max_size_factor)
            scaled = -(-(full * size_factor) // denom)  # ceil(full * sf / max)
            return max(floor, min(scaled, full))
        except Exception as width_exc:  # noqa: BLE001 - fail-open to today's width
            logger.debug("Size-aware seed width gate failed: %s", width_exc)
            try:
                return max(int(family_floor), int(profile_target))
            except Exception:  # noqa: BLE001 - last-resort fail-open
                return profile_target

    def should_continue_progressive_waves(
        self,
        rollout_results: list[Any],
        *,
        remaining_budget: int,
        issue_plan: Optional[IssuePlan] = None,
        task_state_context: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Stop spending rollout budget when early waves are producing no meaningful signal."""
        if remaining_budget <= 0 or not rollout_results:
            return False
        if issue_plan is not None:
            progress_ledger = self.build_progress_ledger(
                issue_plan,
                rollout_results,
                task_state_context=task_state_context,
            )
            next_action = str(progress_ledger.get("next_action") or "").strip().lower()
            recovery_mode = (
                str(
                    issue_plan.planner_metadata.get("recovery_mode")
                    if isinstance(issue_plan.planner_metadata, dict)
                    else ""
                )
                .strip()
                .lower()
            )
            if next_action in {"widen_boundaries", "collapse_to_integrator", "relocalize"}:
                if recovery_mode == "localization_first" and next_action == "relocalize":
                    return False
                return True
        scores = [self.score_rollout_progress(result) for result in rollout_results]
        best_score = max(scores or [0.0])
        meaningful_rollouts = sum(1 for score in scores if score >= _MEANINGFUL_PROGRESS_SCORE)
        return best_score >= _LOW_PROGRESS_SCORE or meaningful_rollouts >= 2

    def summarize_progressive_signals(
        self,
        issue_plan: IssuePlan,
        rollout_results: list[Any],
    ) -> str:
        """Summarize the strongest evidence discovered so far for the next wave."""
        if not rollout_results:
            return ""
        ordered = sorted(
            rollout_results,
            key=self.score_rollout_progress,
            reverse=True,
        )
        best = ordered[0]
        best_score = self.score_rollout_progress(best)
        focus_files = self.extract_progressive_focus_files(issue_plan, rollout_results)
        parts = ["Earlier waves produced partial signal but no accepted patch yet."]
        if getattr(best, "plan_title", ""):
            parts.append(
                f"Strongest current rollout family: {best.plan_title} (progress={best_score:.2f})."
            )
        else:
            parts.append(f"Best current rollout progress score: {best_score:.2f}.")
        if focus_files:
            parts.append("Prioritize files around " + ", ".join(focus_files[:4]) + ".")
        if issue_plan.test_context.failing_test_ids:
            parts.append("Treat the visible failing tests as the primary validation target.")
        return " ".join(parts)

    def extract_progressive_focus_files(
        self,
        issue_plan: IssuePlan,
        rollout_results: list[Any],
    ) -> list[str]:
        """Rank files that keep appearing in promising partial trajectories."""
        counter: Counter[str] = Counter()
        for result in rollout_results:
            weight = max(self.score_rollout_progress(result), 0.05)
            magnitude = max(1, int(round(weight * 4)))
            for path in list(getattr(result, "changed_files", []) or []):
                counter[path] += 2 * magnitude
            for path in self._artifact_list(
                getattr(result, "localization_artifact", None), "files"
            ):
                counter[path] += 2 * magnitude
            for path in self._artifact_list(
                getattr(result, "patch_artifact", None), "changed_files"
            ):
                counter[path] += magnitude

        ordered = [path for path, _ in counter.most_common(8)]
        merged = list(
            dict.fromkeys(ordered + list(issue_plan.risk_files) + list(issue_plan.relevant_files))
        )
        return merged[:8]

    def _task_state_focus_files(self, task_state_context: Optional[dict[str, Any]]) -> list[str]:
        if not isinstance(task_state_context, dict):
            return []
        values = task_state_context.get("focus_files")
        focus_files = [str(value) for value in values if value] if isinstance(values, list) else []
        contested_files = task_state_context.get("contested_files")
        if isinstance(contested_files, list):
            focus_files.extend(str(value) for value in contested_files if value)
        frontier_targets = self._task_state_frontier_targets(task_state_context)
        for target in frontier_targets:
            focus_files.extend(target.get("file_paths") or [])
        return list(dict.fromkeys(str(value) for value in focus_files if value))

    def _task_state_descriptions(
        self,
        task_state_context: Optional[dict[str, Any]],
        key: str,
        *,
        limit: int,
    ) -> list[str]:
        if not isinstance(task_state_context, dict):
            return []
        items = task_state_context.get(key)
        if not isinstance(items, list):
            return []
        descriptions: list[str] = []
        for item in items:
            if isinstance(item, dict):
                text = str(item.get("description") or item.get("summary") or "").strip()
            else:
                text = str(item or "").strip()
            if text:
                descriptions.append(text)
        return list(dict.fromkeys(descriptions))[:limit]

    def _task_state_frontier_targets(
        self,
        task_state_context: Optional[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not isinstance(task_state_context, dict):
            return []
        items = task_state_context.get("frontier_targets")
        if not isinstance(items, list):
            return []

        normalized: list[dict[str, Any]] = []
        for item in items[: max(1, self.config.planning.max_frontier_targets)]:
            if not isinstance(item, dict):
                continue
            description = str(
                item.get("description")
                or item.get("obligation_description")
                or item.get("hypothesis_description")
                or ""
            ).strip()
            if not description:
                continue
            normalized.append(
                {
                    "target_id": str(item.get("target_id") or description),
                    "kind": str(item.get("kind") or "frontier").strip().lower() or "frontier",
                    "description": description,
                    "obligation_description": str(item.get("obligation_description") or "").strip(),
                    "hypothesis_description": str(item.get("hypothesis_description") or "").strip(),
                    "rationale": str(item.get("rationale") or "").strip(),
                    "family": str(item.get("family") or "").strip(),
                    "frontier_score": (
                        float(item.get("frontier_score"))
                        if isinstance(item.get("frontier_score"), (int, float))
                        else 0.0
                    ),
                    "uncertainty_score": (
                        float(item.get("uncertainty_score"))
                        if isinstance(item.get("uncertainty_score"), (int, float))
                        else 0.0
                    ),
                    "file_paths": [
                        str(path) for path in list(item.get("file_paths") or []) if path
                    ][:6],
                    "test_ids": [
                        str(test_id) for test_id in list(item.get("test_ids") or []) if test_id
                    ][:4],
                    "symbols": [
                        str(symbol) for symbol in list(item.get("symbols") or []) if symbol
                    ][:4],
                    "obligation_id": str(item.get("obligation_id") or "").strip(),
                    "hypothesis_id": str(item.get("hypothesis_id") or "").strip(),
                }
            )
        return normalized

    def _task_state_contradiction_metrics(
        self,
        task_state_context: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(task_state_context, dict):
            return {
                "pressure": 0.0,
                "contested_files": [],
                "weakly_supported_hypothesis_count": 0,
            }

        supported = task_state_context.get("supported_hypotheses")
        items = list(supported) if isinstance(supported, list) else []
        conflict_scores: list[float] = []
        contested_files: list[str] = []
        weakly_supported = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            conflict = (
                float(item.get("conflict_score"))
                if isinstance(item.get("conflict_score"), (int, float))
                else 0.0
            )
            contradiction = (
                float(item.get("contradiction_score"))
                if isinstance(item.get("contradiction_score"), (int, float))
                else 0.0
            )
            independent_support = (
                float(item.get("independent_support_score"))
                if isinstance(item.get("independent_support_score"), (int, float))
                else 0.0
            )
            combined = max(conflict, 0.8 * contradiction)
            if combined > 0.0:
                conflict_scores.append(combined)
            if independent_support < 0.34 and (item.get("description") or item.get("summary")):
                weakly_supported += 1
            if combined >= 0.22:
                contested_files.extend(
                    str(path).strip()
                    for path in list(item.get("file_paths") or [])
                    if str(path).strip()
                )

        pressure = (
            float(task_state_context.get("contradiction_pressure"))
            if isinstance(task_state_context.get("contradiction_pressure"), (int, float))
            else 0.0
        )
        if conflict_scores:
            top = max(conflict_scores)
            average_top = sum(sorted(conflict_scores, reverse=True)[:3]) / min(
                len(conflict_scores), 3
            )
            pressure = max(pressure, min(1.0, (0.65 * top) + (0.35 * average_top)))
        return {
            "pressure": round(max(0.0, min(pressure, 1.0)), 4),
            "contested_files": _dedupe_preserve(
                list(task_state_context.get("contested_files") or []) + contested_files
            )[:8],
            "weakly_supported_hypothesis_count": weakly_supported,
        }

    @staticmethod
    def _rollout_verification_residual_test_ids(
        rollout: Any,
        *,
        limit: int = 8,
    ) -> list[str]:
        quick_verification = (
            rollout.quick_verification
            if isinstance(getattr(rollout, "quick_verification", None), dict)
            else {}
        )
        verification = (
            rollout.verification if isinstance(getattr(rollout, "verification", None), dict) else {}
        )
        test_result = (
            verification.get("test_result")
            if isinstance(verification.get("test_result"), dict)
            else {}
        )
        residual_ids: list[str] = []
        for payload in (quick_verification, test_result):
            for key in (
                "failed_tests",
                "error_tests",
                "failed_test_ids",
                "error_test_ids",
                "missing_expected_test_ids",
            ):
                values = payload.get(key)
                if isinstance(values, (list, tuple, set)):
                    residual_ids.extend(
                        str(value).strip() for value in values if str(value).strip()
                    )

        if not residual_ids:
            for key in (
                "structural_precheck_blocker",
                "structural_precheck_excerpt",
                "output_excerpt",
            ):
                residual_ids.extend(
                    _extract_residual_test_ids(
                        str(quick_verification.get(key) or ""),
                        limit=limit,
                    )
                )
        return _dedupe_preserve(residual_ids)[:limit]

    @staticmethod
    def _rollout_verification_residual_signal(
        rollout: Any,
        *,
        max_chars: int = 220,
    ) -> str:
        quick_verification = (
            rollout.quick_verification
            if isinstance(getattr(rollout, "quick_verification", None), dict)
            else {}
        )
        returncode = quick_verification.get("returncode")
        failed = quick_verification.get("failed")
        errors = quick_verification.get("errors")
        if (
            isinstance(returncode, int)
            and returncode == 0
            and int(failed or 0) == 0
            and int(errors or 0) == 0
        ):
            return ""

        classification = quick_verification.get("failure_classification")
        candidates: list[str] = []
        if isinstance(classification, dict):
            candidates.append(str(classification.get("primary_signal") or ""))
        candidates.extend(
            [
                str(quick_verification.get("structural_precheck_blocker") or ""),
                str(quick_verification.get("output_excerpt") or ""),
                str(getattr(rollout, "failure_reason", "") or ""),
            ]
        )
        for candidate in candidates:
            text = re.sub(r"\s+", " ", candidate).strip()
            if not text or text.lower() in {"returncode=0", "ok"}:
                continue
            if len(text) > max_chars:
                return text[: max(0, max_chars - 1)].rstrip() + "..."
            return text
        return ""

    def _build_reflection_memory(
        self,
        issue_plan: IssuePlan,
        rollout_results: list[Any],
    ) -> list[dict[str, Any]]:
        if not self.config.planning.enable_reflective_memory:
            return []

        aggregated: dict[tuple[str, tuple[str, ...], tuple[str, ...]], dict[str, Any]] = {}

        def string_list(payload: Any, key: str) -> list[str]:
            values = payload.get(key) if isinstance(payload, dict) else getattr(payload, key, None)
            if not isinstance(values, (list, tuple, set)):
                return []
            return [str(value).strip() for value in values if str(value).strip()]

        def remember(
            failure_type: str,
            summary: str,
            *,
            file_paths: Optional[list[str]] = None,
            symbols: Optional[list[str]] = None,
        ) -> None:
            normalized_files = tuple(_dedupe_preserve(list(file_paths or []))[:4])
            normalized_symbols = tuple(_dedupe_preserve(list(symbols or []))[:4])
            key = (failure_type, normalized_files, normalized_symbols)
            entry = aggregated.get(key)
            if entry is None:
                entry = {
                    "failure_type": failure_type,
                    "summary": summary,
                    "file_paths": list(normalized_files),
                    "symbols": list(normalized_symbols),
                    "count": 0,
                }
                aggregated[key] = entry
            entry["count"] = int(entry.get("count") or 0) + 1

        for rollout in rollout_results:
            payload = (
                rollout.multi_agent_summary
                if isinstance(getattr(rollout, "multi_agent_summary", None), dict)
                else {}
            )
            boundary_files = [
                str(path).strip()
                for path in list(payload.get("boundary_requested_files") or [])
                if str(path).strip()
            ]
            boundary_symbols = [
                str(symbol).strip()
                for symbol in list(payload.get("boundary_interface_symbols") or [])
                if str(symbol).strip()
            ]
            if int(payload.get("boundary_pressure_count") or 0) > 0:
                summary = (
                    "Recent delegated subtasks showed boundary pressure around "
                    + ", ".join(
                        (boundary_files or issue_plan.risk_files or issue_plan.relevant_files)[:3]
                    )
                    + "."
                )
                remember("bad_split", summary, file_paths=boundary_files)
            if boundary_files or boundary_symbols:
                remember(
                    "missed_interface",
                    "Recent delegated work exposed missing interface assumptions.",
                    file_paths=boundary_files,
                    symbols=boundary_symbols,
                )

            quick_verification = (
                rollout.quick_verification
                if isinstance(getattr(rollout, "quick_verification", None), dict)
                else {}
            )
            changed_files = [
                str(path).strip()
                for path in list(getattr(rollout, "changed_files", []) or [])
                if str(path).strip()
            ]
            patch_artifact = (
                rollout.patch_artifact
                if isinstance(getattr(rollout, "patch_artifact", None), dict)
                else {}
            )
            tests_run = string_list(patch_artifact, "tests_run")
            command = str(quick_verification.get("command") or "").strip()
            if command:
                tests_run.append(command)
            quick_failed = (
                isinstance(quick_verification.get("returncode"), int)
                and int(quick_verification.get("returncode")) != 0
            )
            if (
                quick_failed
                and changed_files
                and float(getattr(rollout, "progress_score", 0.0) or 0.0) < 0.15
            ):
                remember(
                    "false_localization",
                    "A recent rollout likely localized the problem poorly; verification failed with little measurable progress.",
                    file_paths=changed_files,
                )
            if changed_files and not tests_run:
                remember(
                    "verification_gap",
                    "A recent rollout edited files without attached validation evidence; close the verification gap before trusting similar changes.",
                    file_paths=changed_files,
                )

        for summary in summarize_failed_rollouts(rollout_results, limit=8):
            files = [
                str(path)
                for path in list(summary.get("files_edited") or [])
                if str(path).strip()
            ]
            root_failure = str(summary.get("root_failure") or "failed_rollout")
            hypothesis = str(summary.get("hypothesis") or "Recent rollout failed.").strip()
            remember(
                f"failed_attempt:{root_failure}",
                (
                    "Do not repeat this failed attempt without a material change: "
                    + hypothesis[:220]
                ),
                file_paths=files,
            )

        ranked = sorted(
            aggregated.values(),
            key=lambda item: (
                int(item.get("count") or 0),
                len(item.get("file_paths") or []),
                item.get("summary") or "",
            ),
            reverse=True,
        )
        return ranked[: max(1, self.config.planning.max_reflection_memory_items)]

    def _augment_task_state_with_reflection_memory(
        self,
        task_state_context: Optional[dict[str, Any]],
        *,
        issue_plan: IssuePlan,
        rollout_results: list[Any],
    ) -> dict[str, Any]:
        context = dict(task_state_context) if isinstance(task_state_context, dict) else {}
        component_ablation = component_ablation_assignment_for_task(
            config=self.config,
            issue_plan=issue_plan,
        )
        if component_disabled(component_ablation, "reflection_memory"):
            context["component_ablation"] = dict(component_ablation)
            context["reflection_memory"] = []
            return context
        reflection_memory = self._build_reflection_memory(issue_plan, rollout_results)
        if not reflection_memory:
            return context
        context["reflection_memory"] = reflection_memory
        summary = str(context.get("summary") or "").strip()
        if not summary:
            summary = (
                "Use the recent reflective failure memory to avoid repeating weak orchestrations."
            )
        elif "reflective failure memory" not in summary.lower():
            summary = (
                summary
                + " Use the recent reflective failure memory to avoid repeating weak orchestrations."
            ).strip()
        context["summary"] = summary
        return context

    def build_progress_ledger(
        self,
        issue_plan: IssuePlan,
        rollout_results: list[Any],
        *,
        task_state_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not rollout_results:
            return {}

        context = (
            dict(task_state_context)
            if isinstance(task_state_context, dict)
            else dict(issue_plan.task_state_context)
        )
        ordered = sorted(
            rollout_results,
            key=self.score_rollout_progress,
            reverse=True,
        )
        best = ordered[0]
        scores = [self.score_rollout_progress(result) for result in ordered]
        best_score = max(scores or [0.0])
        average_score = sum(scores) / len(scores) if scores else 0.0
        meaningful_rollouts = sum(1 for score in scores if score >= _MEANINGFUL_PROGRESS_SCORE)
        boundary_pressure = self._summarize_boundary_pressure(rollout_results)
        boundary_requested_files = list(boundary_pressure.get("requested_files") or [])
        boundary_interface_symbols = list(boundary_pressure.get("interface_symbols") or [])
        task_state_focus_files = self._task_state_focus_files(context)
        frontier_targets = self._task_state_frontier_targets(context)
        contradiction_metrics = self._task_state_contradiction_metrics(context)
        contradiction_pressure = float(contradiction_metrics.get("pressure") or 0.0)
        contested_files = list(contradiction_metrics.get("contested_files") or [])
        top_frontier_target = frontier_targets[0] if frontier_targets else {}
        top_plan_title = str(getattr(best, "plan_title", "") or "").strip()
        verification_residual_test_ids = _dedupe_preserve(
            [
                test_id
                for result in ordered
                for test_id in self._rollout_verification_residual_test_ids(result, limit=8)
            ]
        )[:8]
        verification_residual_signals = _dedupe_preserve(
            [
                signal
                for result in ordered
                for signal in [self._rollout_verification_residual_signal(result)]
                if signal
            ]
        )[:3]

        failure_type_counts: Counter[str] = Counter()
        for item in list(context.get("reflection_memory") or []):
            if not isinstance(item, dict):
                continue
            failure_type = str(item.get("failure_type") or "").strip().lower()
            if not failure_type:
                continue
            failure_type_counts[failure_type] += int(item.get("count") or 1)

        focus_files = list(
            dict.fromkeys(
                boundary_requested_files
                + contested_files
                + self.extract_progressive_focus_files(issue_plan, rollout_results)
                + task_state_focus_files
                + [
                    str(path).strip()
                    for path in list(top_frontier_target.get("file_paths") or [])
                    if str(path).strip()
                ]
                + list(issue_plan.risk_files)
                + list(issue_plan.relevant_files)
            )
        )[:8]
        unresolved_test_ids = list(
            dict.fromkeys(
                verification_residual_test_ids
                + [
                    str(test_id).strip()
                    for test_id in list(context.get("unresolved_test_ids") or [])
                    if str(test_id).strip()
                ]
                + list(issue_plan.test_context.failing_test_ids)
            )
        )[:4]

        next_action = "continue"
        decision_summary = "Earlier waves are producing enough partial signal to continue the current search family."
        if (
            int(boundary_pressure.get("count") or 0)
            >= self._delegation_boundary_pressure_threshold()
            and best_score < _BOUNDARY_COLLAPSE_PROGRESS_SCORE
            and (
                failure_type_counts.get("bad_split", 0) > 0
                or failure_type_counts.get("missed_interface", 0) > 0
                or contradiction_pressure >= 0.38
            )
        ):
            next_action = "collapse_to_integrator"
            decision_summary = (
                "Delegated workers repeatedly hit adjacent ownership boundaries; collapse the next wave "
                "to an integrator-owned pass across the bridge files."
            )
        elif (
            contradiction_pressure >= 0.38
            and int(contradiction_metrics.get("weakly_supported_hypothesis_count") or 0) >= 2
            and best_score < 0.55
        ):
            next_action = "collapse_to_integrator"
            decision_summary = (
                "Cross-rollout agreement is still weakly corroborated and partially conflicting; "
                "route the next wave through context-preserving integrator passes backed by direct execution."
            )
        elif (
            (best_score < _LOW_PROGRESS_SCORE and meaningful_rollouts == 0)
            or failure_type_counts.get("false_localization", 0) > 0
            or (contradiction_pressure >= 0.48 and best_score < 0.45)
        ):
            next_action = "relocalize"
            decision_summary = (
                "Early waves produced little measurable progress; restart from the highest-value frontier "
                "and re-localize before broad edits."
            )
        elif (
            int(boundary_pressure.get("count") or 0)
            >= self._delegation_boundary_pressure_threshold()
        ):
            next_action = "widen_boundaries"
            decision_summary = (
                "Partial progress exists, but delegated workers requested adjacent files; widen the owned "
                "slice around the current bridge files."
            )

        if top_plan_title:
            decision_summary = (
                f"{decision_summary} Strongest current rollout family: {top_plan_title}."
            )
        if contradiction_pressure >= 0.25 and contested_files:
            decision_summary = (
                f"{decision_summary} Conflict pressure remains high around "
                + ", ".join(contested_files[:3])
                + "."
            )
        if verification_residual_test_ids:
            shown = verification_residual_test_ids[:3]
            extra = (
                f" and {len(verification_residual_test_ids) - len(shown)} more"
                if len(verification_residual_test_ids) > len(shown)
                else ""
            )
            decision_summary = (
                f"{decision_summary} Recent verification residual tests: "
                + ", ".join(shown)
                + extra
                + "."
            )
        if verification_residual_signals:
            decision_summary = (
                f"{decision_summary} Primary residual signal: "
                f"{verification_residual_signals[0]}."
            )

        requires_context_preserving_mode = bool(
            next_action == "collapse_to_integrator"
            or (
                contradiction_pressure >= 0.38
                and int(boundary_pressure.get("count") or 0)
                >= self._delegation_boundary_pressure_threshold()
            )
        )

        return {
            "next_action": next_action,
            "decision_summary": decision_summary,
            "best_progress_score": round(best_score, 4),
            "average_progress_score": round(average_score, 4),
            "meaningful_rollout_count": meaningful_rollouts,
            "top_rollout_id": int(getattr(best, "rollout_id", -1)),
            "top_plan_title": top_plan_title,
            "focus_files": list(focus_files),
            "unresolved_test_ids": list(unresolved_test_ids),
            "frontier_target_id": str(top_frontier_target.get("target_id") or "").strip(),
            "frontier_target_description": str(
                top_frontier_target.get("description")
                or top_frontier_target.get("obligation_description")
                or ""
            ).strip(),
            "contradiction_pressure": round(contradiction_pressure, 4),
            "contested_files": list(contested_files),
            "weakly_supported_hypothesis_count": int(
                contradiction_metrics.get("weakly_supported_hypothesis_count") or 0
            ),
            "requires_context_preserving_mode": requires_context_preserving_mode,
            "boundary_pressure_count": int(boundary_pressure.get("count") or 0),
            "boundary_requested_files": list(boundary_requested_files),
            "boundary_interface_symbols": list(boundary_interface_symbols),
            "reflection_failure_types": dict(failure_type_counts),
            "verification_residual_test_ids": list(verification_residual_test_ids),
            "verification_residual_signals": list(verification_residual_signals),
        }

    def _augment_task_state_with_progress_ledger(
        self,
        task_state_context: Optional[dict[str, Any]],
        *,
        issue_plan: IssuePlan,
        rollout_results: list[Any],
    ) -> dict[str, Any]:
        context = dict(task_state_context) if isinstance(task_state_context, dict) else {}
        progress_ledger = self.build_progress_ledger(
            issue_plan,
            rollout_results,
            task_state_context=context,
        )
        if not progress_ledger:
            return context

        context["progress_ledger"] = progress_ledger
        focus_files = list(
            dict.fromkeys(
                list(progress_ledger.get("focus_files") or [])
                + list(progress_ledger.get("contested_files") or [])
                + list(context.get("focus_files") or [])
            )
        )
        if focus_files:
            context["focus_files"] = focus_files[:8]
        unresolved_test_ids = list(
            dict.fromkeys(
                list(progress_ledger.get("unresolved_test_ids") or [])
                + list(context.get("unresolved_test_ids") or [])
            )
        )
        if unresolved_test_ids:
            context["unresolved_test_ids"] = unresolved_test_ids[:4]

        summary = str(context.get("summary") or "").strip()
        decision_summary = str(progress_ledger.get("decision_summary") or "").strip()
        if decision_summary and decision_summary.lower() not in summary.lower():
            summary = f"{summary} Planner replan signal: {decision_summary}".strip()
        context["summary"] = summary
        return context

    def _summarize_boundary_pressure(
        self,
        rollout_results: list[Any],
    ) -> dict[str, Any]:
        requested_files: list[str] = []
        interface_symbols: list[str] = []
        followups: list[str] = []
        pressure_count = 0

        for rollout in rollout_results:
            payload = (
                rollout.multi_agent_summary
                if isinstance(getattr(rollout, "multi_agent_summary", None), dict)
                else {}
            )
            pressure_count += int(payload.get("boundary_pressure_count") or 0)
            requested_files.extend(
                str(path).strip()
                for path in list(payload.get("boundary_requested_files") or [])
                if str(path).strip()
            )
            interface_symbols.extend(
                str(symbol).strip()
                for symbol in list(payload.get("boundary_interface_symbols") or [])
                if str(symbol).strip()
            )
            followups.extend(
                str(item).strip()
                for item in list(payload.get("boundary_followups") or [])
                if str(item).strip()
            )

        return {
            "count": pressure_count,
            "requested_files": _dedupe_preserve(requested_files)[:8],
            "interface_symbols": _dedupe_preserve(interface_symbols)[:8],
            "followups": _dedupe_preserve(followups)[:8],
        }

    def _apply_frontier_target_to_brief(
        self,
        brief: RolloutBrief,
        target: dict[str, Any],
        *,
        stage_label: str,
    ) -> None:
        policy = self._normalize_brief_search_policy(brief)
        target_files = [str(path) for path in list(target.get("file_paths") or []) if path]
        target_tests = [str(test_id) for test_id in list(target.get("test_ids") or []) if test_id]
        target_symbols = [str(symbol) for symbol in list(target.get("symbols") or []) if symbol]
        obligation_description = str(target.get("obligation_description") or "").strip()
        hypothesis_description = str(target.get("hypothesis_description") or "").strip()

        if target_files:
            brief.focus_files = _dedupe_preserve(target_files + brief.focus_files)[:8]
        if obligation_description:
            brief.success_criteria = _dedupe_preserve(
                [obligation_description] + brief.success_criteria
            )[:6]
        if hypothesis_description:
            brief.hypotheses = _dedupe_preserve([hypothesis_description] + brief.hypotheses)[:5]
        elif target.get("kind") == "hypothesis":
            brief.hypotheses = _dedupe_preserve(
                [str(target.get("description") or "").strip()] + brief.hypotheses
            )[:5]

        rationale = str(target.get("rationale") or "").strip()

        policy.update(
            {
                "graph_target_id": str(target.get("target_id") or "").strip(),
                "graph_target_kind": str(target.get("kind") or "").strip(),
                "graph_target_stage": stage_label,
                "graph_target_description": str(target.get("description") or "").strip(),
                "graph_target_obligation_id": str(target.get("obligation_id") or "").strip(),
                "graph_target_hypothesis_id": str(target.get("hypothesis_id") or "").strip(),
                "graph_target_obligation_description": obligation_description,
                "graph_target_hypothesis_description": hypothesis_description,
                "graph_target_rationale": rationale,
                "graph_target_family": str(target.get("family") or "").strip(),
                "graph_target_score": (
                    float(target.get("frontier_score"))
                    if isinstance(target.get("frontier_score"), (int, float))
                    else 0.0
                ),
                "graph_target_uncertainty": (
                    float(target.get("uncertainty_score"))
                    if isinstance(target.get("uncertainty_score"), (int, float))
                    else 0.0
                ),
                "graph_target_file_paths": target_files[:4],
                "graph_target_test_ids": target_tests[:4],
                "graph_target_symbols": target_symbols[:4],
            }
        )
        if target_tests and policy.get("verification_focus") == "targeted_validation":
            policy["verification_focus"] = "failing_tests"
        brief.set_controller_action(policy, merge_policy=policy)

    def apply_task_state_frontier(
        self,
        issue_plan: IssuePlan,
        repo_context: Optional[RepoContext] = None,
        *,
        task_state_context: Optional[dict[str, Any]] = None,
        stage_label: str = "initial",
        frontier_targets: Optional[list[dict[str, Any]]] = None,
    ) -> IssuePlan:
        if (
            not self.config.planning.enable_task_state_graph
            or not self.config.planning.enable_frontier_targeting
        ):
            return issue_plan

        task_state_context = (
            dict(task_state_context)
            if isinstance(task_state_context, dict)
            else dict(issue_plan.task_state_context)
        )
        frontier_targets = [
            dict(target) for target in (frontier_targets or []) if isinstance(target, dict)
        ] or self._task_state_frontier_targets(task_state_context)
        if not frontier_targets or not issue_plan.rollout_briefs:
            return issue_plan

        completion_like = self._completion_like(issue_plan)
        pre_frontier_relevant_files = list(issue_plan.relevant_files)
        pre_frontier_source_priority = [
            path for path in pre_frontier_relevant_files if not self._looks_like_test_path(path)
        ]
        rollout_briefs = [
            RolloutBrief.from_dict(brief.to_dict()) for brief in issue_plan.rollout_briefs
        ]
        for index, brief in enumerate(rollout_briefs):
            target = frontier_targets[index % len(frontier_targets)]
            self._apply_frontier_target_to_brief(
                brief,
                target,
                stage_label=stage_label,
            )
            if completion_like and pre_frontier_source_priority:
                brief.focus_files = _dedupe_preserve(
                    pre_frontier_source_priority[:7] + brief.focus_files
                )[:8]

        frontier_focus_files = [
            path
            for target in frontier_targets
            for path in list(target.get("file_paths") or [])
            if path
        ]
        if stage_label in {"initial", "escalation", "localization_recovery"}:
            if completion_like:
                focus_files = _dedupe_preserve(
                    pre_frontier_relevant_files
                    + frontier_focus_files
                    + self._task_state_focus_files(task_state_context)
                    + issue_plan.risk_files
                )
            else:
                focus_files = _dedupe_preserve(
                    frontier_focus_files
                    + self._task_state_focus_files(task_state_context)
                    + issue_plan.risk_files
                    + issue_plan.relevant_files
                )
        else:
            # Preserve the plan's existing residual/progressive priority ordering and
            # use frontier files as enrichment rather than replacing that focus.
            focus_files = _dedupe_preserve(
                issue_plan.relevant_files
                + issue_plan.risk_files
                + frontier_focus_files
                + self._task_state_focus_files(task_state_context)
            )

        issue_plan.rollout_briefs = rollout_briefs
        if focus_files:
            issue_plan.relevant_files = list(focus_files)
            issue_plan.risk_files = focus_files[: min(4, len(focus_files))]
            if repo_context is not None:
                issue_plan.repo_focus_map = repo_context.build_context_pack(
                    focus_files[: self.config.planning.max_repo_map_files],
                    max_symbols_per_file=8,
                    seed_symbols=issue_plan.keywords
                    + list(issue_plan.test_context.terminal_reference_symbols or []),
                )

        issue_plan.planner_metadata = dict(issue_plan.planner_metadata)
        issue_plan.planner_metadata.update(
            {
                "frontier_targeting_enabled": True,
                "frontier_target_count": len(frontier_targets),
                "frontier_stage": stage_label,
                "frontier_target_ids": [
                    str(target.get("target_id") or "")
                    for target in frontier_targets[: self.config.planning.max_frontier_targets]
                    if target.get("target_id")
                ],
            }
        )
        self._prune_redundant_overlap_sensitive_rollout_variants(issue_plan)
        return issue_plan

    def _score_brief_for_frontier_target(
        self,
        brief: RolloutBrief,
        target: dict[str, Any],
    ) -> float:
        score = 0.0
        target_kind = str(target.get("kind") or "").strip().lower()
        target_family = str(target.get("family") or "").strip().lower()
        target_description = str(target.get("description") or "").strip().lower()
        obligation_description = str(target.get("obligation_description") or "").strip().lower()
        hypothesis_description = str(target.get("hypothesis_description") or "").strip().lower()
        target_files = {str(path) for path in list(target.get("file_paths") or []) if path}
        target_tests = {str(test_id) for test_id in list(target.get("test_ids") or []) if test_id}
        brief_text = f"{brief.title} {brief.goal} {brief.prompt_hint}".lower()
        brief_hypotheses = " ".join(brief.hypotheses).lower()
        brief_focus = {str(path) for path in brief.focus_files if path}
        policy = self._normalize_brief_search_policy(brief)
        mode = str(policy.get("mode") or "").strip().lower()

        if target_family and target_family in brief_text:
            score += 2.0
        if target_description and target_description in brief_hypotheses:
            score += 1.25
        if hypothesis_description and hypothesis_description in brief_hypotheses:
            score += 1.5
        if obligation_description and any(
            token in brief_text for token in obligation_description.split()[:4]
        ):
            score += 0.75
        overlap = len(target_files.intersection(brief_focus))
        if overlap:
            score += 1.5 + (0.6 * overlap)
        if target_tests and mode in {"test_rooted", "api_contract", "invariant_guard"}:
            score += 1.2
        if target_kind == "hypothesis" and mode in {
            "dependency_trace",
            "source_cluster",
            "surgical",
        }:
            score += 0.9
        if target_kind == "joint":
            score += 0.5
        return score

    def _rank_briefs_for_frontier_target(
        self,
        issue_plan: IssuePlan,
        target: dict[str, Any],
    ) -> list[RolloutBrief]:
        briefs = issue_plan.rollout_briefs or [
            RolloutBrief(title="Search expansion", goal="Explore frontier target.")
        ]
        ranked = sorted(
            (RolloutBrief.from_dict(brief.to_dict()) for brief in briefs),
            key=lambda brief: self._score_brief_for_frontier_target(brief, target),
            reverse=True,
        )
        unique: list[RolloutBrief] = []
        seen_allocation_arms: set[str] = set()
        for brief in ranked:
            allocation_key = _rollout_brief_allocation_key(brief)
            if allocation_key in seen_allocation_arms:
                continue
            seen_allocation_arms.add(allocation_key)
            unique.append(brief)
        return unique

    def _select_briefs_for_frontier_target(
        self,
        issue_plan: IssuePlan,
        target: dict[str, Any],
        *,
        limit: int = 1,
    ) -> list[RolloutBrief]:
        ranked = self._rank_briefs_for_frontier_target(issue_plan, target)
        return ranked[: max(1, int(limit))]

    def _select_brief_for_frontier_target(
        self,
        issue_plan: IssuePlan,
        target: dict[str, Any],
    ) -> RolloutBrief:
        return self._select_briefs_for_frontier_target(
            issue_plan,
            target,
            limit=1,
        )[0]

    def build_search_expansion_plan(
        self,
        issue_plan: IssuePlan,
        repo_context: RepoContext,
        *,
        frontier_target: dict[str, Any],
        task_state_context: Optional[dict[str, Any]] = None,
        search_depth: int = 0,
        brief_limit: int = 1,
    ) -> IssuePlan:
        expansion_plan = IssuePlan.from_dict(issue_plan.to_dict())
        expansion_plan.task_state_context = (
            dict(task_state_context)
            if isinstance(task_state_context, dict)
            else dict(issue_plan.task_state_context)
        )
        selected_briefs = self._select_briefs_for_frontier_target(
            expansion_plan,
            frontier_target,
            limit=brief_limit,
        )
        expansion_plan.rollout_briefs = list(selected_briefs)
        expansion_plan = self.apply_task_state_frontier(
            expansion_plan,
            repo_context,
            task_state_context=expansion_plan.task_state_context,
            stage_label="search_expand",
            frontier_targets=[frontier_target],
        )
        expansion_plan.planner_source = f"{issue_plan.planner_source}+frontier_search"
        expansion_plan.planner_metadata = dict(expansion_plan.planner_metadata)
        expansion_plan.planner_metadata.update(
            {
                "search_mode": self.config.search.mode.value,
                "search_depth": search_depth,
                "search_target_id": str(frontier_target.get("target_id") or ""),
                "search_target_kind": str(frontier_target.get("kind") or ""),
                "search_brief_count": len(selected_briefs),
            }
        )
        return expansion_plan

    def _disabled_delegation_policy(
        self,
        reason: str,
        *,
        existing_policy: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        policy = dict(existing_policy or {}) if isinstance(existing_policy, dict) else {}
        return {
            "enabled": False,
            "mode": "off",
            "reason": reason,
            "allowed_stages": _normalize_delegation_allowed_stages(
                list(policy.get("allowed_stages") or ["patcher"])
            ),
            "max_tasks": 0,
            "parallelism": 0,
            "max_iterations": 0,
            "subtasks": [],
            "split_confidence": float(policy.get("split_confidence") or 0.0),
            "bridge_files": list(policy.get("bridge_files") or []),
            "interface_symbols": list(policy.get("interface_symbols") or []),
            "cluster_hints": list(policy.get("cluster_hints") or []),
            "within_cluster_weight": float(policy.get("within_cluster_weight") or 0.0),
            "cross_cluster_weight": float(policy.get("cross_cluster_weight") or 0.0),
            "graph_supported": bool(policy.get("graph_supported")),
            "interface_prediction_available": bool(policy.get("interface_prediction_available")),
        }

    def _renormalize_followup_delegation_policy(
        self,
        issue_plan: IssuePlan,
        brief: RolloutBrief,
        repo_context: RepoContext,
    ) -> None:
        existing_policy = (
            dict(brief.delegation_policy) if isinstance(brief.delegation_policy, dict) else {}
        )
        seed_policy: dict[str, Any] = {}
        if existing_policy.get("allowed_stages"):
            seed_policy["allowed_stages"] = list(existing_policy.get("allowed_stages") or [])
        brief.delegation_policy = seed_policy
        brief.delegation_policy = self._normalize_brief_delegation_policy(
            issue_plan,
            brief,
            repo_context,
        )

    def _apply_progress_ledger_guidance_to_brief(
        self,
        issue_plan: IssuePlan,
        brief: RolloutBrief,
        repo_context: RepoContext,
        progress_ledger: dict[str, Any],
        *,
        force_disable_delegation: bool = False,
        disabled_reason: str = "",
    ) -> None:
        action = str(progress_ledger.get("next_action") or "").strip().lower()
        decision_summary = str(progress_ledger.get("decision_summary") or "").strip()
        focus_files = [
            str(path).strip()
            for path in list(progress_ledger.get("focus_files") or [])
            if str(path).strip()
        ]
        boundary_requested_files = [
            str(path).strip()
            for path in list(progress_ledger.get("boundary_requested_files") or [])
            if str(path).strip()
        ]
        contested_files = [
            str(path).strip()
            for path in list(progress_ledger.get("contested_files") or [])
            if str(path).strip()
        ]
        if action in {"widen_boundaries", "collapse_to_integrator"} and boundary_requested_files:
            brief.focus_files = _dedupe_preserve(
                boundary_requested_files + contested_files + brief.focus_files
            )[:8]
        elif action == "relocalize" and focus_files:
            brief.focus_files = _dedupe_preserve(contested_files + brief.focus_files + focus_files)[
                :8
            ]
        elif focus_files:
            brief.focus_files = _dedupe_preserve(contested_files + focus_files + brief.focus_files)[
                :8
            ]
        if decision_summary and decision_summary.lower() not in brief.prompt_hint.lower():
            brief.prompt_hint = f"{brief.prompt_hint} {decision_summary}".strip()

        if (
            force_disable_delegation
            or action == "collapse_to_integrator"
            or bool(progress_ledger.get("requires_context_preserving_mode"))
        ):
            brief.delegation_policy = self._disabled_delegation_policy(
                disabled_reason or action or "planner_directive",
                existing_policy=brief.delegation_policy
                if isinstance(brief.delegation_policy, dict)
                else {},
            )
            return
        self._renormalize_followup_delegation_policy(issue_plan, brief, repo_context)

    def build_progressive_wave_plan(
        self,
        source_issue_plan: IssuePlan,
        repo_context: RepoContext,
        rollout_results: list[Any],
        *,
        additional_rollouts: int,
        progressive_summary: str,
        progressive_focus_files: Optional[list[str]] = None,
        task_state_context: Optional[dict[str, Any]] = None,
    ) -> IssuePlan:
        """Reallocate remaining rollout budget toward the strongest observed families."""
        if additional_rollouts <= 0:
            return source_issue_plan

        task_state_context = (
            dict(task_state_context)
            if isinstance(task_state_context, dict)
            else dict(source_issue_plan.task_state_context)
        )
        task_state_context = self._augment_task_state_with_reflection_memory(
            task_state_context,
            issue_plan=source_issue_plan,
            rollout_results=rollout_results,
        )
        task_state_context = self._augment_task_state_with_progress_ledger(
            task_state_context,
            issue_plan=source_issue_plan,
            rollout_results=rollout_results,
        )
        progress_ledger = (
            dict(task_state_context.get("progress_ledger") or {})
            if isinstance(task_state_context, dict)
            else {}
        )
        boundary_pressure = self._summarize_boundary_pressure(rollout_results)
        task_state_focus_files = self._task_state_focus_files(task_state_context)
        open_obligations = self._task_state_descriptions(
            task_state_context,
            "open_obligations",
            limit=3,
        )
        supported_hypotheses = self._task_state_descriptions(
            task_state_context,
            "supported_hypotheses",
            limit=3,
        )
        task_state_summary = str(task_state_context.get("summary") or "").strip()
        progressive_focus_files = list(
            dict.fromkeys(
                (
                    list(progressive_focus_files)
                    if progressive_focus_files is not None
                    else self.extract_progressive_focus_files(source_issue_plan, rollout_results)
                )
                + list(boundary_pressure.get("requested_files") or [])
                + task_state_focus_files
                + list(progress_ledger.get("focus_files") or [])
            )
        )
        if (
            int(boundary_pressure.get("count") or 0)
            >= self._delegation_boundary_pressure_threshold()
        ):
            boundary_summary = (
                "Recent delegated work hit file-boundary pressure around "
                + ", ".join(
                    list(boundary_pressure.get("requested_files") or source_issue_plan.risk_files)[
                        :3
                    ]
                )
                + "."
            )
            progressive_summary = f"{progressive_summary} {boundary_summary}".strip()
        decision_summary = str(progress_ledger.get("decision_summary") or "").strip()
        if decision_summary:
            progressive_summary = f"{progressive_summary} {decision_summary}".strip()
        rollout_briefs = self.allocator.allocate_followup_briefs(
            source_issue_plan.rollout_briefs,
            rollout_results,
            rollout_count=additional_rollouts,
        )
        if not rollout_briefs:
            return source_issue_plan

        focus_files = list(
            dict.fromkeys(
                progressive_focus_files
                + task_state_focus_files
                + list(progress_ledger.get("focus_files") or [])
                + source_issue_plan.risk_files
                + source_issue_plan.relevant_files
            )
        )
        for brief in rollout_briefs:
            brief.focus_files = list(
                dict.fromkeys(
                    task_state_focus_files
                    + progressive_focus_files
                    + brief.focus_files
                    + source_issue_plan.risk_files
                )
            )[:8]
            brief.hypotheses = list(
                dict.fromkeys(
                    supported_hypotheses
                    + ["Earlier waves produced partial progress in this search family."]
                    + brief.hypotheses
                )
            )[:5]
            if open_obligations:
                brief.success_criteria = list(
                    dict.fromkeys(
                        open_obligations[:2]
                        + brief.success_criteria
                        + source_issue_plan.success_criteria
                    )
                )[:6]
            if progressive_summary:
                brief.prompt_hint = f"{brief.prompt_hint} {progressive_summary}".strip()
            if task_state_summary:
                brief.prompt_hint = f"{brief.prompt_hint} {task_state_summary}".strip()
            self._apply_progress_ledger_guidance_to_brief(
                source_issue_plan,
                brief,
                repo_context,
                progress_ledger,
            )

        focus_map = repo_context.build_context_pack(
            focus_files[: self.config.planning.max_repo_map_files],
            max_symbols_per_file=8,
            seed_symbols=source_issue_plan.keywords
            + list(source_issue_plan.test_context.terminal_reference_symbols or []),
        )
        test_context = TestContext.from_dict(source_issue_plan.test_context.to_dict())
        summary = progressive_summary.strip()
        if task_state_summary:
            summary = f"{summary} Task-state summary: {task_state_summary}".strip()
        if summary:
            base_summary = test_context.summary.strip()
            test_context.summary = (
                f"{base_summary} Progressive wave objective: {summary}".strip()
                if base_summary
                else f"Progressive wave objective: {summary}"
            )

        planner_metadata = dict(source_issue_plan.planner_metadata)
        planner_metadata.update(
            {
                "progressive_allocation": "profile_aware_ucb_reallocate",
                "progressive_focus_files": list(progressive_focus_files[:6]),
                "progressive_source_rollouts": len(rollout_results),
                "progressive_wave_rollouts": additional_rollouts,
                "task_state_graph_enabled": bool(task_state_context),
                "task_state_open_obligation_count": len(
                    task_state_context.get("open_obligations") or []
                ),
                "task_state_supported_hypothesis_count": len(
                    task_state_context.get("supported_hypotheses") or []
                ),
                "task_state_reflection_memory_count": len(
                    task_state_context.get("reflection_memory") or []
                ),
                "progress_ledger_action": str(progress_ledger.get("next_action") or "").strip(),
                "progress_ledger_summary": str(
                    progress_ledger.get("decision_summary") or ""
                ).strip(),
                "boundary_pressure_count": int(boundary_pressure.get("count") or 0),
                "boundary_requested_files": list(boundary_pressure.get("requested_files") or []),
                "boundary_interface_symbols": list(
                    boundary_pressure.get("interface_symbols") or []
                ),
            }
        )

        next_plan = IssuePlan(
            summary=source_issue_plan.summary,
            keywords=list(source_issue_plan.keywords),
            relevant_files=focus_files,
            risk_files=focus_files[: min(4, len(focus_files))]
            or list(source_issue_plan.risk_files),
            success_criteria=list(source_issue_plan.success_criteria),
            rollout_briefs=rollout_briefs,
            repo_focus_map=focus_map,
            planner_source=f"{source_issue_plan.planner_source}+progressive_ucb",
            planner_tokens=source_issue_plan.planner_tokens,
            difficulty_estimate=source_issue_plan.difficulty_estimate,
            recommended_rollouts=source_issue_plan.recommended_rollouts,
            orchestration_primitives=list(source_issue_plan.orchestration_primitives),
            allocator_features=dict(source_issue_plan.allocator_features),
            unsolvable_reason=source_issue_plan.unsolvable_reason,
            test_context=test_context,
            task_state_context=dict(task_state_context),
            planner_metadata=planner_metadata,
        )
        return self.apply_task_state_frontier(
            next_plan,
            repo_context,
            task_state_context=task_state_context,
            stage_label="progressive_wave",
        )

    def build_localization_recovery_plan(
        self,
        source_issue_plan: IssuePlan,
        repo_context: RepoContext,
        rollout_results: list[Any],
        *,
        additional_rollouts: int,
        recovery_summary: str,
        task_state_context: Optional[dict[str, Any]] = None,
    ) -> IssuePlan:
        if additional_rollouts <= 0:
            return source_issue_plan

        task_state_context = (
            dict(task_state_context)
            if isinstance(task_state_context, dict)
            else dict(source_issue_plan.task_state_context)
        )
        task_state_context = self._augment_task_state_with_reflection_memory(
            task_state_context,
            issue_plan=source_issue_plan,
            rollout_results=rollout_results,
        )
        task_state_context = self._augment_task_state_with_progress_ledger(
            task_state_context,
            issue_plan=source_issue_plan,
            rollout_results=rollout_results,
        )
        progress_ledger = (
            dict(task_state_context.get("progress_ledger") or {})
            if isinstance(task_state_context, dict)
            else {}
        )
        task_state_focus_files = self._task_state_focus_files(task_state_context)
        open_obligations = self._task_state_descriptions(
            task_state_context,
            "open_obligations",
            limit=3,
        )
        supported_hypotheses = self._task_state_descriptions(
            task_state_context,
            "supported_hypotheses",
            limit=3,
        )
        task_state_summary = str(task_state_context.get("summary") or "").strip()
        decision_summary = str(progress_ledger.get("decision_summary") or "").strip()
        frontier_description = str(progress_ledger.get("frontier_target_description") or "").strip()
        if decision_summary:
            recovery_summary = f"{recovery_summary} {decision_summary}".strip()
        if frontier_description:
            recovery_summary = (
                f"{recovery_summary} Target the frontier obligation: {frontier_description}."
            ).strip()

        rollout_briefs = self.allocator.allocate_followup_briefs(
            source_issue_plan.rollout_briefs,
            rollout_results,
            rollout_count=additional_rollouts,
        )
        if not rollout_briefs:
            return source_issue_plan

        focus_files = list(
            dict.fromkeys(
                task_state_focus_files
                + list(progress_ledger.get("focus_files") or [])
                + source_issue_plan.risk_files
                + source_issue_plan.relevant_files
            )
        )
        recovery_success_criteria = list(
            dict.fromkeys(
                open_obligations[:2]
                + [
                    "Re-localize the root cause before broad edits.",
                    "Use targeted validation to prove the new localization before widening scope.",
                ]
                + source_issue_plan.success_criteria
            )
        )[:6]
        for brief in rollout_briefs:
            brief.agent_mode = AgentMode.SCAFFOLDED
            brief.focus_files = list(dict.fromkeys(focus_files + brief.focus_files))[:8]
            brief.hypotheses = list(
                dict.fromkeys(
                    supported_hypotheses
                    + [
                        "Earlier waves likely localized the bug poorly; re-derive the root-cause path before editing broadly."
                    ]
                    + brief.hypotheses
                )
            )[:5]
            brief.success_criteria = recovery_success_criteria
            if recovery_summary:
                brief.prompt_hint = f"{brief.prompt_hint} {recovery_summary}".strip()
            if task_state_summary:
                brief.prompt_hint = f"{brief.prompt_hint} {task_state_summary}".strip()
            if not brief.title.lower().startswith("localization recovery:"):
                brief.title = f"Localization Recovery: {brief.title}"
            search_policy = self._normalize_brief_search_policy(brief)
            if len(brief.focus_files) >= 2 or progress_ledger.get("boundary_interface_symbols"):
                search_policy["mode"] = "dependency_trace"
            else:
                search_policy["mode"] = "surgical"
            search_policy["verification_focus"] = "focus_test_files"
            brief.set_controller_action(search_policy, merge_policy=search_policy)
            self._apply_progress_ledger_guidance_to_brief(
                source_issue_plan,
                brief,
                repo_context,
                progress_ledger,
                force_disable_delegation=True,
                disabled_reason="localization_recovery",
            )

        focus_map = repo_context.build_context_pack(
            focus_files[: self.config.planning.max_repo_map_files],
            max_symbols_per_file=8,
            seed_symbols=source_issue_plan.keywords
            + list(source_issue_plan.test_context.terminal_reference_symbols or []),
        )
        test_context = TestContext.from_dict(source_issue_plan.test_context.to_dict())
        residual_text = recovery_summary.strip()
        if task_state_summary:
            residual_text = f"{residual_text} Task-state summary: {task_state_summary}".strip()
        if residual_text:
            base_summary = test_context.summary.strip()
            test_context.summary = (
                f"{base_summary} Localization recovery objective: {residual_text}".strip()
                if base_summary
                else f"Localization recovery objective: {residual_text}"
            )

        planner_metadata = dict(source_issue_plan.planner_metadata)
        planner_metadata.update(
            {
                "followup_allocation": "localization_recovery",
                "followup_rollouts": additional_rollouts,
                "followup_focus_files": list(focus_files[:6]),
                "followup_source_rollouts": len(rollout_results),
                "task_state_graph_enabled": bool(task_state_context),
                "task_state_open_obligation_count": len(
                    task_state_context.get("open_obligations") or []
                ),
                "task_state_supported_hypothesis_count": len(
                    task_state_context.get("supported_hypotheses") or []
                ),
                "task_state_reflection_memory_count": len(
                    task_state_context.get("reflection_memory") or []
                ),
                "progress_ledger_action": str(progress_ledger.get("next_action") or "").strip(),
                "progress_ledger_summary": str(
                    progress_ledger.get("decision_summary") or ""
                ).strip(),
                "boundary_pressure_count": int(progress_ledger.get("boundary_pressure_count") or 0),
                "boundary_requested_files": list(
                    progress_ledger.get("boundary_requested_files") or []
                ),
                "boundary_interface_symbols": list(
                    progress_ledger.get("boundary_interface_symbols") or []
                ),
                "recovery_mode": "localization_first",
            }
        )

        recovery_plan = IssuePlan(
            summary=source_issue_plan.summary,
            keywords=list(source_issue_plan.keywords),
            relevant_files=focus_files,
            risk_files=focus_files[: min(4, len(focus_files))]
            or list(source_issue_plan.risk_files),
            success_criteria=recovery_success_criteria,
            rollout_briefs=rollout_briefs,
            repo_focus_map=focus_map,
            planner_source=f"{source_issue_plan.planner_source}+localization_recovery",
            planner_tokens=source_issue_plan.planner_tokens,
            difficulty_estimate=source_issue_plan.difficulty_estimate,
            recommended_rollouts=additional_rollouts,
            orchestration_primitives=list(source_issue_plan.orchestration_primitives),
            allocator_features=dict(source_issue_plan.allocator_features),
            unsolvable_reason=source_issue_plan.unsolvable_reason,
            test_context=test_context,
            task_state_context=dict(task_state_context),
            planner_metadata=planner_metadata,
        )
        return self.apply_task_state_frontier(
            recovery_plan,
            repo_context,
            task_state_context=task_state_context,
            stage_label="localization_recovery",
        )

    def _direct_import_focus_files(
        self,
        repo_context: RepoContext,
        seed_files: list[str],
        *,
        max_files: int,
    ) -> list[str]:
        module_aliases: dict[str, str] = {}
        for file_info in repo_context.files:
            path = str(file_info.path or "")
            if not path or self._looks_like_test_path(path) or file_info.language != "python":
                continue
            path_obj = Path(path)
            if not path_obj.name:
                continue
            module_parts = list(path_obj.with_suffix("").parts)
            if not module_parts:
                continue
            if module_parts[-1] == "__init__":
                module_parts = module_parts[:-1]
            if not module_parts:
                continue
            full_module = ".".join(module_parts)
            module_aliases.setdefault(full_module, path)
            if len(module_parts) == 1:
                # Top-level modules can be imported by basename. Nested modules
                # cannot: external imports such as `hypothesis` must not resolve
                # to `web3/_utils/hypothesis.py`.
                module_aliases.setdefault(module_parts[0], path)

        def resolve_import(import_name: str) -> Optional[str]:
            parts = [part for part in str(import_name or "").strip().rstrip(".*").split(".") if part]
            for width in range(len(parts), 0, -1):
                candidate = ".".join(parts[:width])
                path = module_aliases.get(candidate)
                if path:
                    return path
            return None

        resolved: list[str] = []
        for seed in seed_files:
            file_info = repo_context.get_file_info(seed)
            if file_info is None:
                continue
            for imported in list(file_info.imports or []):
                path = resolve_import(imported)
                if path and path != seed and not self._looks_like_test_path(path):
                    resolved.append(path)
            if len(_dedupe_preserve(resolved)) >= max_files:
                break
        return _dedupe_preserve(resolved)[:max_files]

    def _external_import_roots_for_seed_files(
        self,
        repo_context: RepoContext,
        seed_files: list[str],
    ) -> set[str]:
        local_modules: set[str] = set()
        for file_info in repo_context.files:
            path = str(file_info.path or "")
            if not path or self._looks_like_test_path(path) or file_info.language != "python":
                continue
            path_obj = Path(path)
            if not path_obj.name:
                continue
            module_parts = list(path_obj.with_suffix("").parts)
            if module_parts and module_parts[-1] == "__init__":
                module_parts = module_parts[:-1]
            if module_parts:
                local_modules.add(".".join(module_parts))
                if len(module_parts) == 1:
                    local_modules.add(module_parts[0])

        external_roots: set[str] = set()
        for seed in seed_files:
            file_info = repo_context.get_file_info(seed)
            if file_info is None:
                continue
            for imported in list(file_info.imports or []):
                parts = [part for part in str(imported or "").strip().rstrip(".*").split(".") if part]
                if not parts:
                    continue
                locally_resolved = any(
                    ".".join(parts[:width]) in local_modules
                    for width in range(len(parts), 0, -1)
                )
                if not locally_resolved:
                    external_roots.add(parts[0].lower())
        return external_roots

    def _residual_followup_priority_files(
        self,
        repo_context: RepoContext,
        *,
        residual_summary: str,
        residual_focus_files: list[str],
    ) -> list[str]:
        residual_text = str(residual_summary or "")
        residual_test_ids = _extract_residual_test_ids(residual_text)
        if not residual_test_ids:
            return []
        residual_test_files = [
            path
            for test_id in residual_test_ids
            for path in [_test_file_from_residual_test_id(test_id)]
            if path and repo_context.get_file_info(path) is not None
        ]
        if not residual_test_files:
            return []

        keywords = _extract_residual_keyword_hints(residual_text, residual_test_ids)
        relevant = [
            path
            for path in repo_context.get_relevant_files(
                keywords + residual_test_files,
                max_files=16,
            )
            if path and repo_context.get_file_info(path) is not None
        ]
        direct_imports = self._direct_import_focus_files(
            repo_context,
            residual_test_files,
            max_files=12,
        )
        external_import_roots = self._external_import_roots_for_seed_files(
            repo_context,
            residual_test_files,
        )
        dependency_neighbors = [
            path
            for path in repo_context.get_dependency_neighbors(residual_test_files, max_neighbors=12)
            if path and repo_context.get_file_info(path) is not None
        ]
        text_paths = [
            path
            for path in _extract_residual_paths_from_text(residual_text)
            if path and repo_context.get_file_info(path) is not None
        ]

        def _not_external_import_collision(path: str) -> bool:
            if not external_import_roots:
                return True
            path_obj = Path(path)
            stem = path_obj.stem.lower()
            if stem in external_import_roots:
                return False
            return not any(part.lower() in external_import_roots for part in path_obj.parts[:-1])

        source_priority = [
            path
            for path in (
                direct_imports
                + [
                    path
                    for path in relevant
                    if not self._looks_like_test_path(path)
                    and _not_external_import_collision(path)
                ]
                + [
                    path
                    for path in dependency_neighbors
                    if not self._looks_like_test_path(path)
                    and _not_external_import_collision(path)
                ]
                + [
                    path
                    for path in text_paths
                    if not self._looks_like_test_path(path)
                    and _not_external_import_collision(path)
                ]
            )
            if path not in set(residual_focus_files)
            or path in direct_imports
            or path in relevant
        ]
        return _dedupe_preserve(
            source_priority[:10]
            + residual_test_files[:4]
            + [path for path in text_paths if self._looks_like_test_path(path)][:4]
        )[:14]

    def build_followup_plan(
        self,
        issue_plan: IssuePlan,
        repo_context: RepoContext,
        rollout_results: list[Any],
        *,
        additional_rollouts: int,
        residual_summary: str,
        residual_focus_files: Optional[list[str]] = None,
        task_state_context: Optional[dict[str, Any]] = None,
    ) -> IssuePlan:
        if additional_rollouts <= 0:
            return issue_plan

        task_state_context = (
            dict(task_state_context)
            if isinstance(task_state_context, dict)
            else dict(issue_plan.task_state_context)
        )
        residual_focus_files = _dedupe_preserve(list(residual_focus_files or []))
        residual_priority_files = self._residual_followup_priority_files(
            repo_context,
            residual_summary=residual_summary,
            residual_focus_files=residual_focus_files,
        )
        if residual_priority_files:
            residual_focus_files = _dedupe_preserve(
                residual_priority_files + residual_focus_files
            )
        task_state_context = _recenter_task_state_context_on_residual(
            task_state_context,
            residual_summary=residual_summary,
            residual_focus_files=residual_focus_files,
        )
        task_state_context = self._augment_task_state_with_reflection_memory(
            task_state_context,
            issue_plan=issue_plan,
            rollout_results=rollout_results,
        )
        task_state_context = self._augment_task_state_with_progress_ledger(
            task_state_context,
            issue_plan=issue_plan,
            rollout_results=rollout_results,
        )
        progress_ledger = (
            dict(task_state_context.get("progress_ledger") or {})
            if isinstance(task_state_context, dict)
            else {}
        )
        boundary_pressure = self._summarize_boundary_pressure(rollout_results)
        task_state_focus_files = self._task_state_focus_files(task_state_context)
        open_obligations = self._task_state_descriptions(
            task_state_context,
            "open_obligations",
            limit=3,
        )
        supported_hypotheses = self._task_state_descriptions(
            task_state_context,
            "supported_hypotheses",
            limit=3,
        )
        task_state_summary = str(task_state_context.get("summary") or "").strip()
        residual_focus_files = list(dict.fromkeys(residual_focus_files or []))
        diagnostic_locations: list[dict[str, Any]] = []
        verifier_repair_objective = ""
        verifier_diagnostic_source_context = ""
        verifier_repair_test_ids: list[str] = []
        residual_summary_text = str(residual_summary or "")
        validity_repair_followup = bool(
            re.search(
                r"\bVerifier (?:lint|static validity|prune|coverage) rejection\b",
                residual_summary_text,
                flags=re.IGNORECASE,
            )
            or re.search(
                r"\bverifier/validity-rejected files\b",
                residual_summary_text,
                flags=re.IGNORECASE,
            )
            or "Unimplemented function bodies still in the candidate patch:" in residual_summary_text
            or "Public symbols present in the baseline but missing from the candidate:"
            in residual_summary_text
        )
        if validity_repair_followup:
            # Verifier rejections are hard validity evidence; advisory frontier
            # memory must not displace the concrete diagnostics to repair.
            diagnostic_locations = _extract_verifier_diagnostic_locations(residual_summary)
            verifier_repair_objective = _verifier_repair_objective_text(
                residual_summary,
                diagnostic_locations,
            )
            verifier_diagnostic_source_context = _extract_verifier_diagnostic_source_context(
                residual_summary
            )
            diagnostic_focus_files = _verifier_repair_source_focus_paths(
                _extract_verifier_validity_focus_paths(residual_summary)
            )
            if diagnostic_focus_files:
                residual_focus_files = diagnostic_focus_files
            else:
                # Verifier-repair edit focus must stay source-only. Coverage
                # residuals can name missing test IDs/files as evidence, but
                # those are non-editable diagnostics, not action files.
                residual_focus_files = _verifier_repair_source_focus_paths(
                    residual_focus_files + issue_plan.risk_files + issue_plan.relevant_files
                )
            additional_validity_diagnostics = _extract_additional_validity_residual_text(
                residual_summary
            )
            verifier_repair_test_ids = [
                test_id
                for test_id in _extract_residual_test_ids(residual_summary, limit=24)
                if self._looks_like_test_path(_test_file_from_residual_test_id(test_id))
            ][:12]
            task_state_focus_files = []
            task_state_summary = ""
            open_obligations = []
            supported_hypotheses = []
            task_state_context = {
                "summary": "",
                "focus_files": list(residual_focus_files),
                "open_obligations": [],
                "supported_hypotheses": [],
                "reflection_memory": [],
                "progress_ledger": {},
                "verifier_validity_repair": True,
            }
            progress_ledger = {}
        if boundary_pressure.get("requested_files") and not validity_repair_followup:
            residual_focus_files = list(
                dict.fromkeys(
                    list(boundary_pressure.get("requested_files") or []) + residual_focus_files
                )
            )
        if progress_ledger.get("focus_files") and not validity_repair_followup:
            residual_focus_files = list(
                dict.fromkeys(residual_focus_files + list(progress_ledger.get("focus_files") or []))
            )
        if (
            int(boundary_pressure.get("count") or 0)
            >= self._delegation_boundary_pressure_threshold()
            and not validity_repair_followup
        ):
            boundary_summary = (
                "Recent delegated work exposed boundary pressure around "
                + ", ".join(
                    list(boundary_pressure.get("requested_files") or issue_plan.risk_files)[:3]
                )
                + "."
            )
            residual_summary = f"{residual_summary} {boundary_summary}".strip()
        decision_summary = str(progress_ledger.get("decision_summary") or "").strip()
        if decision_summary and not validity_repair_followup:
            residual_summary = f"{residual_summary} {decision_summary}".strip()
        rollout_briefs = self.allocator.allocate_followup_briefs(
            issue_plan.rollout_briefs,
            rollout_results,
            rollout_count=additional_rollouts,
        )
        if not rollout_briefs:
            return issue_plan

        if validity_repair_followup and residual_focus_files:
            focus_files = list(dict.fromkeys(residual_focus_files))
            followup_success_criteria = [
                "Repair the verifier validity diagnostics named in the residual objective.",
                "Submit a new edit in at least one verifier-rejected file; no-op repairs and generated harness-only diffs do not satisfy this follow-up.",
                "When diagnostics name concrete source lines, keep the repair adjacent to at least one listed diagnostic line.",
                "Do not edit generated harness helpers, expected-test-id inventories, or tests for this verifier repair.",
                "Preserve the already passing verification behavior.",
            ]
        else:
            focus_files = list(
                dict.fromkeys(
                    residual_focus_files
                    + task_state_focus_files
                    + list(progress_ledger.get("focus_files") or [])
                    + issue_plan.risk_files
                    + issue_plan.relevant_files
                )
            )
            followup_success_criteria = list(
                dict.fromkeys(
                    issue_plan.success_criteria
                    + open_obligations[:2]
                    + ["Resolve the remaining regressions from earlier candidate patches."]
                )
            )
        for brief in rollout_briefs:
            if validity_repair_followup and residual_focus_files:
                brief.focus_files = _dedupe_preserve(residual_focus_files)[:12]
                brief.hypotheses = [
                    "A previous candidate passed behavioral verification but failed hard verifier validity diagnostics.",
                    "The concrete verifier diagnostics in the residual objective are authoritative for this follow-up.",
                ]
            else:
                brief.focus_files = list(
                    dict.fromkeys(residual_focus_files + task_state_focus_files + brief.focus_files)
                )[:8]
                brief.hypotheses = list(
                    dict.fromkeys(
                        supported_hypotheses
                        + [
                            "A previous rollout found part of the fix, but a residual regression remains."
                        ]
                        + brief.hypotheses
                    )
                )[:5]
            brief.success_criteria = followup_success_criteria
            if validity_repair_followup:
                verifier_file_list = (
                    ", ".join(residual_focus_files[:12])
                    if residual_focus_files
                    else "the verifier-rejected files"
                )
                verification_focus = str(
                    dict(brief.search_policy or {}).get("verification_focus")
                    or "gold_expected_suite"
                )
                repair_search_policy = {
                    "mode": "surgical",
                    "origin": "verifier_validity_repair",
                    "verification_focus": verification_focus,
                    "action_file_paths": list(residual_focus_files),
                    "verifier_diagnostic_locations": list(diagnostic_locations),
                    "verifier_repair_objective": verifier_repair_objective,
                    "verifier_diagnostic_source_context": verifier_diagnostic_source_context,
                    "additional_validity_diagnostics": additional_validity_diagnostics,
                    "verifier_validity_repair": True,
                    "disable_strategy_prefix": True,
                    "cli_agent_use_masai_preround": "off",
                }
                if verifier_repair_test_ids:
                    repair_search_policy["action_test_ids"] = list(verifier_repair_test_ids)
                    repair_search_policy["graph_target_test_ids"] = list(
                        verifier_repair_test_ids
                    )
                brief.search_policy = repair_search_policy
                brief.set_controller_action(
                    ControllerAction(
                        kind="rollout_brief",
                        mode="surgical",
                        origin="verifier_validity_repair",
                        verification_focus=verification_focus,
                        file_paths=list(residual_focus_files),
                        test_ids=list(verifier_repair_test_ids),
                    ),
                    merge_policy=repair_search_policy,
                )
                brief.title = "Follow-up: Verifier validity repair"
                brief.agent_mode = AgentMode.CLI_AGENT
                brief.goal = (
                    "Make the smallest source edit needed in "
                    f"{verifier_file_list} to clear the hard verifier validity diagnostics "
                    "while preserving the already passing behavior."
                )
                prompt_parts = [
                    (
                        "Repair only the verifier validity diagnostics named below. "
                        "Start with the listed verifier files, preserve the current "
                        "passing verification behavior, and do not revisit earlier "
                        "baseline traceback or advisory frontier files unless the "
                        "verifier output changes. Your submitted patch must include "
                        "a new edit to at least one verifier-rejected file listed here; "
                        "when a diagnostic names a line, make at least one hunk adjacent "
                        "to a listed diagnostic line; "
                        "a no-op, a test rerun, or edits only to generated harness or "
                        "inventory files are invalid even if behavioral tests pass."
                    )
                ]
            else:
                prompt_parts = [_strip_residual_followup_text(brief.prompt_hint)]
            if validity_repair_followup:
                if verifier_repair_objective:
                    prompt_parts.append(
                        f"Verifier validity objective: {verifier_repair_objective}"
                    )
                if additional_validity_diagnostics:
                    prompt_parts.append(
                        "Additional hard validity diagnostics: "
                        + additional_validity_diagnostics
                    )
                if verifier_diagnostic_source_context:
                    prompt_parts.append(verifier_diagnostic_source_context)
            elif residual_summary:
                prompt_parts.append(f"Residual follow-up objective: {residual_summary}")
            if task_state_summary and not validity_repair_followup:
                prompt_parts.append(f"Task-state summary: {task_state_summary}")
            brief.prompt_hint = " ".join(part for part in prompt_parts if part).strip()
            if not brief.title.lower().startswith("follow-up:"):
                brief.title = f"Follow-up: {brief.title}"
            if not validity_repair_followup:
                self._apply_progress_ledger_guidance_to_brief(
                    issue_plan,
                    brief,
                    repo_context,
                    progress_ledger,
                )
            if validity_repair_followup and residual_focus_files:
                brief.focus_files = _dedupe_preserve(residual_focus_files)[:12]
                brief.delegation_policy = self._disabled_delegation_policy(
                    "verifier_validity_repair",
                    existing_policy=brief.delegation_policy
                    if isinstance(brief.delegation_policy, dict)
                    else {},
                )

        focus_map = repo_context.build_context_pack(
            focus_files[: self.config.planning.max_repo_map_files],
            max_symbols_per_file=8,
            seed_symbols=issue_plan.keywords
            + list(issue_plan.test_context.terminal_reference_symbols or []),
        )
        test_context = TestContext.from_dict(issue_plan.test_context.to_dict())
        residual_text = (
            verifier_repair_objective.strip()
            if validity_repair_followup
            else residual_summary.strip()
        )
        if task_state_summary and not validity_repair_followup:
            residual_text = f"{residual_text} Task-state summary: {task_state_summary}".strip()
        if residual_text:
            base_summary = _strip_residual_followup_text(test_context.summary)
            test_context.summary = (
                f"{base_summary} Residual follow-up objective: {residual_text}".strip()
                if base_summary
                else f"Residual follow-up objective: {residual_text}"
            )
        if validity_repair_followup:
            test_context.summary = (
                f"Residual follow-up objective: {residual_text}"
                if residual_text
                else "Residual follow-up objective: repair hard verifier validity diagnostics."
            )
            test_context.focus_test_files = []
            test_context.incomplete_test_files = []
            test_context.source_focus_files = list(focus_files)
            test_context.incomplete_source_files = []
            test_context.terminal_source_files = []
            test_context.terminal_reference_symbols = []
            test_context.exception_summaries = []
            test_context.failing_test_ids = []

        planner_metadata = dict(issue_plan.planner_metadata)
        planner_metadata.update(
            {
                "followup_allocation": "profile_aware_ucb_reallocate",
                "followup_rollouts": additional_rollouts,
                "followup_focus_files": list(focus_files[:12]),
                "followup_source_rollouts": len(rollout_results),
                "task_state_graph_enabled": bool(task_state_context),
                "task_state_open_obligation_count": len(
                    task_state_context.get("open_obligations") or []
                ),
                "task_state_supported_hypothesis_count": len(
                    task_state_context.get("supported_hypotheses") or []
                ),
                "task_state_reflection_memory_count": len(
                    task_state_context.get("reflection_memory") or []
                ),
                "progress_ledger_action": str(progress_ledger.get("next_action") or "").strip(),
                "progress_ledger_summary": str(
                    progress_ledger.get("decision_summary") or ""
                ).strip(),
                "boundary_pressure_count": int(boundary_pressure.get("count") or 0),
                "boundary_requested_files": list(boundary_pressure.get("requested_files") or []),
                "boundary_interface_symbols": list(
                    boundary_pressure.get("interface_symbols") or []
                ),
            }
        )

        followup_plan = IssuePlan(
            summary=(
                "Repair hard verifier validity diagnostics."
                if validity_repair_followup
                else issue_plan.summary
            ),
            keywords=(
                ["verifier", "validity", "repair"]
                if validity_repair_followup
                else list(issue_plan.keywords)
            ),
            relevant_files=focus_files,
            risk_files=focus_files[: min(4, len(focus_files))] or list(issue_plan.risk_files),
            success_criteria=followup_success_criteria,
            rollout_briefs=rollout_briefs,
            repo_focus_map=focus_map,
            planner_source=f"{issue_plan.planner_source}+followup_ucb",
            planner_tokens=issue_plan.planner_tokens,
            difficulty_estimate=issue_plan.difficulty_estimate,
            recommended_rollouts=additional_rollouts,
            orchestration_primitives=list(issue_plan.orchestration_primitives),
            allocator_features=dict(issue_plan.allocator_features),
            unsolvable_reason=issue_plan.unsolvable_reason,
            test_context=test_context,
            task_state_context=dict(task_state_context),
            planner_metadata=planner_metadata,
            # Propagate the parent plan's evaluation constraints
            # (protect_visible_test_files, expected_test_count, etc.)
            # so the followup runs under the SAME contract as the
            # initial wave. Without this, the followup's persisted
            # ``issue_plan.json`` shows ``protect_visible_test_files=False``
            # while the rollouts actually executed under the parent's
            # constraints — which made post-hoc debugging impossible.
            evaluation_constraints=issue_plan.evaluation_constraints,
        )
        return self.apply_task_state_frontier(
            followup_plan,
            repo_context,
            task_state_context=task_state_context,
            stage_label="selection_followup",
        ) if not validity_repair_followup else followup_plan

    def _progressive_brief_family_key(self, brief: RolloutBrief) -> Any:
        policy = brief.search_policy if isinstance(brief.search_policy, dict) else {}
        family_index = policy.get("family_index")
        if isinstance(family_index, int):
            return family_index
        return brief.title.strip() or brief.goal.strip()

    def _artifact_list(self, artifact: Any, key: str) -> list[str]:
        if isinstance(artifact, dict):
            values = artifact.get(key)
        else:
            values = getattr(artifact, key, None)
        if not isinstance(values, (list, tuple, set)):
            return []
        return [str(value) for value in values if value]

    def _select_source_focus_files(
        self,
        issue_description: str,
        repo_context: RepoContext,
        keywords: list[str],
        relevant_files: list[str],
        task_regime: Optional[TaskRegimeProfile] = None,
    ) -> list[str]:
        source_dir = self._extract_primary_source_directory(issue_description)
        module_hints = self._extract_test_module_hints(relevant_files)
        if not source_dir and not (
            self._task_regime_probability(task_regime, "contract_gap") >= 0.5 or module_hints
        ):
            return []

        source_candidates = [
            file_info
            for file_info in repo_context.files
            if not self._looks_like_test_path(file_info.path)
            and (not source_dir or self._is_within_directory(file_info.path, source_dir))
        ]
        if not source_candidates:
            return []

        scored: list[tuple[float, str]] = []
        for file_info in source_candidates:
            path_lower = file_info.path.lower()
            symbol_text = " ".join(
                " ".join(
                    part
                    for part in [
                        symbol.name.lower(),
                        symbol.signature.lower(),
                        (symbol.docstring or "").lower(),
                    ]
                    if part
                )
                for symbol in file_info.symbols
            )
            score = 1.0
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in path_lower:
                    score += 4.0
                if keyword_lower in symbol_text:
                    score += 2.0
                if keyword_lower in " ".join(file_info.imports).lower():
                    score += 1.0

            stem = Path(file_info.path).stem.lower()
            for hint in module_hints:
                if stem == hint:
                    score += 8.0
                elif stem.startswith(f"{hint}_") or hint in stem:
                    score += 4.0

            if Path(file_info.path).name == "__init__.py":
                score -= 1.0
            scored.append((score, file_info.path))

        scored.sort(key=lambda item: (-item[0], item[1]))
        limit = min(max(6, len(module_hints) + 2), self.config.planning.max_relevant_files)
        return [path for _, path in scored[:limit]]

    def _extract_baseline_focus_files(
        self,
        *,
        repo_context: RepoContext,
        baseline_result: Optional[Any],
    ) -> list[str]:
        if baseline_result is None:
            return []

        failing_test_files = [
            path
            for path in self._normalize_repo_file_hints(
                repo_context,
                _baseline_test_ids(baseline_result, "failing_tests"),
            )
            if self._looks_like_test_path(path)
        ]
        traceback_signal = self._extract_traceback_signal(
            repo_context,
            _baseline_output(baseline_result),
        )
        traceback_source_files = _dedupe_preserve(
            list(traceback_signal.terminal_source_files)
            + list(traceback_signal.referenced_source_files)
            + list(traceback_signal.source_files)
        )
        traceback_test_files = list(traceback_signal.test_files)
        return self._normalize_repo_file_hints(
            repo_context,
            traceback_source_files + failing_test_files + traceback_test_files,
        )

    def _normalize_repo_python_path(
        self,
        repo_context: RepoContext,
        candidate: str,
    ) -> Optional[str]:
        return repo_context.normalize_repo_path_candidate(candidate)

    def _normalize_repo_file_hint(
        self,
        repo_context: RepoContext,
        candidate: str,
    ) -> Optional[str]:
        text = str(candidate or "").strip()
        if not text:
            return None
        return self._normalize_repo_python_path(
            repo_context,
            text.split("::", 1)[0].strip(),
        )

    def _normalize_repo_file_hints(
        self,
        repo_context: RepoContext,
        candidates: list[str],
    ) -> list[str]:
        normalized: list[str] = []
        for candidate in candidates:
            path = self._normalize_repo_file_hint(repo_context, str(candidate))
            if path:
                normalized.append(path)
        return _dedupe_preserve(normalized)

    def _extract_traceback_frame(
        self,
        repo_context: RepoContext,
        line: str,
    ) -> Optional[tuple[str, int]]:
        match = re.search(
            r"((?:[A-Za-z]:)?/?[A-Za-z0-9_./-]+\.[A-Za-z0-9]+):(\d+)",
            line,
        )
        if not match:
            match = re.search(
                r'File "((?:[A-Za-z]:)?/?[^"]+\.[A-Za-z0-9]+)", line (\d+)',
                line,
            )
            if not match:
                return None
        normalized = self._normalize_repo_python_path(
            repo_context,
            match.group(1),
        )
        if not normalized:
            return None
        try:
            line_number = int(match.group(2))
        except ValueError:
            return None
        return normalized, line_number

    def _extract_exception_summary(self, line: str) -> Optional[str]:
        text = line.strip()
        if not text:
            return None

        summary_patterns = (
            r"^ERROR\s+.+?-\s+([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)):\s*(.+)$",
            r"^(?:E\s+)?([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)):\s*(.+)$",
        )
        for pattern in summary_patterns:
            match = re.match(pattern, text)
            if match:
                return f"{match.group(1)}: {match.group(2).strip()}".strip()
        return None

    def _score_exception_summary(self, summary: str) -> float:
        text = summary.strip()
        if not text:
            return 0.0

        score = min(len(text) / 80.0, 2.0)
        if "@ " in text:
            score += 2.0
        if "..." not in text:
            score += 1.5
        if any(
            token in text.lower()
            for token in (
                "has no attribute",
                "cannot import name",
                "no module named",
                "not implemented",
            )
        ):
            score += 1.0
        return score

    def _source_line_for_frame(
        self,
        repo_context: RepoContext,
        path: str,
        line_number: int,
    ) -> str:
        if line_number <= 0:
            return ""
        repo_root = Path(repo_context.repo_path)
        try:
            return (repo_root / path).read_text(errors="replace").splitlines()[
                line_number - 1
            ]
        except (IndexError, OSError):
            return ""

    def _symbol_references_from_line(self, line: str) -> list[str]:
        text = str(line or "")
        if not text.strip():
            return []
        names: list[str] = []
        for pattern in (
            r"@\s*([A-Za-z_][A-Za-z0-9_]*)",
            r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?=\()",
        ):
            names.extend(match.group(1) for match in re.finditer(pattern, text))
        return _dedupe_preserve(
            [
                name
                for name in names
                if name.lower() not in _TRACEBACK_LINE_SYMBOL_NOISE
                and not name.startswith("__")
            ]
        )[:6]

    def _traceback_line_reference_signal(
        self,
        repo_context: RepoContext,
        frame: tuple[str, int],
    ) -> tuple[list[str], list[str]]:
        path, line_number = frame
        symbols = self._symbol_references_from_line(
            self._source_line_for_frame(repo_context, path, line_number)
        )
        referenced_files: list[str] = []
        referenced_symbols: list[str] = []
        for symbol in symbols:
            for node in repo_context.lookup_definition(symbol)[:4]:
                file_path = str(getattr(node, "file_path", "") or "").strip()
                if not file_path or file_path == path or self._looks_like_test_path(file_path):
                    continue
                referenced_files.append(file_path)
                referenced_symbols.append(symbol)
                file_info = repo_context.get_file_info(file_path)
                if file_info is None:
                    continue
                start_line = int(getattr(node, "start_line", 0) or 0)
                end_line = int(getattr(node, "end_line", start_line) or start_line)
                for neighbor in list(file_info.symbols or []):
                    neighbor_name = str(getattr(neighbor, "name", "") or "").strip()
                    if not neighbor_name or neighbor_name == symbol:
                        continue
                    neighbor_line = int(getattr(neighbor, "line_number", 0) or 0)
                    if start_line <= neighbor_line <= end_line + 80:
                        referenced_symbols.append(neighbor_name)
        return _dedupe_preserve(referenced_files), _dedupe_preserve(referenced_symbols)

    def _repo_paths_from_diagnostic_line(
        self,
        repo_context: RepoContext,
        line: str,
        *,
        limit: int = 8,
    ) -> list[str]:
        suffix = "|".join(re.escape(ext) for ext in _RESIDUAL_PATH_EXTENSIONS)
        pattern = re.compile(
            rf"(?P<path>(?:[A-Za-z]:)?[A-Za-z0-9_./\\-]+\.(?:{suffix}))(?::\d+)?"
        )
        paths: list[str] = []
        for match in pattern.finditer(str(line or "")):
            normalized = repo_context.normalize_repo_path_candidate(match.group("path"))
            if normalized:
                paths.append(normalized)
            if len(paths) >= limit:
                break
        return _dedupe_preserve(paths)[:limit]

    def _extract_traceback_signal(
        self,
        repo_context: RepoContext,
        output: str,
    ) -> _TracebackSignal:
        output = normalize_terminal_output(output)
        if not output:
            return _TracebackSignal()

        source_frames: list[str] = []
        test_frames: list[str] = []
        terminal_source_files: list[str] = []
        referenced_source_files: list[str] = []
        referenced_symbols: list[str] = []
        exception_summaries: list[str] = []
        current_frames: list[tuple[str, int]] = []

        for line in output.splitlines():
            frame = self._extract_traceback_frame(repo_context, line)
            if frame is not None:
                path, _ = frame
                current_frames.append(frame)
                if self._looks_like_test_path(path):
                    test_frames.append(path)
                else:
                    source_frames.append(path)
                continue

            exception_summary = self._extract_exception_summary(line)
            if not exception_summary:
                continue

            line_reference_files = [
                path
                for path in self._repo_paths_from_diagnostic_line(repo_context, line)
                if not self._looks_like_test_path(path)
            ]
            referenced_source_files.extend(line_reference_files)

            terminal_frame = next(
                (
                    item
                    for item in reversed(current_frames)
                    if not self._looks_like_test_path(item[0])
                ),
                None,
            )
            if terminal_frame is not None:
                terminal_source_files.append(terminal_frame[0])
                frame_reference_files, frame_reference_symbols = (
                    self._traceback_line_reference_signal(repo_context, terminal_frame)
                )
                referenced_source_files.extend(frame_reference_files)
                referenced_symbols.extend(frame_reference_symbols)
                exception_summaries.append(
                    f"{exception_summary} @ {terminal_frame[0]}:{terminal_frame[1]}"
                )
            else:
                exception_summaries.append(exception_summary)
            current_frames = []

        ordered_source_files = _dedupe_preserve(list(reversed(source_frames)))
        ordered_test_files = _dedupe_preserve(list(reversed(test_frames)))
        if not terminal_source_files and ordered_source_files:
            terminal_source_files = ordered_source_files[:2]
        ranked_exception_summaries = sorted(
            _dedupe_preserve(exception_summaries),
            key=lambda item: (-self._score_exception_summary(item), -len(item), item),
        )

        return _TracebackSignal(
            source_files=ordered_source_files,
            test_files=ordered_test_files,
            terminal_source_files=_dedupe_preserve(terminal_source_files),
            referenced_source_files=_dedupe_preserve(referenced_source_files),
            referenced_symbols=_dedupe_preserve(referenced_symbols),
            exception_summaries=ranked_exception_summaries[:4],
        )

    def _extract_repo_paths_from_output(
        self,
        repo_context: RepoContext,
        output: str,
    ) -> list[str]:
        if not output:
            return []

        traceback_signal = self._extract_traceback_signal(repo_context, output)
        ordered_traceback_paths = _dedupe_preserve(
            list(traceback_signal.source_files) + list(traceback_signal.test_files)
        )
        if ordered_traceback_paths:
            return ordered_traceback_paths

        candidate_paths = re.findall(
            r"(?:[A-Za-z]:)?/?[A-Za-z0-9_./-]+\.py",
            output,
        )
        if not candidate_paths:
            return []

        found: list[str] = []
        for candidate in candidate_paths:
            normalized = self._normalize_repo_python_path(repo_context, candidate)
            if normalized:
                found.append(normalized)
        return _dedupe_preserve(found)

    def _build_success_criteria(self, issue_description: str) -> list[str]:
        criteria = [
            "Reproduce the described behavior before or during the fix.",
            "Keep the final patch focused on the root cause.",
            "Run targeted verification before submitting the patch.",
        ]
        lowered = issue_description.lower()
        if "regression" in lowered or "existing test" in lowered or "tests" in lowered:
            criteria.append("Preserve existing behavior outside the bug fix.")
        if "edge" in lowered or "boundary" in lowered or "inclusive" in lowered:
            criteria.append("Handle the boundary conditions described in the issue.")
        return criteria

    def _extract_issue_summary(self, issue_description: str) -> str:
        fallback = "Resolve the reported issue."
        lines = [self._normalize_issue_line(line) for line in issue_description.splitlines()]
        candidates = [line for line in lines if line]
        if not candidates:
            return fallback

        best_line = ""
        best_score = float("-inf")
        for index, line in enumerate(candidates):
            score = self._score_issue_summary_line(line, position=index)
            if score > best_score:
                best_score = score
                best_line = line

        if best_line:
            return best_line[:220]

        return candidates[0][:220]

    def _normalize_issue_line(self, line: str) -> str:
        text = re.sub(r"\s+", " ", line.strip())
        return text.lstrip("-*0123456789. )")

    def _score_issue_summary_line(self, line: str, *, position: int) -> float:
        lowered = line.lower()
        score = 1.0 - min(position * 0.08, 0.4)

        if self._looks_like_issue_metadata(line):
            score -= 3.0
        if lowered.startswith(("this is ", "the repository starts ", "read the existing tests")):
            score -= 1.5
        if any(
            token in lowered
            for token in (
                "fix",
                "implement",
                "restore",
                "ensure",
                "support",
                "handle",
                "allow",
                "prevent",
                "correct",
                "complete",
                "repair",
                "make ",
            )
        ):
            score += 2.0
        if any(
            token in lowered
            for token in (
                "should",
                "must",
                "fails",
                "failure",
                "error",
                "traceback",
                "bug",
                "wrong",
                "missing",
                "pass",
                "regression",
            )
        ):
            score += 1.5
        if any(
            token in lowered
            for token in (
                "source directory",
                "test command",
                "install command",
                "benchmark instance",
                "benchmark repo",
            )
        ):
            score -= 2.0
        if len(line.split()) < 4:
            score -= 0.5
        return score

    def _looks_like_issue_metadata(self, line: str) -> bool:
        match = re.match(r"^(?P<key>[A-Za-z][A-Za-z0-9 _/()-]{0,48}):\s+.+$", line)
        if not match:
            return False
        key = match.group("key").strip().lower()
        if key in {"traceback", "error", "expected", "actual", "problem", "issue"}:
            return False
        metadata_keys = {
            "benchmark instance",
            "benchmark repo",
            "original upstream repo",
            "target python version",
            "primary source directory",
            "reference specification",
            "benchmark install command",
            "benchmark test command",
            "repository test command",
            "python version",
            "install command",
            "test command",
            "environment",
            "repo",
            "instance id",
            "base commit",
            "reference commit",
        }
        return key in metadata_keys

    def _build_evaluation_constraints(
        self,
        *,
        benchmark_metadata: Optional[dict[str, Any]] = None,
        test_command: Optional[str] = None,
    ) -> EvaluationConstraints:
        benchmark_metadata = (
            dict(benchmark_metadata) if isinstance(benchmark_metadata, dict) else {}
        )
        expected_test_ids = _dedupe_preserve(
            [
                str(test_id).strip()
                for test_id in list(benchmark_metadata.get("expected_test_ids") or [])
                if str(test_id).strip()
            ]
        )
        expected_test_count = 0
        raw_expected_test_count = benchmark_metadata.get("expected_test_count")
        if isinstance(raw_expected_test_count, int) and raw_expected_test_count > 0:
            expected_test_count = raw_expected_test_count
        if expected_test_ids:
            expected_test_count = max(expected_test_count, len(expected_test_ids))
        raw_test_inventory = dict(benchmark_metadata.get("test_inventory") or {})
        for metadata_key, inventory_key in (
            ("test_inventory_framework", "framework"),
            ("test_inventory_language", "language"),
            ("test_inventory_source", "source"),
            ("test_inventory_collection_command", "collection_command"),
            ("test_inventory_test_command", "test_command"),
        ):
            if (
                metadata_key in benchmark_metadata
                and benchmark_metadata.get(metadata_key)
                and not raw_test_inventory.get(inventory_key)
            ):
                raw_test_inventory[inventory_key] = benchmark_metadata.get(metadata_key)
        fallback_framework = infer_test_inventory_framework(
            expected_test_ids=expected_test_ids,
            test_command=test_command,
            explicit_framework=raw_test_inventory.get("framework") or "",
        )
        test_inventory = TestInventory.from_dict(raw_test_inventory).merged_with(
            TestInventory(
                framework=fallback_framework,
                language=str(benchmark_metadata.get("test_inventory_language") or "")
                .strip()
                .lower()
                or default_test_inventory_language(fallback_framework),
                source=(
                    str(benchmark_metadata.get("test_inventory_source") or "").strip().lower()
                    or ("benchmark_expected" if expected_test_count > 0 else "")
                ),
                expected_test_count=expected_test_count,
                expected_test_ids=expected_test_ids,
                collection_command=str(
                    benchmark_metadata.get("test_inventory_collection_command") or ""
                ).strip()
                or derive_test_collection_command(
                    test_command,
                    framework=fallback_framework,
                ),
                test_command=str(
                    benchmark_metadata.get("test_inventory_test_command") or test_command or ""
                ).strip(),
            )
        )
        expected_test_count = int(test_inventory.expected_test_count or 0)
        expected_test_ids = list(test_inventory.expected_test_ids)
        evidence_policy = infer_evidence_policy(
            evidence_mode=benchmark_metadata.get("evidence_mode")
            or benchmark_metadata.get("test_suite_evidence_mode")
            or benchmark_metadata.get("suite_evidence_mode"),
            test_inventory=test_inventory,
            has_visible_test_command=bool(test_command),
            expected_test_count=expected_test_count,
            expected_test_ids=expected_test_ids,
            default_provenance_label=str(test_inventory.source or ""),
        )
        return EvaluationConstraints(
            expected_test_count=expected_test_count,
            expected_test_ids=expected_test_ids,
            preserve_collected_test_coverage=expected_test_count > 0,
            protect_visible_test_files=bool(benchmark_metadata.get("protect_visible_test_files")),
            metadata={
                "benchmark_expected_test_count": expected_test_count,
                "evidence_mode": evidence_policy.mode,
            },
            test_inventory=test_inventory,
            evidence_policy=evidence_policy,
        )

    def _build_test_context(
        self,
        *,
        issue_description: str,
        repo_context: RepoContext,
        relevant_files: list[str],
        keywords: list[str],
        test_command: Optional[str] = None,
        baseline_result: Optional[Any] = None,
        evaluation_constraints: Optional[EvaluationConstraints] = None,
    ) -> TestContext:
        evaluation_constraints = (
            evaluation_constraints
            if isinstance(evaluation_constraints, EvaluationConstraints)
            else EvaluationConstraints()
        )
        resolved_test_inventory = evaluation_constraints.resolved_test_inventory().merged_with(
            TestInventory(
                framework=infer_test_inventory_framework(
                    expected_test_ids=evaluation_constraints.expected_test_ids,
                    test_command=test_command,
                    explicit_framework=(
                        evaluation_constraints.test_inventory.framework
                        if isinstance(evaluation_constraints.test_inventory, TestInventory)
                        else ""
                    ),
                ),
                language=default_test_inventory_language(
                    (
                        evaluation_constraints.test_inventory.framework
                        if isinstance(evaluation_constraints.test_inventory, TestInventory)
                        else ""
                    )
                    or infer_test_inventory_framework(
                        expected_test_ids=evaluation_constraints.expected_test_ids,
                        test_command=test_command,
                    )
                ),
                source=(
                    str(evaluation_constraints.test_inventory.source or "").strip().lower()
                    if isinstance(evaluation_constraints.test_inventory, TestInventory)
                    else ""
                ),
                expected_test_count=evaluation_constraints.expected_test_count,
                expected_test_ids=evaluation_constraints.expected_test_ids,
                collection_command=derive_test_collection_command(
                    test_command,
                    framework=(
                        evaluation_constraints.test_inventory.framework
                        if isinstance(evaluation_constraints.test_inventory, TestInventory)
                        else ""
                    )
                    or infer_test_inventory_framework(
                        expected_test_ids=evaluation_constraints.expected_test_ids,
                        test_command=test_command,
                    ),
                ),
                test_command=str(test_command or "").strip(),
            )
        )
        evidence_policy = evaluation_constraints.resolved_evidence_policy()

        traceback_signal = self._extract_traceback_signal(
            repo_context,
            _baseline_output(baseline_result),
        )
        source_focus_files = _dedupe_preserve(
            list(traceback_signal.referenced_source_files) + list(traceback_signal.source_files)
        )
        if not source_focus_files:
            source_focus_files = [
                path
                for path in self._extract_repo_paths_from_output(
                    repo_context,
                    _baseline_output(baseline_result),
                )
                if not self._looks_like_test_path(path)
            ]
        focus_test_files = self._collect_focus_test_files(
            repo_context=repo_context,
            relevant_files=relevant_files,
            keywords=keywords,
            issue_description=issue_description,
            source_focus_files=source_focus_files,
        )
        focus_test_files = _dedupe_preserve(
            list(traceback_signal.test_files) + list(focus_test_files)
        )
        failing_tests = _baseline_test_ids(baseline_result, "failing_tests")
        passing_tests = _baseline_test_ids(baseline_result, "passing_tests")
        failing_count = _baseline_test_count(baseline_result, failing_tests, "failing_tests")
        passing_count = _baseline_test_count(baseline_result, passing_tests, "passing_tests")
        if (
            not failing_tests
            and failing_count > 0
            and passing_count == 0
            and traceback_signal.test_files
        ):
            failing_tests = list(traceback_signal.test_files[:4])
        incomplete_test_files = self._detect_incomplete_test_files(
            repo_context,
            list(
                dict.fromkeys(
                    focus_test_files
                    + [test_id for test_id in failing_tests if self._looks_like_test_path(test_id)]
                )
            ),
        )
        incomplete_source_files = self._detect_incomplete_source_files(
            repo_context,
            _dedupe_preserve(
                list(traceback_signal.terminal_source_files)
                + list(source_focus_files)
                + [path for path in relevant_files if not self._looks_like_test_path(path)]
            )[:8],
        )

        expectation_ids = list(
            dict.fromkeys(
                failing_tests + self._extract_test_expectations(repo_context, focus_test_files)
            )
        )
        synthesized_authoritative_tests = (
            bool(focus_test_files)
            and not traceback_signal.test_files
            and not any(self._looks_like_test_path(path) for path in relevant_files)
        )
        summary_parts: list[str] = []
        planner_invariants = list(evaluation_constraints.planner_invariants())
        if evidence_policy.mode == EVIDENCE_MODE_GOLD_SUITE_VISIBLE:
            summary_parts.append(
                "The provided visible test suite is declared as the gold evaluation suite; mine it as the primary contract and optimize direct progress against it."
            )
        elif evidence_policy.mode == EVIDENCE_MODE_PARTIAL_SUITE_VISIBLE:
            summary_parts.append(
                "The provided visible tests are useful contract evidence but may not cover every evaluated behavior; implement the issue robustly beyond the visible cases."
            )
        elif evidence_policy.mode == EVIDENCE_MODE_EVAL_ONLY_SUITE:
            summary_parts.append(
                "The official evaluator is not rollout-visible; use only the repository, issue statement, and agent-visible commands as development evidence."
            )
        elif evidence_policy.mode == EVIDENCE_MODE_NO_SUITE_VISIBLE:
            summary_parts.append(
                "No authoritative visible test suite is declared; synthesize focused repros or checks from the issue and repository behavior."
            )
        if test_command:
            summary_parts.append("Use the repository's visible tests as part of the specification.")
        if failing_count or passing_count:
            summary_parts.append(
                "Baseline under the provided test command: "
                f"{failing_count} failing and {passing_count} already passing test cases."
            )
        if traceback_signal.exception_summaries:
            summary_parts.append(
                "Direct baseline exception: "
                + traceback_signal.exception_summaries[0].rstrip(".")
                + "."
            )
        if traceback_signal.terminal_source_files:
            summary_parts.append(
                "Terminal traceback focus: "
                + ", ".join(traceback_signal.terminal_source_files[:3])
                + "."
            )
        if source_focus_files:
            summary_parts.append(
                "Traceback and import paths point through: "
                + ", ".join(source_focus_files[:4])
                + "."
            )
        elif focus_test_files:
            summary_parts.append(
                f"Relevant visible tests are concentrated in {len(focus_test_files)} file(s)."
            )
        if synthesized_authoritative_tests:
            summary_parts.append(
                "Nearest authoritative tests: " + ", ".join(focus_test_files[:3]) + "."
            )
        if incomplete_source_files:
            summary_parts.append(
                "Likely incomplete source scaffolds near the traceback focus: "
                + ", ".join(incomplete_source_files[:3])
                + ". Complete the nearby implementation contract instead of masking only one failing reference."
            )
        if traceback_signal.referenced_symbols:
            summary_parts.append(
                "The terminal failing line references symbol(s): "
                + ", ".join(traceback_signal.referenced_symbols[:4])
                + ". Inspect their definitions before editing only the traceback line."
            )
        if incomplete_test_files:
            summary_parts.append(
                "Some visible test files appear intentionally incomplete with TODO/NotImplemented "
                "scaffolding: "
                + ", ".join(incomplete_test_files[:3])
                + ". Treat them primarily as specification. Prefer source fixes first; "
                "only complete explicit placeholder sections when the repository contract "
                "clearly requires it, and do not weaken assertions."
            )
        if "do not modify" in issue_description.lower() and "test" in issue_description.lower():
            summary_parts.append(
                "Do not edit existing tests unless the task explicitly requires it."
            )

        return TestContext(
            command=test_command,
            summary=" ".join(summary_parts).strip(),
            planner_invariants=planner_invariants,
            focus_test_files=focus_test_files[:6],
            incomplete_test_files=incomplete_test_files[:6],
            source_focus_files=source_focus_files[:6],
            incomplete_source_files=incomplete_source_files[:6],
            terminal_source_files=traceback_signal.terminal_source_files[:4],
            terminal_reference_symbols=traceback_signal.referenced_symbols[:6],
            exception_summaries=traceback_signal.exception_summaries[:4],
            failing_test_ids=failing_tests[:8],
            passing_test_ids=passing_tests[:6],
            failing_test_count=failing_count,
            passing_test_count=passing_count,
            expectations=expectation_ids[:8],
            expected_test_count=resolved_test_inventory.expected_test_count,
            expected_test_ids=list(resolved_test_inventory.expected_test_ids),
            test_inventory_framework=resolved_test_inventory.framework,
            test_inventory_language=resolved_test_inventory.language,
            test_inventory_source=resolved_test_inventory.source,
            test_collection_command=resolved_test_inventory.collection_command,
            evidence_mode=evidence_policy.mode,
            evidence_policy=evidence_policy.to_dict(),
        )

    def _collect_focus_test_files(
        self,
        *,
        repo_context: RepoContext,
        relevant_files: list[str],
        keywords: list[str],
        issue_description: str = "",
        source_focus_files: Optional[list[str]] = None,
    ) -> list[str]:
        explicit_focus = [
            path
            for path in self._normalize_repo_file_hints(repo_context, relevant_files)
            if self._looks_like_test_path(path)
        ]
        source_seeds = [
            path
            for path in self._normalize_repo_file_hints(
                repo_context,
                list(relevant_files or []) + list(source_focus_files or []),
            )
            if not self._looks_like_test_path(path)
        ]
        keyword_texts = _dedupe_preserve(
            list(keywords or [])
            + list(repo_context.extract_issue_keywords(issue_description, max_keywords=16))
        )
        keyword_tokens: set[str] = set()
        for value in keyword_texts:
            keyword_tokens.update(repo_context._path_affinity_tokens(value))
            keyword_tokens.update(repo_context._symbol_name_candidates(value))
        contract_anchor_tokens = self._extract_issue_contract_anchor_tokens(
            repo_context,
            issue_description=issue_description,
        )
        source_seeds = self._filter_focus_test_source_seeds(
            repo_context,
            source_seeds=source_seeds,
            keyword_tokens=keyword_tokens,
            contract_anchor_tokens=contract_anchor_tokens,
        )
        related_tests = repo_context.find_related_tests(
            source_seeds,
            seed_symbols=keywords,
            issue_description=issue_description,
            max_files=6,
        )
        candidate_pool = list(dict.fromkeys(explicit_focus + related_tests))
        repo_root = Path(repo_context.repo_path)
        if repo_root.is_dir() and keyword_tokens:
            ignored_dirs = {
                ".git",
                "__pycache__",
                "node_modules",
                ".tox",
                ".venv",
                "venv",
                ".mypy_cache",
                ".pytest_cache",
                "dist",
                "build",
                ".eggs",
                ".idea",
                ".vscode",
            }
            for root, dirs, files in os.walk(repo_root):
                dirs[:] = [
                    directory
                    for directory in dirs
                    if directory not in ignored_dirs and not directory.startswith(".")
                ]
                for filename in files:
                    rel_path = (Path(root) / filename).relative_to(repo_root).as_posix()
                    if rel_path in candidate_pool or not self._looks_like_test_path(rel_path):
                        continue
                    candidate_affinity_tokens = repo_context._path_affinity_tokens(rel_path)
                    if repo_context.get_file_info(rel_path) is not None:
                        candidate_affinity_tokens.update(repo_context._file_symbol_tokens(rel_path))
                    if not candidate_affinity_tokens.intersection(
                        keyword_tokens | contract_anchor_tokens
                    ):
                        continue
                    candidate_pool.append(rel_path)

        if candidate_pool:
            explicit_set = set(explicit_focus)
            related_set = set(related_tests)
            source_token_cache = {
                path: (
                    repo_context._path_affinity_tokens(path)
                    | repo_context._file_symbol_tokens(path)
                )
                for path in source_seeds
            }
            test_source_link_weight = getattr(repo_context, "_test_source_link_weight", None)

            def candidate_tokens(path: str) -> set[str]:
                tokens = repo_context._path_affinity_tokens(
                    path
                ) | repo_context._file_symbol_tokens(path)
                if repo_context.get_file_info(path) is not None:
                    return tokens
                file_path = repo_root / path
                try:
                    content = file_path.read_text(errors="replace")[:16000]
                except OSError:
                    return tokens
                for token in re.findall(r"[A-Za-z_][A-Za-z0-9_./-]{2,}", content):
                    for piece in re.split(r"[^A-Za-z0-9]+", token):
                        if not piece:
                            continue
                        tokens.update(repo_context._split_identifier_tokens(piece))
                return tokens

            candidate_token_map = {path: candidate_tokens(path) for path in candidate_pool}
            keyword_token_weights = self._focus_test_keyword_token_weights(
                candidate_token_map=candidate_token_map,
                keyword_tokens=keyword_tokens,
            )
            contract_token_weights = self._focus_test_keyword_token_weights(
                candidate_token_map=candidate_token_map,
                keyword_tokens=contract_anchor_tokens,
            )
            weighted_contract_overlap_map: dict[str, float] = {}
            weighted_basename_contract_overlap_map: dict[str, float] = {}
            sibling_contract_specificity: dict[str, float] = {}
            for path, tokens in candidate_token_map.items():
                contract_overlap = tokens.intersection(contract_anchor_tokens)
                weighted_contract_overlap_map[path] = sum(
                    contract_token_weights.get(token, 0.0) for token in contract_overlap
                )
                basename_contract_overlap = self._focus_test_basename_tokens(
                    path,
                    repo_context=repo_context,
                ).intersection(contract_anchor_tokens)
                weighted_basename_contract_overlap_map[path] = sum(
                    contract_token_weights.get(token, 0.0) for token in basename_contract_overlap
                )
                path_obj = Path(path)
                if path_obj.suffix.lower() in _FOCUS_TEST_DATA_FILE_SUFFIXES:
                    parent_key = path_obj.parent.as_posix()
                    sibling_contract_specificity[parent_key] = max(
                        sibling_contract_specificity.get(parent_key, 0.0),
                        weighted_basename_contract_overlap_map[path],
                    )
            scored_candidates: list[tuple[float, str]] = []
            for path in candidate_pool:
                tokens = candidate_token_map.get(path, set())
                keyword_overlap = tokens.intersection(keyword_tokens)
                weighted_keyword_overlap = sum(
                    keyword_token_weights.get(token, 0.0) for token in keyword_overlap
                )
                weighted_contract_overlap = weighted_contract_overlap_map.get(path, 0.0)
                weighted_basename_contract_overlap = weighted_basename_contract_overlap_map.get(
                    path,
                    0.0,
                )
                rare_keyword_hits = sum(
                    1 for token in keyword_overlap if keyword_token_weights.get(token, 0.0) >= 1.15
                )
                source_overlap = max(
                    (
                        len(tokens.intersection(source_tokens))
                        for source_tokens in source_token_cache.values()
                    ),
                    default=0,
                )
                source_link = (
                    max(
                        (
                            float(test_source_link_weight(path, source_path))
                            for source_path in source_seeds
                        ),
                        default=0.0,
                    )
                    if callable(test_source_link_weight)
                    else 0.0
                )
                if (
                    path in explicit_set
                    and path not in related_set
                    and len(explicit_focus) > 1
                    and weighted_keyword_overlap <= 0.0
                    and source_overlap <= 0
                    and source_link <= 0.0
                ):
                    continue

                score = 0.0
                if path in explicit_set:
                    score += (
                        1.0 if self._looks_like_support_test_artifact(path, repo_context) else 1.6
                    )
                if path in related_set:
                    score += (
                        2.1 if self._looks_like_support_test_artifact(path, repo_context) else 4.2
                    )
                if weighted_keyword_overlap > 0.0:
                    score += min(6.0, weighted_keyword_overlap)
                if rare_keyword_hits > 0:
                    score += min(2.1, 0.8 * rare_keyword_hits)
                if weighted_contract_overlap > 0.0:
                    score += min(7.2, 1.9 * weighted_contract_overlap)
                if weighted_basename_contract_overlap > 0.0:
                    score += min(4.8, 2.4 * weighted_basename_contract_overlap)
                if source_overlap > 0:
                    score += min(4.2, 1.0 * source_overlap)
                if source_link > 0.0:
                    score += min(3.6, 1.4 * source_link)
                score += self._focus_test_candidate_authority_bias(
                    path,
                    repo_context=repo_context,
                    weighted_keyword_overlap=weighted_keyword_overlap,
                    weighted_contract_overlap=weighted_contract_overlap,
                    weighted_basename_contract_overlap=weighted_basename_contract_overlap,
                    sibling_contract_specificity=sibling_contract_specificity.get(
                        Path(path).parent.as_posix(),
                        0.0,
                    ),
                )
                if score <= 0.0:
                    continue
                scored_candidates.append((score, path))

            scored_candidates.sort(key=lambda item: (-item[0], item[1]))
            if scored_candidates:
                primary_paths = [
                    path
                    for _score, path in scored_candidates
                    if not self._looks_like_support_test_artifact(path, repo_context)
                ]
                promotable_primary_paths = [
                    path
                    for path in primary_paths
                    if self._focus_test_candidate_promotable(
                        path,
                        weighted_contract_overlap=weighted_contract_overlap_map.get(path, 0.0),
                        weighted_basename_contract_overlap=weighted_basename_contract_overlap_map.get(
                            path,
                            0.0,
                        ),
                        sibling_contract_specificity=sibling_contract_specificity.get(
                            Path(path).parent.as_posix(),
                            0.0,
                        ),
                    )
                ]
                support_paths = [
                    path
                    for _score, path in scored_candidates
                    if self._looks_like_support_test_artifact(path, repo_context)
                ]
                if promotable_primary_paths:
                    return _dedupe_preserve(promotable_primary_paths)[:6]
                if primary_paths:
                    return _dedupe_preserve(primary_paths)[:6]
                return _dedupe_preserve(support_paths)[:6]

        if explicit_focus:
            return list(dict.fromkeys(explicit_focus))
        if related_tests:
            return related_tests

        scored: list[tuple[float, str]] = []
        lowered_keywords = [keyword.lower() for keyword in keywords if keyword]
        for file_info in repo_context.files:
            if not self._looks_like_test_path(file_info.path):
                continue
            path_lower = file_info.path.lower()
            symbol_text = " ".join(symbol.name.lower() for symbol in file_info.symbols)
            score = 0.0
            for keyword in lowered_keywords:
                if keyword in path_lower:
                    score += 2.0
                if keyword in symbol_text:
                    score += 1.0
            scored.append((score, file_info.path))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [path for _, path in scored[:6] if path]

    def _focus_test_keyword_token_weights(
        self,
        *,
        candidate_token_map: dict[str, set[str]],
        keyword_tokens: set[str],
    ) -> dict[str, float]:
        token_doc_frequency: Counter[str] = Counter()
        for tokens in candidate_token_map.values():
            for token in tokens.intersection(keyword_tokens):
                token_doc_frequency[token] += 1
        return {
            token: max(0.35, 2.2 - math.log1p(doc_frequency))
            for token, doc_frequency in token_doc_frequency.items()
        }

    def _extract_issue_contract_anchor_tokens(
        self,
        repo_context: RepoContext,
        *,
        issue_description: str,
    ) -> set[str]:
        anchors: set[str] = set()
        headline = self._extract_issue_headline(issue_description)
        fragments = [headline] if headline else []
        fragments.extend(
            fragment
            for fragment in re.findall(r"`([^`]+)`", str(issue_description or ""))
            if str(fragment).strip()
        )
        for fragment in fragments:
            anchors.update(repo_context._path_affinity_tokens(fragment))
            anchors.update(repo_context._symbol_name_candidates(fragment))
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.-]{2,}", fragment):
                anchors.update(repo_context._split_identifier_tokens(token))
        return anchors

    def _filter_focus_test_source_seeds(
        self,
        repo_context: RepoContext,
        *,
        source_seeds: list[str],
        keyword_tokens: set[str],
        contract_anchor_tokens: set[str],
    ) -> list[str]:
        normalized = _dedupe_preserve(
            path for path in list(source_seeds or []) if str(path or "").strip()
        )
        if len(normalized) <= 1:
            return normalized

        ranked: list[tuple[float, str]] = []
        for path in normalized:
            tokens = repo_context._path_affinity_tokens(path) | repo_context._file_symbol_tokens(
                path
            )
            contract_overlap = tokens.intersection(contract_anchor_tokens)
            keyword_overlap = tokens.intersection(keyword_tokens)
            basename_overlap = self._focus_test_basename_tokens(
                path,
                repo_context=repo_context,
            ).intersection(contract_anchor_tokens)
            score = 0.0
            if basename_overlap:
                score += 4.0 * len(basename_overlap)
            if contract_overlap:
                score += 2.4 * len(contract_overlap)
            if keyword_overlap:
                score += min(4.0, 0.9 * len(keyword_overlap))
            ranked.append((score, path))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        best_score = ranked[0][0] if ranked else 0.0
        if best_score <= 0.0:
            return normalized[:4]

        filtered = [path for score, path in ranked if score >= max(1.5, best_score * 0.45)]
        return filtered[:4] if filtered else [ranked[0][1]]

    def _extract_issue_headline(self, issue_description: str) -> str:
        for raw_line in str(issue_description or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r"^#+\s*", "", line).strip()
            title_match = re.match(r"(?i)^title:\s*(.+)$", line)
            if title_match:
                return title_match.group(1).strip()
            if not re.match(r"(?i)^(summary|description):\s*$", line):
                return line
        return ""

    def _focus_test_basename_tokens(
        self,
        path: str,
        *,
        repo_context: RepoContext,
    ) -> set[str]:
        target = Path(str(path or "").strip().replace("\\", "/"))
        stem = target.stem
        if stem == "__init__":
            stem = target.parent.name
        tokens: set[str] = set()
        for raw_token in re.split(r"[^A-Za-z0-9]+", stem):
            if not raw_token:
                continue
            tokens.update(repo_context._split_identifier_tokens(raw_token))
        return tokens

    def _is_direct_named_test_file(self, path: str) -> bool:
        name = Path(str(path or "")).name.lower()
        return (
            name == "conftest.py"
            or name.startswith("test_")
            or name.endswith("_test.py")
            or name.endswith("_test.go")
            or name.endswith("_test.rs")
            or name.endswith("_spec.rb")
            or ".test." in name
            or ".spec." in name
            or re.search(r"(?:^|[._-])test[s]?\.(?:java|kt|kts|scala|cs)$", name) is not None
            or re.search(r"(?:^|[._-])spec\.(?:java|kt|kts|scala|cs)$", name) is not None
        )

    def _looks_like_support_test_artifact(
        self,
        path: str,
        repo_context: RepoContext,
    ) -> bool:
        normalized = str(path or "").strip().replace("\\", "/")
        path_obj = Path(normalized)
        parts = {part.lower() for part in path_obj.parts}
        name = path_obj.name.lower()
        direct_named_test = self._is_direct_named_test_file(normalized)
        if parts.intersection(_FOCUS_TEST_SUPPORT_PATH_TOKENS):
            return True
        if ("test" in parts or "tests" in parts) and "lib" in parts and not direct_named_test:
            return True
        if (
            "targets" in parts
            and "library" in parts
            and path_obj.suffix.lower() in _FOCUS_TEST_CODE_FILE_SUFFIXES
            and not direct_named_test
        ):
            return True
        file_info = repo_context.get_file_info(normalized)
        if (
            file_info is not None
            and path_obj.suffix.lower() in _FOCUS_TEST_CODE_FILE_SUFFIXES
            and name != "conftest.py"
            and not direct_named_test
            and not any(symbol.name.startswith("test_") for symbol in file_info.symbols)
        ):
            return True
        return False

    def _focus_test_candidate_authority_bias(
        self,
        path: str,
        *,
        repo_context: RepoContext,
        weighted_keyword_overlap: float,
        weighted_contract_overlap: float = 0.0,
        weighted_basename_contract_overlap: float = 0.0,
        sibling_contract_specificity: float = 0.0,
    ) -> float:
        normalized = str(path or "").strip().replace("\\", "/")
        path_obj = Path(normalized)
        parts = {part.lower() for part in path_obj.parts}
        suffix = path_obj.suffix.lower()
        direct_named_test = self._is_direct_named_test_file(normalized)
        support_like = self._looks_like_support_test_artifact(normalized, repo_context)
        bias = 0.0
        if parts.intersection(_FOCUS_TEST_HIGH_AUTHORITY_PATH_TOKENS):
            bias += 1.2
        if suffix in _FOCUS_TEST_DATA_FILE_SUFFIXES and ("test" in parts or "tests" in parts):
            bias += 1.1
        if weighted_keyword_overlap >= 2.0 and parts.intersection(
            _FOCUS_TEST_HIGH_AUTHORITY_PATH_TOKENS
        ):
            bias += 0.8
        if (
            suffix in _FOCUS_TEST_DATA_FILE_SUFFIXES
            and path_obj.stem.lower() in _FOCUS_TEST_GENERIC_SCENARIO_BASENAMES
            and weighted_contract_overlap < 2.4
            and weighted_basename_contract_overlap <= 0.0
        ):
            bias -= 1.8
        if (
            suffix in _FOCUS_TEST_DATA_FILE_SUFFIXES
            and weighted_basename_contract_overlap <= 0.0
            and sibling_contract_specificity >= 1.1
        ):
            bias -= min(3.8, 1.6 * sibling_contract_specificity)
        if support_like:
            bias -= 3.2 if path_obj.name == "__init__.py" else 2.2
        elif path_obj.name == "__init__.py" and not direct_named_test:
            bias -= 1.0
        return bias

    def _focus_test_candidate_promotable(
        self,
        path: str,
        *,
        weighted_contract_overlap: float,
        weighted_basename_contract_overlap: float,
        sibling_contract_specificity: float,
    ) -> bool:
        path_obj = Path(str(path or "").strip().replace("\\", "/"))
        if path_obj.suffix.lower() not in _FOCUS_TEST_DATA_FILE_SUFFIXES:
            return True
        if weighted_basename_contract_overlap > 0.0:
            return True
        if sibling_contract_specificity >= 1.1:
            return False
        return weighted_contract_overlap >= 1.2

    def _extract_test_expectations(
        self,
        repo_context: RepoContext,
        focus_test_files: list[str],
    ) -> list[str]:
        expectations: list[str] = []
        for path in focus_test_files:
            if self._looks_like_support_test_artifact(path, repo_context):
                continue
            file_info = repo_context.get_file_info(path)
            if file_info is None:
                expectations.append(path)
                continue
            test_symbols = [
                symbol.name for symbol in file_info.symbols if symbol.name.startswith("test_")
            ]
            if not test_symbols:
                expectations.append(path)
                continue
            expectations.extend(f"{path}::{symbol}" for symbol in test_symbols[:4])
        return list(dict.fromkeys(expectations))

    def _detect_incomplete_test_files(
        self,
        repo_context: RepoContext,
        candidate_test_files: list[str],
    ) -> list[str]:
        repo_root = Path(repo_context.repo_path)
        incomplete: list[str] = []
        for rel_path in candidate_test_files:
            if not self._looks_like_test_path(rel_path):
                continue
            file_path = repo_root / rel_path
            try:
                content = file_path.read_text(errors="replace")
            except OSError:
                continue
            lowered = content.lower()
            if (
                "raise notimplementederror" in lowered
                or "todo: implement" in lowered
                or "need to implement for task" in lowered
            ):
                incomplete.append(rel_path)
        return list(dict.fromkeys(incomplete))

    def _detect_incomplete_source_files(
        self,
        repo_context: RepoContext,
        candidate_source_files: list[str],
    ) -> list[str]:
        repo_root = Path(repo_context.repo_path)
        incomplete: list[str] = []
        for rel_path in candidate_source_files:
            if self._looks_like_test_path(rel_path):
                continue
            file_path = repo_root / rel_path
            try:
                content = file_path.read_text(errors="replace")
            except OSError:
                continue
            lowered = content.lower()
            pass_count = len(re.findall(r"(?m)^\s+pass\s*$", content))
            scaffold_markers = pass_count
            if "raise notimplementederror" in lowered:
                scaffold_markers += 2
            if "todo: implement" in lowered or "need to implement for task" in lowered:
                scaffold_markers += 2
            if scaffold_markers >= 2:
                incomplete.append(rel_path)
        return list(dict.fromkeys(incomplete))

    def _completion_neighbor_focus_files(
        self,
        issue_plan: IssuePlan,
        repo_context: RepoContext,
        test_context: TestContext,
    ) -> list[str]:
        entrypoint_seed_files = _dedupe_preserve(
            [
                path
                for path in issue_plan.relevant_files
                if not self._looks_like_test_path(path) and Path(path).name == "__init__.py"
            ]
        )[:2]
        remaining_seed_files = _dedupe_preserve(
            list(test_context.terminal_source_files)
            + list(test_context.source_focus_files)
            + [path for path in issue_plan.relevant_files if not self._looks_like_test_path(path)][
                :2
            ]
        )[:6]
        seed_files = _dedupe_preserve(entrypoint_seed_files + remaining_seed_files)[:6]
        if not seed_files:
            return []

        direct_import_neighbors = self._local_import_focus_files(
            repo_context,
            entrypoint_seed_files,
            max_files=8,
        )
        direct_import_neighbors.extend(
            self._local_import_focus_files(
                repo_context,
                remaining_seed_files,
                max_files=8,
            )
        )
        dependency_neighbors = [
            path
            for path in repo_context.get_dependency_neighbors(seed_files, max_neighbors=8)
            if not self._looks_like_test_path(path)
        ]
        return _dedupe_preserve(direct_import_neighbors + dependency_neighbors)

    def _local_import_focus_files(
        self,
        repo_context: RepoContext,
        seed_files: list[str],
        *,
        max_files: int,
    ) -> list[str]:
        module_to_path: dict[str, str] = {}
        for file_info in repo_context.files:
            if file_info.language != "python":
                continue
            path = file_info.path
            # Guard against root or empty paths — `with_suffix("")` raises
            # `ValueError: PosixPath('/') has an empty name` on those.
            path_obj = Path(path)
            if not path_obj.name:
                continue
            module_parts = list(path_obj.with_suffix("").parts)
            if path_obj.name == "__init__.py":
                module_parts = list(path_obj.parts[:-1])
            if not module_parts:
                continue
            module_to_path[".".join(module_parts)] = path

        resolved: list[str] = []
        for seed in seed_files:
            file_info = repo_context.get_file_info(seed)
            if file_info is None:
                continue
            seed_path = Path(seed)
            if not seed_path.name:
                continue
            package_parts = list(seed_path.parts[:-1])
            if seed_path.name != "__init__.py":
                package_parts = list(seed_path.with_suffix("").parts[:-1])
            package_prefix = ".".join(package_parts)
            for import_name in file_info.imports:
                normalized = str(import_name or "").strip().rstrip(".*")
                if not normalized:
                    continue
                candidates = [normalized]
                parts = [part for part in normalized.split(".") if part]
                for end in range(len(parts) - 1, 0, -1):
                    candidates.append(".".join(parts[:end]))
                if package_prefix:
                    candidates.extend(
                        f"{package_prefix}.{candidate}"
                        for candidate in list(candidates)
                        if candidate and candidate != package_prefix
                    )
                for candidate in _dedupe_preserve(candidates):
                    path = module_to_path.get(candidate)
                    if path and path != seed and not self._looks_like_test_path(path):
                        resolved.append(path)
                        break
            if len(_dedupe_preserve(resolved)) >= max_files:
                break
        return _dedupe_preserve(resolved)[:max_files]

    def _looks_like_test_path(self, path: str) -> bool:
        normalized = str(path or "").strip().replace("\\", "/")
        name = Path(normalized).name.lower()
        parts = {part.lower() for part in Path(normalized).parts}
        return (
            "test" in parts
            or "tests" in parts
            or "spec" in parts
            or "__tests__" in parts
            or "testdata" in parts
            or name == "conftest.py"
            or name.startswith("test_")
            or name.endswith("_test.py")
            or name.endswith("_test.go")
            or name.endswith("_test.rs")
            or name.endswith("_spec.rb")
            or ".test." in name
            or ".spec." in name
            or re.search(r"(?:^|[._-])test[s]?\.(?:java|kt|kts|scala|cs)$", name) is not None
            or re.search(r"(?:^|[._-])spec\.(?:java|kt|kts|scala|cs)$", name) is not None
        )

    def _extract_primary_source_directory(self, issue_description: str) -> Optional[str]:
        match = re.search(
            r"Primary source directory:\s*([^\n]+)",
            issue_description,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        source_dir = match.group(1).strip().rstrip("/")
        if source_dir in {"", "."}:
            return None
        return source_dir.lstrip("./")

    def _extract_test_module_hints(self, relevant_files: list[str]) -> list[str]:
        hints = []
        for path in relevant_files:
            name = Path(path).name.lower()
            if not name.startswith("test_") or not name.endswith(".py"):
                continue
            hint = Path(path).stem[len("test_") :]
            if hint:
                hints.append(hint)
        return list(dict.fromkeys(hints))

    def _is_within_directory(self, path: str, directory: str) -> bool:
        normalized_directory = directory.rstrip("/")
        return path == normalized_directory or path.startswith(normalized_directory + "/")

    def _build_rollout_briefs(
        self,
        issue_description: str,
        repo_context: RepoContext,
        relevant_files: list[str],
        success_criteria: list[str],
        rollout_count: Optional[int] = None,
        task_regime: Optional[TaskRegimeProfile] = None,
    ) -> list[RolloutBrief]:
        requested_rollouts = rollout_count or self.config.rollout.num_rollouts
        agent_mode = self._choose_agent_mode(repo_context, relevant_files)
        families = self._build_rollout_brief_families(
            issue_description=issue_description,
            repo_context=repo_context,
            relevant_files=relevant_files,
            success_criteria=success_criteria,
            agent_mode=agent_mode,
            requested_rollouts=requested_rollouts,
            task_regime=task_regime,
        )
        return self._expand_rollout_briefs(
            base_briefs=families,
            issue_description=issue_description,
            repo_context=repo_context,
            relevant_files=relevant_files,
            requested_rollouts=requested_rollouts,
            task_regime=task_regime,
        )

    def _build_rollout_brief_families(
        self,
        *,
        issue_description: str,
        repo_context: RepoContext,
        relevant_files: list[str],
        success_criteria: list[str],
        agent_mode: AgentMode,
        requested_rollouts: int,
        task_regime: Optional[TaskRegimeProfile] = None,
    ) -> list[RolloutBrief]:
        task_regime = task_regime or TaskRegimeProfile()
        family_cap = max(
            1,
            min(
                requested_rollouts,
                self.config.planning.max_rollout_brief_families,
            ),
        )
        source_files = self._source_file_candidates(relevant_files)
        tests = [path for path in relevant_files if self._looks_like_test_path(path)]
        primary = source_files[:4] or relevant_files[:4]
        neighbors = repo_context.get_dependency_neighbors(primary[:3], max_neighbors=4)
        source_windows = self._build_source_windows(
            source_files or relevant_files,
            max_windows=max(family_cap + 1, 3),
        )
        seed_window = source_windows[0] if source_windows else primary
        completion_task = self._task_regime_probability(task_regime, "contract_gap") >= 0.5
        public_api_task = self._task_regime_probability(task_regime, "high_interface_risk") >= 0.45

        templates: list[RolloutBrief] = [
            RolloutBrief(
                title="Minimal patch",
                goal="Fix the most likely root cause with the smallest safe edit.",
                focus_files=primary,
                hypotheses=[
                    "A boundary check or return path is incorrect.",
                    "The failing behavior can be resolved without refactoring.",
                ],
                success_criteria=success_criteria,
                prompt_hint="Bias toward direct, minimal edits.",
                agent_mode=agent_mode,
                search_policy={"mode": "surgical", "verification_focus": "targeted_validation"},
            ),
            RolloutBrief(
                title="Test-first validation",
                goal="Derive the intended behavior from visible tests, then fix the implementation.",
                focus_files=_dedupe_preserve(tests[:2] + primary),
                hypotheses=[
                    "Existing tests already describe the bug or are close to it.",
                    "Visible tests are the fastest route to the correct contract.",
                ],
                success_criteria=success_criteria,
                prompt_hint="Lead with reproduction and targeted validation.",
                agent_mode=agent_mode,
                search_policy={"mode": "test_rooted", "verification_focus": "focus_test_files"},
            ),
            RolloutBrief(
                title="Dependency-aware fix",
                goal="Trace the failure through directly related modules before editing.",
                focus_files=_dedupe_preserve(primary + neighbors),
                hypotheses=[
                    "The bug may live in a helper or imported module adjacent to the obvious file.",
                    "Neighbor modules may reveal the intended contract.",
                ],
                success_criteria=success_criteria,
                prompt_hint="Inspect surrounding dependencies before changing code.",
                agent_mode=agent_mode,
                search_policy={
                    "mode": "dependency_trace",
                    "verification_focus": "targeted_validation",
                },
            ),
            RolloutBrief(
                title="Regression hardening",
                goal="Fix the issue while also checking nearby edge cases and invariants.",
                focus_files=relevant_files[:8],
                hypotheses=[
                    "The immediate bug may share a root cause with adjacent edge cases.",
                    "A slightly broader fix may prevent follow-on regressions.",
                ],
                success_criteria=success_criteria,
                prompt_hint="Prefer robust fixes if they stay tightly scoped.",
                agent_mode=agent_mode,
                search_policy={"mode": "invariant_guard", "verification_focus": "focus_test_files"},
            ),
        ]

        if completion_task or public_api_task:
            templates.append(
                RolloutBrief(
                    title="API contract sweep",
                    goal="Infer missing library behavior from tests and public entrypoints, then complete the contract.",
                    focus_files=_dedupe_preserve(tests[:2] + seed_window),
                    hypotheses=[
                        "The task is closer to repository completion than a single-line bug fix.",
                        "Public entrypoints and their tests encode the missing behavior.",
                    ],
                    success_criteria=success_criteria,
                    prompt_hint="Prioritize visible behavior and public contracts over narrow local edits.",
                    agent_mode=agent_mode,
                    search_policy={
                        "mode": "api_contract",
                        "verification_focus": "focus_test_files",
                        "origin": "regime_candidate"
                        if completion_task or public_api_task
                        else "heuristic",
                        "origin_regime_state": (
                            "high_interface_risk" if public_api_task else "contract_gap"
                        ),
                    },
                )
            )

        for cluster_index, window in enumerate(source_windows[1:], start=1):
            templates.append(
                RolloutBrief(
                    title=f"Source cluster {cluster_index}",
                    goal="Inspect a distinct source cluster for an alternate root cause or missing implementation path.",
                    focus_files=window,
                    hypotheses=[
                        "A different source cluster may contain the real failure or missing behavior.",
                    ],
                    success_criteria=success_criteria,
                    prompt_hint="Probe a distinct code cluster rather than duplicating earlier search.",
                    agent_mode=agent_mode,
                    search_policy={
                        "mode": "source_cluster",
                        "verification_focus": "targeted_validation",
                        "cluster_index": cluster_index,
                    },
                )
            )
            if len(templates) >= family_cap:
                break

        deduped: list[RolloutBrief] = []
        seen_titles: set[str] = set()
        for template in templates:
            if template.title in seen_titles:
                continue
            seen_titles.add(template.title)
            deduped.append(template)
            if len(deduped) >= family_cap:
                break
        return deduped

    def _expand_rollout_briefs(
        self,
        *,
        base_briefs: list[RolloutBrief],
        issue_description: str,
        repo_context: RepoContext,
        relevant_files: list[str],
        requested_rollouts: int,
        task_regime: Optional[TaskRegimeProfile] = None,
    ) -> list[RolloutBrief]:
        if not base_briefs:
            return []

        source_files = self._source_file_candidates(relevant_files) or relevant_files
        test_files = [path for path in relevant_files if self._looks_like_test_path(path)]
        risk_files = relevant_files[:4]
        source_windows = self._build_source_windows(
            source_files or relevant_files,
            max_windows=max(requested_rollouts, len(base_briefs)),
        )

        expanded: list[RolloutBrief] = []
        for rollout_index in range(max(1, requested_rollouts)):
            template = RolloutBrief.from_dict(
                base_briefs[rollout_index % len(base_briefs)].to_dict()
            )
            policy = self._normalize_brief_search_policy(template)
            variant_index = rollout_index // len(base_briefs)
            if source_windows:
                window = source_windows[(rollout_index + variant_index) % len(source_windows)]
            else:
                window = template.focus_files or relevant_files[:4]

            template.focus_files = self._apply_search_policy_focus(
                policy.get("mode", "surgical"),
                current_focus=template.focus_files,
                source_window=window,
                test_files=test_files,
                risk_files=risk_files,
                repo_context=repo_context,
            )[:8]
            if variant_index > 0:
                template.prompt_hint = (
                    f"{template.prompt_hint} Cover a distinct file cluster and avoid duplicating earlier rollouts."
                ).strip()
                template.hypotheses = _dedupe_preserve(
                    template.hypotheses
                    + [
                        self._variant_hypothesis(
                            policy.get("mode", "surgical"), window, variant_index
                        )
                    ]
                )[:5]
            policy.update(
                {
                    "family_index": rollout_index % len(base_briefs),
                    "variant_index": variant_index,
                    "requested_rollouts": requested_rollouts,
                    "rollout_index": rollout_index,
                    "rollout_stage_model_signature": list(
                        self.config.get_rollout_stage_model_signature(rollout_index)
                    ),
                    "rollout_profile_signature": list(
                        self.config.get_rollout_diversity_signature(
                            rollout_index,
                            include_prompt_strategy=bool(
                                self.config.rollout.portfolio_diversity_include_prompt_strategy
                            ),
                            include_temperature=bool(
                                self.config.rollout.portfolio_diversity_include_temperature
                            ),
                        )
                    ),
                    "rollout_route_signature": list(
                        self.config.get_rollout_diversity_signature(
                            rollout_index,
                            include_prompt_strategy=True,
                            include_temperature=True,
                        )
                    ),
                    "rollout_prompt_strategy": (
                        self.config.get_prompt_strategy_for_rollout(rollout_index).value
                    ),
                    "rollout_temperature": float(
                        self.config.get_temperature_for_rollout(rollout_index)
                    ),
                }
            )
            template.set_controller_action(policy, merge_policy=policy)
            template.search_policy["allocator_arm"] = _rollout_brief_allocation_key(template)
            template.controller_action.allocator_arm = template.search_policy["allocator_arm"]
            template.search_policy = template.controller_action.to_search_policy(
                base=template.search_policy
            )
            expanded.append(template)
        self._assign_rollout_profiles(
            briefs=expanded,
            issue_description=issue_description,
            repo_context=repo_context,
            relevant_files=relevant_files,
            test_files=test_files,
            task_regime=task_regime,
        )
        return expanded

    def _build_rollout_profile_descriptors(self) -> list[dict[str, Any]]:
        if not self.config.rollout.llm_profiles:
            return []
        resolve_target_runtime_availability = any(
            str(
                (getattr(llm_config, "cli_env_overrides", {}) or {}).get(
                    "APEX_TARGET_TOOL_CONTEXT"
                )
                or ""
            ).strip()
            for llm_config in list(self.config.llm_configs or [])
        )

        def _resolved_descriptor_llm(
            requested: LLMConfig,
            *,
            profile_index: int,
            stage_name: str,
        ) -> tuple[LLMConfig, dict[str, Any]]:
            unavailable_reason = (
                llm_backend_unavailable_reason(requested)
                if resolve_target_runtime_availability
                else ""
            )
            backend = str(getattr(requested.backend, "value", requested.backend) or "")
            model = str(requested.model or "")
            routing = {
                "purpose": f"planning_rollout_profile:{profile_index}:{stage_name}",
                "fallback_applied": False,
                "fallback_kind": "",
                "requested_unavailable_reason": unavailable_reason,
                "requested_backend": backend,
                "requested_model": model,
                "resolved_backend": backend,
                "resolved_model": model,
                "resolved_fingerprint": llm_backend_fingerprint(requested),
            }
            if not resolve_target_runtime_availability:
                routing["requested_unavailable_reason"] = ""
            return requested, routing

        descriptors: list[dict[str, Any]] = []
        for profile_index in range(len(self.config.rollout.llm_profiles)):
            planned_rollout_llm = self.config.get_llm_for_rollout_profile(profile_index)
            rollout_llm, rollout_routing = _resolved_descriptor_llm(
                planned_rollout_llm,
                profile_index=profile_index,
                stage_name="rollout",
            )
            planned_stage_llms = {
                stage_name: self.config.get_llm_for_profile_stage(profile_index, stage_name)
                for stage_name in ROLLOUT_PROFILE_STAGE_ORDER
            }
            stage_routings: dict[str, dict[str, Any]] = {}
            stage_llms: dict[str, LLMConfig] = {}
            for stage_name, planned_llm in planned_stage_llms.items():
                resolved_llm, routing = _resolved_descriptor_llm(
                    planned_llm,
                    profile_index=profile_index,
                    stage_name=stage_name,
                )
                stage_llms[stage_name] = resolved_llm
                stage_routings[stage_name] = dict(routing)
            stage_families = {
                stage_name: _llm_route_family(llm_config)
                for stage_name, llm_config in stage_llms.items()
            }
            family_values = {
                stage_families[stage_name] for stage_name in ROLLOUT_PROFILE_STAGE_ORDER
            }
            family_values.add(_llm_route_family(rollout_llm))
            search_family_counts = Counter(
                stage_families[stage_name] for stage_name in _PROFILE_SEARCH_STAGE_NAMES
            )
            search_family = (
                search_family_counts.most_common(1)[0][0]
                if search_family_counts
                else _llm_route_family(rollout_llm)
            )
            patch_family = stage_families["patcher"]
            rollout_family = _llm_route_family(rollout_llm)
            is_pure = len(family_values) == 1
            family_key = (
                rollout_family if is_pure else f"{search_family}_search_{patch_family}_patch"
            )
            descriptors.append(
                {
                    "profile_index": profile_index,
                    "rollout_family": rollout_family,
                    "search_family": search_family,
                    "patch_family": patch_family,
                    "stage_families": dict(stage_families),
                    "stage_models": tuple(
                        stage_llms[stage_name].model or ""
                        for stage_name in ROLLOUT_PROFILE_STAGE_ORDER
                    ),
                    "planned_stage_models": tuple(
                        planned_stage_llms[stage_name].model or ""
                        for stage_name in ROLLOUT_PROFILE_STAGE_ORDER
                    ),
                    "availability_fallback_applied": bool(
                        rollout_routing.get("fallback_applied")
                        or any(
                            bool(routing.get("fallback_applied"))
                            for routing in stage_routings.values()
                        )
                    ),
                    "availability_fallback_reasons": {
                        stage_name: str(
                            routing.get("requested_unavailable_reason")
                            or routing.get("fallback_kind")
                            or ""
                        )
                        for stage_name, routing in {
                            "rollout": dict(rollout_routing),
                            **stage_routings,
                        }.items()
                        if bool(routing.get("fallback_applied"))
                        or str(routing.get("requested_unavailable_reason") or "").strip()
                    },
                    "profile_kind": "pure" if is_pure else "hybrid",
                    "family_key": family_key,
                    "is_pure": is_pure,
                }
            )
        return descriptors

    def _score_pure_rollout_profile(
        self,
        *,
        descriptor: dict[str, Any],
        brief: RolloutBrief,
        issue_description: str,
        relevant_files: list[str],
        test_files: list[str],
        repo_context: RepoContext,
        used_families: set[str],
        profile_usage: Counter[int],
        task_regime: Optional[TaskRegimeProfile] = None,
    ) -> float:
        mode = str((brief.search_policy or {}).get("mode") or "").strip().lower()
        family = str(descriptor.get("rollout_family") or "")
        score = 0.0
        if family not in used_families:
            score += 1.0
        score += {
            "codex": 0.45,
            "claude": 0.40,
            "gemini": 0.28,
            "meta": 0.18,
        }.get(family, 0.0)

        if mode == "surgical":
            score += {"codex": 1.45, "claude": 0.95, "gemini": 0.55, "meta": 0.45}.get(
                family,
                0.0,
            )
        elif mode == "test_rooted":
            score += {"gemini": 1.20, "claude": 0.95, "codex": 0.85, "meta": 0.65}.get(
                family,
                0.0,
            )
        elif mode == "dependency_trace":
            score += {"gemini": 1.00, "meta": 0.95, "codex": 0.85, "claude": 0.72}.get(
                family,
                0.0,
            )
        elif mode == "invariant_guard":
            score += {"claude": 1.35, "codex": 0.82, "gemini": 0.58, "meta": 0.52}.get(
                family,
                0.0,
            )
        elif mode == "api_contract":
            score += {"claude": 1.45, "meta": 0.88, "codex": 0.72, "gemini": 0.68}.get(
                family,
                0.0,
            )
        elif mode == "source_cluster":
            score += {"meta": 1.00, "gemini": 0.92, "codex": 0.75, "claude": 0.70}.get(
                family,
                0.0,
            )
        elif mode == "agentless_pipeline":
            score += {"codex": 1.10, "claude": 0.90, "gemini": 0.75, "meta": 0.45}.get(
                family,
                0.0,
            )

        completion_like = self._task_regime_probability(task_regime, "contract_gap") >= 0.5
        public_api_task = self._task_regime_probability(task_regime, "high_interface_risk") >= 0.45
        broad_context = len(relevant_files) >= 6 or len(repo_context.files) >= 120
        search_heavy = len(test_files) >= 2 or len(relevant_files) >= 5
        if (completion_like or public_api_task) and family == "claude":
            score += 0.60
        if broad_context and family == "meta":
            score += 0.25
        if search_heavy and family == "gemini":
            score += 0.25
        score -= 0.45 * float(profile_usage[int(descriptor["profile_index"])])
        return score

    def _score_hybrid_rollout_profile(
        self,
        *,
        descriptor: dict[str, Any],
        brief: RolloutBrief,
        issue_description: str,
        relevant_files: list[str],
        test_files: list[str],
        repo_context: RepoContext,
        used_profile_keys: set[str],
        profile_usage: Counter[int],
        task_regime: Optional[TaskRegimeProfile] = None,
    ) -> float:
        mode = str((brief.search_policy or {}).get("mode") or "").strip().lower()
        search_family = str(descriptor.get("search_family") or "")
        patch_family = str(descriptor.get("patch_family") or "")
        rollout_family = str(descriptor.get("rollout_family") or "")
        family_key = str(descriptor.get("family_key") or "")
        score = 0.6
        if family_key not in used_profile_keys:
            score += 0.65
        if rollout_family == patch_family:
            score += 0.20
        if patch_family in {"codex", "claude"}:
            score += 0.55
        if search_family in {"gemini", "meta"}:
            score += 0.45
        if search_family != patch_family:
            score += 0.25

        if mode == "surgical":
            if patch_family == "codex":
                score += 1.15
            if search_family in {"claude", "gemini"}:
                score += 0.20
        elif mode == "test_rooted":
            if search_family == "gemini":
                score += 0.90
            if patch_family == "claude":
                score += 0.72
            if patch_family == "codex":
                score += 0.45
        elif mode == "dependency_trace":
            if search_family == "meta":
                score += 0.86
            if search_family == "gemini":
                score += 0.82
            if patch_family == "codex":
                score += 0.55
            if patch_family == "claude":
                score += 0.35
        elif mode == "invariant_guard":
            if patch_family == "claude":
                score += 0.98
            if search_family in {"gemini", "meta"}:
                score += 0.25
        elif mode == "api_contract":
            if patch_family == "claude":
                score += 1.08
            if search_family == "meta":
                score += 0.45
            if search_family == "gemini":
                score += 0.30
        elif mode == "source_cluster":
            if search_family == "meta":
                score += 0.82
            if search_family == "gemini":
                score += 0.72
            if patch_family == "codex":
                score += 0.45
        elif mode == "agentless_pipeline":
            if patch_family == "codex":
                score += 0.88
            if patch_family == "claude":
                score += 0.65
            if search_family in {"gemini", "meta"}:
                score += 0.25

        completion_like = self._task_regime_probability(task_regime, "contract_gap") >= 0.5
        public_api_task = self._task_regime_probability(task_regime, "high_interface_risk") >= 0.45
        broad_context = len(relevant_files) >= 6 or len(repo_context.files) >= 120
        search_heavy = len(test_files) >= 2 or len(relevant_files) >= 5
        if (completion_like or public_api_task) and patch_family == "claude":
            score += 0.52
        if broad_context and search_family == "meta":
            score += 0.25
        if search_heavy and search_family == "gemini":
            score += 0.25
        if search_family == "codex" and patch_family == "claude":
            score += 0.10 if mode in {"surgical", "invariant_guard"} else 0.0
        if search_family == "claude" and patch_family == "codex":
            score += 0.10 if mode in {"surgical", "api_contract"} else 0.0
        score -= 0.35 * float(profile_usage[int(descriptor["profile_index"])])
        return score

    def _select_best_rollout_profile(
        self,
        *,
        candidates: list[dict[str, Any]],
        brief: RolloutBrief,
        issue_description: str,
        relevant_files: list[str],
        test_files: list[str],
        repo_context: RepoContext,
        used_families: set[str],
        used_profile_keys: set[str],
        profile_usage: Counter[int],
        task_regime: Optional[TaskRegimeProfile] = None,
    ) -> Optional[dict[str, Any]]:
        if not candidates:
            return None

        action = brief.resolved_controller_action()

        def _score(descriptor: dict[str, Any]) -> tuple[float, int]:
            if bool(descriptor.get("is_pure")):
                heuristic_value = self._score_pure_rollout_profile(
                    descriptor=descriptor,
                    brief=brief,
                    issue_description=issue_description,
                    relevant_files=relevant_files,
                    test_files=test_files,
                    repo_context=repo_context,
                    used_families=used_families,
                    profile_usage=profile_usage,
                    task_regime=task_regime,
                )
            else:
                heuristic_value = self._score_hybrid_rollout_profile(
                    descriptor=descriptor,
                    brief=brief,
                    issue_description=issue_description,
                    relevant_files=relevant_files,
                    test_files=test_files,
                    repo_context=repo_context,
                    used_profile_keys=used_profile_keys,
                    profile_usage=profile_usage,
                    task_regime=task_regime,
                )
            evaluation = evaluate_policy_model(
                getattr(self.config, "controller_models", None),
                model_name="planning.rollout_profile_score",
                features={
                    "heuristic_score": float(heuristic_value),
                    "is_pure_profile": 1.0 if bool(descriptor.get("is_pure")) else 0.0,
                    "relevant_file_count": float(len(relevant_files)),
                    "test_file_count": float(len(test_files)),
                    "repo_file_count": float(len(repo_context.files)),
                    "action_symbol_count": float(len(list(action.symbols or []))),
                    "action_edit_span_count": float(len(list(action.edit_spans or []))),
                    "contract_gap_probability": float(
                        self._task_regime_probability(task_regime, "contract_gap")
                    ),
                    "interface_probability": float(
                        self._task_regime_probability(task_regime, "high_interface_risk")
                    ),
                    "search_family_is_meta": (
                        1.0 if str(descriptor.get("search_family") or "") == "meta" else 0.0
                    ),
                    "search_family_is_gemini": (
                        1.0 if str(descriptor.get("search_family") or "") == "gemini" else 0.0
                    ),
                    "patch_family_is_claude": (
                        1.0 if str(descriptor.get("patch_family") or "") == "claude" else 0.0
                    ),
                    "patch_family_is_codex": (
                        1.0 if str(descriptor.get("patch_family") or "") == "codex" else 0.0
                    ),
                },
                baseline_value=float(heuristic_value),
            )
            value = float(evaluation.value or heuristic_value)
            return (value, -int(descriptor["profile_index"]))

        return max(candidates, key=_score)

    def _apply_rollout_profile_assignment(
        self,
        *,
        brief: RolloutBrief,
        rollout_index: int,
        descriptor: dict[str, Any],
    ) -> None:
        profile_index = int(descriptor["profile_index"])
        prompt_strategy = self.config.get_prompt_strategy_for_rollout(rollout_index)
        temperature = self.config.get_temperature_for_rollout(rollout_index)
        stage_model_signature = list(
            descriptor.get("stage_models")
            or self.config.get_rollout_stage_model_signature_for_profile(profile_index)
        )
        planned_stage_model_signature = list(
            descriptor.get("planned_stage_models") or stage_model_signature
        )

        def _descriptor_signature(
            *,
            include_prompt_strategy: bool,
            include_temperature: bool,
        ) -> list[str]:
            signature = list(stage_model_signature)
            if include_prompt_strategy:
                signature.append(prompt_strategy.value)
            if include_temperature:
                signature.append(f"{float(temperature):.2f}")
            return signature

        policy = dict(brief.search_policy or {})
        policy.pop("allocator_arm", None)
        action = ControllerAction.from_dict(brief.resolved_controller_action().to_dict())
        action.allocator_arm = ""
        policy.update(
            {
                "rollout_profile_index": profile_index,
                "rollout_profile_kind": descriptor["profile_kind"],
                "rollout_profile_family": descriptor["family_key"],
                "rollout_profile_rollout_family": descriptor["rollout_family"],
                "rollout_profile_search_family": descriptor["search_family"],
                "rollout_profile_patch_family": descriptor["patch_family"],
                "rollout_stage_model_signature": stage_model_signature,
                "rollout_profile_planned_stage_model_signature": planned_stage_model_signature,
                "rollout_profile_availability_fallback_applied": bool(
                    descriptor.get("availability_fallback_applied")
                ),
                "rollout_profile_availability_fallback_reasons": dict(
                    descriptor.get("availability_fallback_reasons") or {}
                ),
                "rollout_profile_signature": _descriptor_signature(
                    include_prompt_strategy=bool(
                        self.config.rollout.portfolio_diversity_include_prompt_strategy
                    ),
                    include_temperature=bool(
                        self.config.rollout.portfolio_diversity_include_temperature
                    ),
                ),
                "rollout_route_signature": _descriptor_signature(
                    include_prompt_strategy=True,
                    include_temperature=True,
                ),
                "rollout_prompt_strategy": prompt_strategy.value,
                "rollout_temperature": float(temperature),
            }
        )
        brief.set_controller_action(action, merge_policy=policy)
        brief.search_policy["allocator_arm"] = _rollout_brief_allocation_key(brief)
        brief.controller_action.allocator_arm = brief.search_policy["allocator_arm"]
        brief.search_policy = brief.controller_action.to_search_policy(base=brief.search_policy)

    def _determine_rollout_profile_seed_budget(
        self,
        *,
        briefs: list[RolloutBrief],
        issue_description: str,
        repo_context: RepoContext,
        relevant_files: list[str],
        test_files: list[str],
        pure_descriptors: list[dict[str, Any]],
        hybrid_descriptors: list[dict[str, Any]],
        task_regime: Optional[TaskRegimeProfile] = None,
    ) -> int:
        configured_seed_budget = int(self.config.rollout.portfolio_seed_profile_count or 0)
        max_seed_budget = (
            min(len(briefs), configured_seed_budget)
            if configured_seed_budget > 0
            else min(len(briefs), len(pure_descriptors) + len(hybrid_descriptors))
        )
        if max_seed_budget <= 0:
            return 0

        pure_floor = min(max_seed_budget, len(pure_descriptors))
        if pure_floor >= max_seed_budget or not hybrid_descriptors:
            return max_seed_budget

        completion_like = self._task_regime_probability(task_regime, "contract_gap") >= 0.5
        public_api_task = self._task_regime_probability(task_regime, "high_interface_risk") >= 0.45
        search_heavy = len(test_files) >= 2 or len(relevant_files) >= 5
        broad_context = len(repo_context.files) >= 120 or len(relevant_files) >= 6
        very_broad_context = (
            len(repo_context.files) >= 220 or len(relevant_files) >= 8 or len(test_files) >= 4
        )

        hybrid_budget = 1
        if search_heavy:
            hybrid_budget += 1
        if broad_context or completion_like or public_api_task:
            hybrid_budget += 1
        if very_broad_context and (search_heavy or completion_like or public_api_task):
            hybrid_budget += 1

        heuristic_budget = pure_floor + min(
            hybrid_budget,
            max_seed_budget - pure_floor,
            len(hybrid_descriptors),
        )
        evaluation = evaluate_policy_model(
            getattr(self.config, "controller_models", None),
            model_name="planning.profile_seed_budget",
            features={
                "heuristic_score": float(heuristic_budget),
                "brief_count": float(len(briefs)),
                "relevant_file_count": float(len(relevant_files)),
                "test_file_count": float(len(test_files)),
                "repo_file_count": float(len(repo_context.files)),
                "completion_like": 1.0 if completion_like else 0.0,
                "public_api_task": 1.0 if public_api_task else 0.0,
                "search_heavy": 1.0 if search_heavy else 0.0,
                "broad_context": 1.0 if broad_context else 0.0,
                "very_broad_context": 1.0 if very_broad_context else 0.0,
                "max_seed_budget": float(max_seed_budget),
                "pure_floor": float(pure_floor),
            },
            baseline_value=float(heuristic_budget),
            lower=0.0,
            upper=float(max_seed_budget),
        )
        calibrated_budget = int(round(float(evaluation.value or heuristic_budget)))
        return max(0, min(max_seed_budget, calibrated_budget))

    def _assign_rollout_profiles(
        self,
        *,
        briefs: list[RolloutBrief],
        issue_description: str,
        repo_context: RepoContext,
        relevant_files: list[str],
        test_files: list[str],
        task_regime: Optional[TaskRegimeProfile] = None,
    ) -> None:
        descriptors = self._build_rollout_profile_descriptors()
        if not descriptors or not briefs:
            return

        pure_descriptors = [item for item in descriptors if bool(item.get("is_pure"))]
        hybrid_descriptors = [item for item in descriptors if not bool(item.get("is_pure"))]
        seed_budget = self._determine_rollout_profile_seed_budget(
            briefs=briefs,
            issue_description=issue_description,
            repo_context=repo_context,
            relevant_files=relevant_files,
            test_files=test_files,
            pure_descriptors=pure_descriptors,
            hybrid_descriptors=hybrid_descriptors,
            task_regime=task_regime,
        )
        pure_phase_count = min(seed_budget, len(pure_descriptors))

        used_pure_families: set[str] = set()
        used_hybrid_keys: set[str] = set()
        profile_usage: Counter[int] = Counter()
        seeded_profile_indices: set[int] = set()

        for rollout_index, brief in enumerate(briefs):
            candidates: list[dict[str, Any]]
            if rollout_index < pure_phase_count and pure_descriptors:
                candidates = [
                    descriptor
                    for descriptor in pure_descriptors
                    if str(descriptor.get("rollout_family") or "") not in used_pure_families
                ]
                if not candidates:
                    candidates = [
                        descriptor
                        for descriptor in pure_descriptors
                        if int(descriptor["profile_index"]) not in seeded_profile_indices
                    ]
                if not candidates:
                    candidates = list(pure_descriptors)
            elif rollout_index < seed_budget and hybrid_descriptors:
                candidates = [
                    descriptor
                    for descriptor in hybrid_descriptors
                    if int(descriptor["profile_index"]) not in seeded_profile_indices
                ]
                if not candidates:
                    candidates = list(hybrid_descriptors)
            else:
                candidates = list(descriptors)

            descriptor = self._select_best_rollout_profile(
                candidates=candidates,
                brief=brief,
                issue_description=issue_description,
                relevant_files=relevant_files,
                test_files=test_files,
                repo_context=repo_context,
                used_families=used_pure_families,
                used_profile_keys=used_hybrid_keys,
                profile_usage=profile_usage,
                task_regime=task_regime,
            )
            if descriptor is None:
                continue
            self._apply_rollout_profile_assignment(
                brief=brief,
                rollout_index=rollout_index,
                descriptor=descriptor,
            )
            profile_index = int(descriptor["profile_index"])
            profile_usage[profile_index] += 1
            if rollout_index < seed_budget:
                seeded_profile_indices.add(profile_index)
            if bool(descriptor.get("is_pure")):
                used_pure_families.add(str(descriptor.get("rollout_family") or ""))
            else:
                used_hybrid_keys.add(str(descriptor.get("family_key") or ""))

    def _brief_overlap_focus_signature(
        self,
        issue_plan: IssuePlan,
        brief: RolloutBrief,
    ) -> tuple[set[str], set[str]]:
        test_context = issue_plan.test_context
        search_policy = brief.search_policy if isinstance(brief.search_policy, dict) else {}
        source_focus = _dedupe_preserve(
            [
                str(path).strip()
                for path in (
                    list(brief.focus_files or [])
                    + list(test_context.source_focus_files or [])
                    + list(test_context.terminal_source_files or [])
                    + list(issue_plan.risk_files or [])
                )
                if str(path).strip() and not self._looks_like_test_path(str(path))
            ]
        )[:8]
        test_focus = _dedupe_preserve(
            [
                _strip_pytest_node_id(path)
                for path in (
                    [
                        str(path).strip()
                        for path in list(brief.focus_files or [])
                        if str(path).strip() and self._looks_like_test_path(str(path))
                    ]
                    + list(search_policy.get("graph_target_test_ids") or [])
                    + list(test_context.focus_test_files or [])
                    + list(test_context.failing_test_ids or [])
                )
                if _strip_pytest_node_id(path)
            ]
        )[:4]
        return set(source_focus), set(test_focus)

    def _brief_is_overlap_sensitive(
        self,
        issue_plan: IssuePlan,
        brief: RolloutBrief,
    ) -> bool:
        search_policy = self._normalize_brief_search_policy(brief)
        action = brief.resolved_controller_action()
        origin_state = (
            str(action.regime_state or search_policy.get("origin_regime_state") or "")
            .strip()
            .lower()
        )
        if origin_state in {"importability_blocker", "broad_regression"}:
            return True
        if self._task_regime_probability(
            issue_plan.task_regime, "importability_blocker"
        ) >= self.regime_policy.threshold("importability_blocker"):
            return True
        test_context = issue_plan.test_context
        if (
            self._task_regime_probability(issue_plan.task_regime, "broad_regression")
            >= self.regime_policy.threshold("broad_regression")
            and test_context.failing_test_count >= 3
        ):
            return True
        return False

    def _brief_overlap_variant_key(
        self,
        brief: RolloutBrief,
    ) -> tuple[Any, ...]:
        search_policy = self._normalize_brief_search_policy(brief)
        family_label = str(brief.title or "").strip().lower()
        if not family_label:
            family_label = str(search_policy.get("graph_target_family") or "").strip().lower()
        return (
            str(search_policy.get("mode") or "").strip().lower() or "surgical",
            family_label or "untitled",
            str(search_policy.get("origin_regime_state") or "").strip().lower(),
            str(search_policy.get("verification_focus") or "").strip().lower(),
            brief.delegation_enabled("patcher"),
        )

    def _focus_signatures_overlap(
        self,
        left_sources: set[str],
        left_tests: set[str],
        right_sources: set[str],
        right_tests: set[str],
    ) -> bool:
        overlap_policy = getattr(self.config.rollout, "overlap_policy", None)
        source_overlap = _jaccard_similarity(left_sources, right_sources)
        test_overlap = _jaccard_similarity(left_tests, right_tests)
        combined_overlap = (0.75 * source_overlap) + (0.25 * test_overlap)
        return (
            source_overlap >= float(getattr(overlap_policy, "source_overlap_threshold", 0.7) or 0.7)
            or test_overlap >= float(getattr(overlap_policy, "test_overlap_threshold", 0.6) or 0.6)
            or combined_overlap
            >= float(getattr(overlap_policy, "combined_overlap_threshold", 0.68) or 0.68)
        )

    def _prune_redundant_overlap_sensitive_rollout_variants(
        self,
        issue_plan: IssuePlan,
    ) -> None:
        briefs = list(issue_plan.rollout_briefs or [])
        if len(briefs) < 2:
            return

        retained: list[RolloutBrief] = []
        seen_signatures: dict[tuple[Any, ...], list[tuple[set[str], set[str]]]] = {}
        pruned_count = 0
        for brief in briefs:
            brief.search_policy = self._normalize_brief_search_policy(brief)
            if not self._brief_is_overlap_sensitive(issue_plan, brief):
                retained.append(brief)
                continue

            variant_key = self._brief_overlap_variant_key(brief)
            source_focus, test_focus = self._brief_overlap_focus_signature(issue_plan, brief)
            prior_signatures = seen_signatures.setdefault(variant_key, [])
            if any(
                self._focus_signatures_overlap(
                    source_focus,
                    test_focus,
                    prior_sources,
                    prior_tests,
                )
                for prior_sources, prior_tests in prior_signatures
            ):
                pruned_count += 1
                continue

            prior_signatures.append((source_focus, test_focus))
            retained.append(brief)

        if pruned_count <= 0:
            return

        issue_plan.rollout_briefs = retained
        issue_plan.planner_metadata = dict(issue_plan.planner_metadata or {})
        issue_plan.planner_metadata["redundant_overlap_pruned_count"] = (
            int(issue_plan.planner_metadata.get("redundant_overlap_pruned_count") or 0)
            + pruned_count
        )
        issue_plan.planner_metadata["retained_rollout_count"] = len(retained)

    def _regime_candidate_rollout_briefs(
        self,
        issue_plan: IssuePlan,
        *,
        focus_tests: list[str],
        incomplete_tests: list[str],
        incomplete_sources: list[str],
        terminal_sources: list[str],
        source_focus: list[str],
        completion_neighbor_focus: list[str],
    ) -> list[RolloutBrief]:
        regime = issue_plan.task_regime
        candidates: list[RolloutBrief] = []
        importability_probability = self._task_regime_probability(
            regime,
            "importability_blocker",
        )
        contract_gap_probability = self._task_regime_probability(regime, "contract_gap")
        broad_regression_probability = self._task_regime_probability(
            regime,
            "broad_regression",
        )
        interface_probability = self._task_regime_probability(
            regime,
            "high_interface_risk",
        )
        failing_focus = focus_tests[:1]

        if importability_probability >= self.regime_policy.threshold("importability_blocker") and (
            terminal_sources or source_focus
        ):
            direct_focus = _dedupe_preserve(
                terminal_sources
                + incomplete_sources
                + source_focus
                + completion_neighbor_focus
                + failing_focus
                + list(issue_plan.relevant_files)
            )[:8]
            success = _dedupe_preserve(
                [
                    "Restore clean import and collection along the currently observed failing path.",
                    "Rerun the visible suite after the current blocker clears before broadening the patch.",
                ]
                + list(issue_plan.success_criteria)
            )[:6]
            candidates.extend(
                [
                    RolloutBrief(
                        title="Direct blocker elimination",
                        goal=(
                            "Clear the direct import or collection blocker in the terminal traceback "
                            "region before widening the patch."
                        ),
                        focus_files=direct_focus,
                        hypotheses=[
                            "The immediate next change is in the terminal traceback file or its direct import neighbors.",
                            "Importability must recover before broader contract work can be trusted.",
                        ],
                        success_criteria=success,
                        prompt_hint=(
                            "Avoid a bottom-up repository sweep. Start with the terminal traceback region "
                            "and nearest import edges, then rerun validation before widening the patch."
                        ),
                        agent_mode=issue_plan.rollout_briefs[0].agent_mode
                        if issue_plan.rollout_briefs
                        else AgentMode.ADAPTIVE,
                        search_policy={
                            "mode": "test_rooted",
                            "verification_focus": "failing_tests",
                            "origin": "regime_candidate",
                            "origin_regime_state": "importability_blocker",
                        },
                    ),
                    RolloutBrief(
                        title="Import-chain continuation",
                        goal=(
                            "Use the next visible import edge after each fix to continue repairing adjacent "
                            "modules without turning the patch into a repo-wide sweep."
                        ),
                        focus_files=_dedupe_preserve(completion_neighbor_focus + direct_focus)[:8],
                        hypotheses=[
                            "Adjacent imported modules may reveal the next missing contract after the current blocker clears.",
                            "Keeping the patch on the same interface chain is safer than broad early edits.",
                        ],
                        success_criteria=success,
                        prompt_hint=(
                            "Treat the import chain as an ordinary candidate family and advance one surfaced blocker "
                            "at a time."
                        ),
                        agent_mode=issue_plan.rollout_briefs[0].agent_mode
                        if issue_plan.rollout_briefs
                        else AgentMode.ADAPTIVE,
                        search_policy={
                            "mode": "dependency_trace",
                            "verification_focus": "failing_tests",
                            "origin": "regime_candidate",
                            "origin_regime_state": "importability_blocker",
                        },
                    ),
                ]
            )

        if contract_gap_probability >= self.regime_policy.threshold("contract_gap") and (
            incomplete_sources or focus_tests or incomplete_tests
        ):
            candidates.append(
                RolloutBrief(
                    title="Contract recovery",
                    goal=(
                        "Recover missing behavior from visible tests, nearby incomplete scaffolds, and public "
                        "entrypoints without weakening the contract."
                    ),
                    focus_files=_dedupe_preserve(
                        incomplete_sources
                        + terminal_sources
                        + focus_tests
                        + incomplete_tests[:1]
                        + completion_neighbor_focus
                        + list(issue_plan.relevant_files)
                    )[:8],
                    hypotheses=[
                        "The task is better explained as a contract gap than as a one-line regression.",
                        "Visible tests and nearby scaffolds reveal the missing behavior to complete.",
                    ],
                    success_criteria=_dedupe_preserve(
                        [
                            "Preserve the visible behavioral contract while completing missing functionality.",
                        ]
                        + list(issue_plan.success_criteria)
                    )[:6],
                    prompt_hint="Infer the missing contract from visible evidence before broad code cleanup.",
                    agent_mode=issue_plan.rollout_briefs[0].agent_mode
                    if issue_plan.rollout_briefs
                    else AgentMode.ADAPTIVE,
                    search_policy={
                        "mode": "api_contract",
                        "verification_focus": "failing_tests"
                        if focus_tests
                        else "focus_test_files",
                        "origin": "regime_candidate",
                        "origin_regime_state": "contract_gap",
                    },
                )
            )

        if broad_regression_probability >= self.regime_policy.threshold("broad_regression"):
            candidates.append(
                RolloutBrief(
                    title="Regression surface stabilization",
                    goal=(
                        "Repair the issue while preserving nearby behavior and collected visible coverage across "
                        "the broader failure surface."
                    ),
                    focus_files=_dedupe_preserve(
                        terminal_sources
                        + source_focus
                        + focus_tests
                        + list(issue_plan.relevant_files)
                    )[:8],
                    hypotheses=[
                        "The failure surface is broad enough that regression containment matters as much as the first fix.",
                    ],
                    success_criteria=_dedupe_preserve(
                        [
                            "Maintain collected visible coverage while improving the failing surface.",
                        ]
                        + list(issue_plan.success_criteria)
                    )[:6],
                    prompt_hint="Treat regressions and coverage shrinkage as first-class risks while fixing the bug.",
                    agent_mode=issue_plan.rollout_briefs[0].agent_mode
                    if issue_plan.rollout_briefs
                    else AgentMode.ADAPTIVE,
                    search_policy={
                        "mode": "invariant_guard",
                        "verification_focus": "failing_tests"
                        if focus_tests
                        else "focus_test_files",
                        "origin": "regime_candidate",
                        "origin_regime_state": "broad_regression",
                    },
                )
            )

        if interface_probability >= self.regime_policy.threshold("high_interface_risk"):
            candidates.append(
                RolloutBrief(
                    title="Interface boundary stabilization",
                    goal=(
                        "Follow the shared symbol and module boundaries around the failing region so the fix "
                        "preserves interface compatibility while changing implementation details."
                    ),
                    focus_files=_dedupe_preserve(
                        source_focus
                        + terminal_sources
                        + completion_neighbor_focus
                        + list(issue_plan.relevant_files)
                    )[:8],
                    hypotheses=[
                        "The likely edits cross shared interface boundaries and need contract-aware coordination.",
                    ],
                    success_criteria=_dedupe_preserve(
                        [
                            "Preserve shared interface behavior across touched modules.",
                        ]
                        + list(issue_plan.success_criteria)
                    )[:6],
                    prompt_hint="Reason in terms of interface boundaries and shared symbols, not only file breadth.",
                    agent_mode=issue_plan.rollout_briefs[0].agent_mode
                    if issue_plan.rollout_briefs
                    else AgentMode.ADAPTIVE,
                    search_policy={
                        "mode": "dependency_trace",
                        "verification_focus": "failing_tests"
                        if focus_tests
                        else "focus_test_files",
                        "origin": "regime_candidate",
                        "origin_regime_state": "high_interface_risk",
                    },
                )
            )
        return candidates

    def _rollout_brief_candidate_option_id(self, brief: RolloutBrief) -> str:
        policy = self._normalize_brief_search_policy(brief)
        action = brief.resolved_controller_action()
        origin = str(action.origin or policy.get("origin") or "heuristic").strip().lower()
        state = str(action.regime_state or policy.get("origin_regime_state") or "").strip().lower()
        family_index = policy.get("family_index")
        variant_index = policy.get("variant_index")
        focus = ",".join(
            _dedupe_preserve(list(action.file_paths or []) + list(brief.focus_files or []))[:4]
        )
        symbols = ",".join(list(action.symbols or [])[:3])
        return "|".join(
            fragment
            for fragment in (
                origin,
                state,
                str(action.mode or policy.get("mode") or "").strip().lower(),
                str(brief.title or "").strip(),
                f"family={family_index}" if isinstance(family_index, int) else "",
                f"variant={variant_index}" if isinstance(variant_index, int) else "",
                focus,
                symbols,
            )
            if fragment
        )

    def _score_rollout_brief_candidate(
        self,
        issue_plan: IssuePlan,
        brief: RolloutBrief,
    ) -> tuple[float, str, dict[str, Any]]:
        policy = self._normalize_brief_search_policy(brief)
        action = brief.resolved_controller_action()
        mode = str(action.mode or policy.get("mode") or "surgical").strip().lower()
        regime_state = (
            str(action.regime_state or policy.get("origin_regime_state") or "").strip().lower()
        )
        regime = issue_plan.task_regime
        focus_files = _dedupe_preserve(
            list(action.file_paths or []) + list(brief.focus_files or [])
        )
        terminal_sources = _dedupe_preserve(
            list(issue_plan.test_context.terminal_source_files or [])
        )
        source_focus = _dedupe_preserve(list(issue_plan.test_context.source_focus_files or []))
        focus_tests = _dedupe_preserve(list(issue_plan.test_context.focus_test_files or []))
        failing_tests = _dedupe_preserve(list(issue_plan.test_context.failing_test_ids or []))
        score = 0.35
        reasons: list[str] = []
        objective_collection_blocker = self._issue_plan_has_objective_collection_blocker(
            issue_plan
        )
        direct_collection_unblock = self._brief_targets_direct_collection_unblock(
            brief,
            policy=policy,
            regime_state=regime_state,
        )

        focus_overlap = len(set(focus_files).intersection(set(terminal_sources + source_focus)))
        if focus_overlap > 0:
            score += 0.12 + (0.05 * min(focus_overlap, 3))
            reasons.append("matches current source focus")
        test_overlap = len(
            {
                _strip_pytest_node_id(path) for path in focus_files if _strip_pytest_node_id(path)
            }.intersection(
                {
                    _strip_pytest_node_id(path)
                    for path in (focus_tests + failing_tests)
                    if _strip_pytest_node_id(path)
                }
            )
        )
        if test_overlap > 0:
            score += 0.10 + (0.04 * min(test_overlap, 2))
            reasons.append("matches visible tests")
        if policy.get("origin") == "regime_candidate":
            score += 0.05
            reasons.append("originates from regime evidence")
        if regime_state == "importability_blocker":
            if mode == "test_rooted":
                score += 0.30
                reasons.append("starts from the sharpest importability blocker")
            elif mode == "dependency_trace":
                score += 0.10
                reasons.append("keeps recovery on the same dependency chain")
        if objective_collection_blocker:
            if direct_collection_unblock:
                score += 1.25
                reasons.append("prioritizes direct import/collection unblock")
                if brief.delegation_enabled("patcher"):
                    score -= 0.35
                    reasons.append(
                        "keeps direct collection/import unblock ahead of delegated breadth"
                    )
                else:
                    score += 0.45
                    reasons.append(
                        "uses a single repair worker for the direct collection/import unblock"
                    )
            else:
                score -= 0.45
                reasons.append("defers broad work until collection/import is restored")
        if bool(policy.get("planner_authored_subtasks")) and not objective_collection_blocker:
            score += 1.0
            reasons.append("preserves planner-authored delegation structure for evaluation")
        elif bool(policy.get("planner_authored_subtasks")) and direct_collection_unblock:
            score += 0.15
            reasons.append("keeps bounded planner subtasks after direct unblock alignment")
        if action.symbols:
            score += 0.03 * min(len(action.symbols), 4)
            reasons.append("tracks symbol-level interface focus")
        if action.edit_spans:
            score += 0.02 * min(len(action.edit_spans), 4)
            reasons.append("targets bounded edit spans")

        importability_probability = self._task_regime_probability(regime, "importability_blocker")
        contract_gap_probability = self._task_regime_probability(regime, "contract_gap")
        broad_regression_probability = self._task_regime_probability(regime, "broad_regression")
        interface_probability = self._task_regime_probability(regime, "high_interface_risk")

        if mode in {"test_rooted", "dependency_trace"}:
            delta = 0.22 * importability_probability
            score += delta
            if delta > 0:
                reasons.append("aligned with importability blocker regime")
        if mode in {"api_contract", "dependency_trace"}:
            delta = 0.20 * max(contract_gap_probability, interface_probability)
            score += delta
            if delta > 0:
                reasons.append("aligned with contract/interface regime")
        if mode == "invariant_guard":
            delta = 0.24 * broad_regression_probability
            score += delta
            if delta > 0:
                reasons.append("aligned with broad regression regime")
        if brief.delegation_enabled("patcher"):
            score += 0.04 * max(contract_gap_probability, interface_probability)
            if max(contract_gap_probability, interface_probability) > 0:
                reasons.append("can expose bounded delegation surface")
        score += min(len(focus_files), 6) * 0.01
        feature_view = {
            **self._brief_action_feature_view(issue_plan, brief),
            "focus_overlap": float(focus_overlap),
            "test_overlap": float(test_overlap),
            "planner_authored_subtasks": (
                1.0 if bool(policy.get("planner_authored_subtasks")) else 0.0
            ),
            "regime_state_is_importability_blocker": (
                1.0 if regime_state == "importability_blocker" else 0.0
            ),
            "mode_is_test_rooted": 1.0 if mode == "test_rooted" else 0.0,
            "mode_is_dependency_trace": 1.0 if mode == "dependency_trace" else 0.0,
            "mode_is_api_contract": 1.0 if mode == "api_contract" else 0.0,
            "mode_is_invariant_guard": 1.0 if mode == "invariant_guard" else 0.0,
            "heuristic_score": float(score),
        }
        evaluation = evaluate_policy_model(
            getattr(self.config, "controller_models", None),
            model_name="planning.rollout_brief_score",
            features=feature_view,
            baseline_value=score,
            lower=0.0,
        )
        calibrated_score = float(evaluation.value or score)
        return (
            calibrated_score,
            "; ".join(reasons) or "generic heuristic rollout family",
            evaluation.to_dict(),
        )

    def _issue_plan_has_objective_collection_blocker(self, issue_plan: IssuePlan) -> bool:
        planner_metadata = (
            issue_plan.planner_metadata if isinstance(issue_plan.planner_metadata, dict) else {}
        )
        if bool(planner_metadata.get("baseline_signal_has_collection_signature")):
            return True
        test_context = issue_plan.test_context
        if int(test_context.passing_test_count or 0) > 0:
            return False
        if any(_looks_like_collection_error_test_id(test_id) for test_id in test_context.failing_test_ids):
            return True
        return bool(
            test_context.exception_summaries
            and (test_context.terminal_source_files or test_context.source_focus_files)
        )

    def _brief_targets_direct_collection_unblock(
        self,
        brief: RolloutBrief,
        *,
        policy: Optional[dict[str, Any]] = None,
        regime_state: str = "",
    ) -> bool:
        search_policy = dict(policy if isinstance(policy, dict) else (brief.search_policy or {}))
        normalized_regime = str(
            regime_state or search_policy.get("origin_regime_state") or ""
        ).strip().lower()
        if normalized_regime == "importability_blocker" or bool(
            search_policy.get("collection_error_fast_path")
        ):
            return True
        mode = str(search_policy.get("mode") or "").strip().lower()
        if normalized_regime == "broad_regression" or mode == "invariant_guard":
            return False
        text = " ".join(
            [
                str(brief.title or ""),
                str(brief.goal or ""),
                str(brief.prompt_hint or ""),
                " ".join(str(item) for item in list(brief.hypotheses or [])),
                " ".join(str(item) for item in list(brief.success_criteria or [])),
            ]
        ).lower()
        return any(
            marker in text
            for marker in (
                "direct blocker",
                "direct import failure",
                "import baseline",
                "import root",
                "import unblock",
                "import failure",
                "import collection",
                "import surface",
                "import-chain",
                "import-time",
                "import-time root",
                "module import",
                "root import",
                "collection blocker",
                "collection unblock",
                "clear the direct",
                "first import",
            )
        )

    def _rank_rollout_brief_candidates(
        self,
        issue_plan: IssuePlan,
        candidates: list[RolloutBrief],
        *,
        limit: int,
    ) -> list[RolloutBrief]:
        if not candidates:
            return []
        remaining = [RolloutBrief.from_dict(brief.to_dict()) for brief in candidates]
        selected: list[RolloutBrief] = []
        planner_metadata = dict(issue_plan.planner_metadata or {})
        while remaining and len(selected) < max(1, int(limit)):
            options: list[tuple[RolloutBrief, ShadowPolicyOption]] = []
            seen_option_ids: set[str] = set()
            for brief in remaining:
                option_id = self._rollout_brief_candidate_option_id(brief)
                if option_id in seen_option_ids:
                    continue
                seen_option_ids.add(option_id)
                score, rationale, evaluation = self._score_rollout_brief_candidate(
                    issue_plan, brief
                )
                options.append(
                    (
                        brief,
                        ShadowPolicyOption(
                            option_id=option_id,
                            score=score,
                            rationale=rationale,
                            category="rollout_brief",
                            metadata={
                                "title": brief.title,
                                "mode": str((brief.search_policy or {}).get("mode") or ""),
                                "origin": str(
                                    (brief.search_policy or {}).get("origin") or "heuristic"
                                ),
                                "controller_action": brief.resolved_controller_action().to_dict(),
                                "policy_evaluation": evaluation,
                            },
                        ),
                    )
                )
            if not options:
                break
            chosen_brief, chosen_option = max(
                options,
                key=lambda item: (item[1].score, item[1].option_id),
            )
            chosen_option.selected = True
            self._append_shadow_policy_trace(
                planner_metadata,
                build_shadow_policy_trace(
                    decision=f"rollout_brief_rank_{len(selected) + 1}",
                    options=[option for _, option in options],
                    max_logged_options=self._shadow_policy_limit(),
                ),
            )
            append_controller_decision(
                self.config,
                stage="planning",
                decision_type="rollout_brief_rank",
                chosen_option=chosen_option.option_id,
                feature_view=self._brief_action_feature_view(issue_plan, chosen_brief),
                options=[option for _, option in options],
                metadata={
                    "rank_index": len(selected) + 1,
                    "controller_action": chosen_brief.resolved_controller_action().to_dict(),
                },
            )
            selected.append(chosen_brief)
            chosen_id = self._rollout_brief_candidate_option_id(chosen_brief)
            remaining = [
                brief
                for brief in remaining
                if self._rollout_brief_candidate_option_id(brief) != chosen_id
            ]
        issue_plan.planner_metadata = planner_metadata
        return selected or candidates[: max(1, int(limit))]

    def _retune_rollout_briefs_with_test_context(
        self,
        issue_plan: IssuePlan,
        repo_context: RepoContext,
        *,
        issue_description: str,
    ) -> None:
        if not isinstance(issue_plan.task_regime, TaskRegimeProfile) or (
            not issue_plan.task_regime.state_probabilities and not issue_plan.task_regime.evidence
        ):
            issue_plan.task_regime = self._infer_task_regime(
                issue_description=issue_description,
                repo_context=repo_context,
                relevant_files=list(issue_plan.relevant_files),
                baseline_result=None,
                test_context=issue_plan.test_context,
                evaluation_constraints=issue_plan.evaluation_constraints,
            )
            issue_plan.planner_metadata = dict(issue_plan.planner_metadata or {})
            issue_plan.planner_metadata.update(self._task_regime_metadata(issue_plan.task_regime))
        test_context = issue_plan.test_context
        objective_collection_blocker = self._issue_plan_has_objective_collection_blocker(
            issue_plan
        )
        completion_like = (
            self._task_regime_probability(issue_plan.task_regime, "contract_gap")
            >= self.regime_policy.threshold("contract_gap")
        ) or bool(test_context.incomplete_source_files or test_context.incomplete_test_files)
        focus_tests = list(test_context.focus_test_files[:2])
        incomplete_tests = list(test_context.incomplete_test_files[:2])
        incomplete_sources = list(test_context.incomplete_source_files[:3])
        terminal_sources = list(test_context.terminal_source_files[:2])
        source_focus = _dedupe_preserve(
            terminal_sources + incomplete_sources + list(test_context.source_focus_files[:3])
        )
        completion_neighbors = (
            self._completion_neighbor_focus_files(issue_plan, repo_context, test_context)
            if completion_like
            else []
        )
        completion_neighbor_focus = list(completion_neighbors[:4])
        original_brief_count = max(1, len(issue_plan.rollout_briefs or []))
        if completion_neighbor_focus:
            issue_plan.relevant_files = _dedupe_preserve(
                terminal_sources
                + incomplete_sources
                + completion_neighbor_focus
                + list(issue_plan.relevant_files)
            )[: self.config.planning.max_relevant_files]
            risk_files = _dedupe_preserve(
                terminal_sources
                + incomplete_sources
                + completion_neighbor_focus
                + list(issue_plan.risk_files)
            )
            if risk_files:
                issue_plan.risk_files = risk_files[: min(4, len(risk_files))]
            issue_plan.repo_focus_map = repo_context.build_context_pack(
                issue_plan.relevant_files[: self.config.planning.max_repo_map_files],
                max_symbols_per_file=8,
                seed_symbols=issue_plan.keywords
                + list(issue_plan.test_context.terminal_reference_symbols or []),
            )
            neighbor_summary = (
                "Completion-relevant import neighbors to inspect after the direct blocker clears: "
                + ", ".join(completion_neighbor_focus)
                + "."
            )
            if neighbor_summary not in test_context.summary:
                test_context.summary = f"{test_context.summary} {neighbor_summary}".strip()
        if completion_like:
            test_contract_focus = list(focus_tests[:2])
            scaffold_test_focus = list(incomplete_tests[:1])
            source_priority_tail = (
                completion_neighbor_focus + test_contract_focus + scaffold_test_focus
            )
            test_priority_tail = (
                completion_neighbor_focus + test_contract_focus + scaffold_test_focus
            )
        else:
            test_contract_focus = list(focus_tests)
            scaffold_test_focus = list(incomplete_tests)
            source_priority_tail = scaffold_test_focus + focus_tests[:1] + completion_neighbor_focus
            test_priority_tail = (
                scaffold_test_focus + test_contract_focus + completion_neighbor_focus
            )

        candidate_briefs = [
            RolloutBrief.from_dict(brief.to_dict())
            for brief in list(issue_plan.rollout_briefs or [])
        ]
        candidate_briefs.extend(
            self._regime_candidate_rollout_briefs(
                issue_plan,
                focus_tests=focus_tests,
                incomplete_tests=incomplete_tests,
                incomplete_sources=incomplete_sources,
                terminal_sources=terminal_sources,
                source_focus=source_focus,
                completion_neighbor_focus=completion_neighbor_focus,
            )
        )
        for brief in candidate_briefs:
            policy = self._normalize_brief_search_policy(brief)
            policy.setdefault("origin", "heuristic")
            if objective_collection_blocker and self._brief_targets_direct_collection_unblock(
                brief,
                policy=policy,
            ):
                policy["collection_error_fast_path"] = True
                policy["origin_regime_state"] = "importability_blocker"
            evidence_policy = issue_plan.evaluation_constraints.resolved_evidence_policy()
            policy["evidence_mode"] = evidence_policy.mode
            policy["visible_tests_completeness"] = evidence_policy.visible_tests_completeness
            if evidence_policy.mode == EVIDENCE_MODE_GOLD_SUITE_VISIBLE:
                policy["contract_source"] = "gold_visible_suite"
                policy["verification_focus"] = "gold_expected_suite"
            elif evidence_policy.mode == EVIDENCE_MODE_PARTIAL_SUITE_VISIBLE:
                policy["contract_source"] = "issue_plus_partial_visible_tests"
                policy["hidden_risk_review_required"] = True
            elif evidence_policy.mode == EVIDENCE_MODE_NO_SUITE_VISIBLE:
                policy["contract_source"] = "issue_and_generated_repro"
                policy.setdefault("verification_focus", "generated_repro")
            elif evidence_policy.mode == EVIDENCE_MODE_EVAL_ONLY_SUITE:
                policy["contract_source"] = "agent_visible_issue_and_repo_only"
            if isinstance(brief.delegation_policy, dict) and list(
                brief.delegation_policy.get("subtasks") or []
            ):
                policy["planner_authored_subtasks"] = True
            mode = policy.get("mode", "surgical")
            if (
                test_context.failing_test_ids
                and evidence_policy.mode != EVIDENCE_MODE_GOLD_SUITE_VISIBLE
            ):
                policy["verification_focus"] = "failing_tests"
            elif (
                not policy.get("verification_focus")
                or policy["verification_focus"] == "targeted_validation"
            ):
                if focus_tests:
                    policy["verification_focus"] = "focus_test_files"
            _brief_sp = brief.search_policy if isinstance(brief.search_policy, dict) else {}
            if _brief_sp.get("decomposition_module_group") and _brief_sp.get(
                "module_group_owned_files"
            ):
                # Decomposition module-group briefs own a DISJOINT file partition.
                # Point the agent at its OWN owned files (plus a few bridge files as
                # coordination context) instead of letting the mode branches below
                # prepend the shared plan-level generic focus and truncate to 8 —
                # that clobber gave all groups the SAME 8 files, so every rollout
                # edited the same overlapping core surface and the N-way union
                # collapsed to one member. Layer-A general: gated purely on the
                # decomposition flag, byte-identical for every non-decomposition
                # brief (the elif chain below is unchanged).
                _owned = _dedupe_preserve(
                    [f for f in (_brief_sp.get("module_group_owned_files") or []) if f]
                )
                _bridge = _dedupe_preserve(
                    [f for f in (_brief_sp.get("module_group_bridge_files") or []) if f]
                )
                brief.focus_files = _dedupe_preserve(_owned + _bridge)[:24]
            elif mode in {"test_rooted", "invariant_guard", "api_contract"}:
                brief.focus_files = _dedupe_preserve(
                    terminal_sources
                    + incomplete_sources
                    + source_focus
                    + test_priority_tail
                    + brief.focus_files
                )[:8]
            elif mode in {"surgical", "source_cluster"}:
                brief.focus_files = _dedupe_preserve(
                    terminal_sources
                    + incomplete_sources
                    + source_focus
                    + source_priority_tail
                    + brief.focus_files
                )[:8]
            elif mode == "dependency_trace":
                neighbors = repo_context.get_dependency_neighbors(
                    brief.focus_files[:2], max_neighbors=2
                )
                brief.focus_files = _dedupe_preserve(
                    terminal_sources
                    + incomplete_sources
                    + source_focus
                    + source_priority_tail
                    + brief.focus_files
                    + neighbors
                )[:8]
            elif source_focus or incomplete_sources or incomplete_tests:
                brief.focus_files = _dedupe_preserve(
                    terminal_sources
                    + incomplete_sources
                    + source_focus
                    + source_priority_tail
                    + brief.focus_files
                )[:8]

            hypothesis_prefixes: list[str] = []
            if test_context.exception_summaries:
                hypothesis_prefixes.append(
                    f"Direct baseline exception: {test_context.exception_summaries[0]}."
                )
            if incomplete_sources:
                hypothesis_prefixes.append(
                    f"{incomplete_sources[0]} contains obvious implementation scaffolds; inspect the nearby API or backend contract instead of fixing only one missing reference."
                )
            if terminal_sources:
                hypothesis_prefixes.append(
                    f"The direct failure terminates in {terminal_sources[0]}."
                )
            if completion_neighbor_focus:
                hypothesis_prefixes.append(
                    "After the first import blocker clears, adjacent imported modules may reveal the next missing contract."
                )
            if test_context.expectations:
                hypothesis_prefixes.append(
                    f"Visible tests define a contract around {test_context.expectations[0]}."
                )
            if source_focus:
                hypothesis_prefixes.append(f"The traceback converges through {source_focus[0]}.")
            if evidence_policy.mode == EVIDENCE_MODE_GOLD_SUITE_VISIBLE:
                hypothesis_prefixes.append(
                    "Use the declared gold visible suite as the contract source; every missing or failing expected test is direct progress feedback."
                )
            elif evidence_policy.mode == EVIDENCE_MODE_PARTIAL_SUITE_VISIBLE:
                hypothesis_prefixes.append(
                    "Visible tests may be incomplete; add hidden-test-risk reasoning around edge cases, interface compatibility, and nearby regressions."
                )
            if incomplete_sources:
                hypothesis_prefixes.append(
                    "The failing module likely needs grouped repository-completion work, not a single-symbol patch."
                )
            if incomplete_tests:
                hypothesis_prefixes.append(
                    "Some visible tests contain TODO/NotImplemented scaffolding and may require completion without weakening their assertions."
                )
            if hypothesis_prefixes:
                brief.hypotheses = _dedupe_preserve(hypothesis_prefixes + brief.hypotheses)[:5]
            if terminal_sources and test_context.exception_summaries:
                brief.success_criteria = _dedupe_preserve(
                    [
                        f"Clear the direct baseline exception in {terminal_sources[0]} before broader changes."
                    ]
                    + list(brief.success_criteria or issue_plan.success_criteria)
                )[:5]
            policy["failing_test_count"] = test_context.failing_test_count
            policy["focus_test_file_count"] = len(focus_tests)
            self._sync_brief_controller_action(
                brief,
                policy=policy,
                issue_plan=issue_plan,
                repo_context=repo_context,
            )
            brief.delegation_policy = self._normalize_brief_delegation_policy(
                issue_plan,
                brief,
                repo_context,
            )

        ranked_briefs = self._rank_rollout_brief_candidates(
            issue_plan,
            candidate_briefs,
            limit=original_brief_count,
        )
        if focus_tests and not any(
            self._normalize_brief_search_policy(brief).get("mode") == "test_rooted"
            for brief in ranked_briefs
        ):
            fallback = next(
                (
                    brief
                    for brief in candidate_briefs
                    if self._normalize_brief_search_policy(brief).get("mode") == "test_rooted"
                ),
                None,
            )
            if fallback is not None:
                if len(ranked_briefs) >= original_brief_count and ranked_briefs:
                    ranked_briefs[-1] = fallback
                else:
                    ranked_briefs.append(fallback)
        issue_plan.rollout_briefs = ranked_briefs

        self._assign_rollout_profiles(
            briefs=issue_plan.rollout_briefs,
            issue_description=issue_description,
            repo_context=repo_context,
            relevant_files=list(issue_plan.relevant_files),
            test_files=[
                path for path in list(issue_plan.relevant_files) if self._looks_like_test_path(path)
            ],
            task_regime=issue_plan.task_regime,
        )

    def _brief_action_test_ids(
        self,
        issue_plan: Optional[IssuePlan],
        policy: dict[str, Any],
    ) -> list[str]:
        if isinstance(policy.get("graph_target_test_ids"), list):
            explicit = _dedupe_preserve(
                [
                    str(item).strip()
                    for item in list(policy.get("graph_target_test_ids") or [])
                    if str(item).strip()
                ]
            )
            if explicit:
                return explicit[:6]
        if not isinstance(issue_plan, IssuePlan):
            return []
        test_context = issue_plan.test_context
        verification_focus = str(policy.get("verification_focus") or "").strip().lower()
        if verification_focus == "failing_tests":
            return _dedupe_preserve(list(test_context.failing_test_ids or []))[:6]
        if verification_focus == "focus_test_files":
            return _dedupe_preserve(list(test_context.focus_test_files or []))[:4]
        return _dedupe_preserve(
            list(test_context.failing_test_ids or []) + list(test_context.focus_test_files or [])
        )[:6]

    def _sync_brief_controller_action(
        self,
        brief: RolloutBrief,
        *,
        policy: Optional[dict[str, Any]] = None,
        issue_plan: Optional[IssuePlan] = None,
        repo_context: Optional[RepoContext] = None,
    ) -> ControllerAction:
        merged_policy = dict(policy if isinstance(policy, dict) else (brief.search_policy or {}))
        source_files = _dedupe_preserve(
            [
                str(path).strip()
                for path in (
                    list(brief.focus_files or [])
                    + list(merged_policy.get("action_file_paths") or [])
                )
                if str(path).strip() and not self._looks_like_test_path(path)
            ]
        )[:8]
        if not source_files and isinstance(issue_plan, IssuePlan):
            source_files = [
                path
                for path in _dedupe_preserve(list(issue_plan.relevant_files or []))
                if not self._looks_like_test_path(path)
            ][:8]
        action_symbols = _dedupe_preserve(
            list(merged_policy.get("action_symbols") or [])
            + list(merged_policy.get("interface_symbols") or [])
        )
        if isinstance(issue_plan, IssuePlan):
            action_symbols = _dedupe_preserve(
                list(issue_plan.test_context.terminal_reference_symbols or [])
                + action_symbols
            )
        edit_spans = list(merged_policy.get("edit_spans") or [])
        if isinstance(repo_context, RepoContext) and source_files:
            owned_symbols = self._owned_symbol_names(repo_context, source_files[:6])
            action_symbols = _dedupe_preserve(action_symbols + owned_symbols)[:8]
            if not edit_spans:
                edit_spans = self._owned_edit_spans(
                    repo_context,
                    source_files[:6],
                    _dedupe_preserve(action_symbols + owned_symbols),
                )
        if source_files:
            merged_policy["action_file_paths"] = list(source_files)
        test_ids = self._brief_action_test_ids(issue_plan, merged_policy)
        if test_ids:
            merged_policy["action_test_ids"] = list(test_ids)
        if action_symbols:
            merged_policy["action_symbols"] = list(action_symbols)
        if edit_spans:
            merged_policy["edit_spans"] = list(edit_spans[:8])
        return brief.set_controller_action(merged_policy, merge_policy=merged_policy)

    def _brief_action_feature_view(
        self,
        issue_plan: IssuePlan,
        brief: RolloutBrief,
    ) -> dict[str, float]:
        action = brief.resolved_controller_action()
        return {
            "focus_file_count": float(len(list(brief.focus_files or []))),
            "action_file_count": float(len(list(action.file_paths or []))),
            "action_symbol_count": float(len(list(action.symbols or []))),
            "action_edit_span_count": float(len(list(action.edit_spans or []))),
            "failing_test_count": float(
                max(
                    int(issue_plan.test_context.failing_test_count or 0),
                    len(issue_plan.test_context.failing_test_ids or []),
                )
            ),
            "relevant_file_count": float(len(list(issue_plan.relevant_files or []))),
            "difficulty_estimate": float(issue_plan.difficulty_estimate or 0.0),
            "contract_gap_probability": float(
                self._task_regime_probability(issue_plan.task_regime, "contract_gap")
            ),
            "interface_probability": float(
                self._task_regime_probability(issue_plan.task_regime, "high_interface_risk")
            ),
            "importability_probability": float(
                self._task_regime_probability(issue_plan.task_regime, "importability_blocker")
            ),
        }

    def _normalize_brief_search_policy(self, brief: RolloutBrief) -> dict[str, Any]:
        action = brief.resolved_controller_action()
        policy = dict(brief.search_policy or {})
        mode = str(policy.get("mode") or "").strip().lower()
        if not mode:
            mode = _infer_rollout_brief_mode(brief.title, brief.goal)
        if not mode:
            mode = str(action.mode or "").strip().lower() or "surgical"
        policy["mode"] = mode
        policy.setdefault("verification_focus", "targeted_validation")
        brief.set_controller_action(policy, merge_policy=policy)
        return dict(brief.search_policy)

    def _normalize_brief_delegation_policy(
        self,
        issue_plan: IssuePlan,
        brief: RolloutBrief,
        repo_context: RepoContext,
    ) -> dict[str, Any]:
        policy = dict(brief.delegation_policy or {})
        allowed_stages = _normalize_delegation_allowed_stages(
            list(policy.get("allowed_stages") or ["patcher"])
        )

        if not self.config.rollout.enable_orchestrated_multi_agent:
            return {
                "enabled": False,
                "mode": "off",
                "reason": "orchestrator_multi_agent_disabled",
                "allowed_stages": allowed_stages,
            }
        if not self.config.aci.enable_agent_teams:
            return {
                "enabled": False,
                "mode": "off",
                "reason": "agent_team_infrastructure_disabled",
                "allowed_stages": allowed_stages,
            }
        if self._should_disable_collection_error_fast_path_delegation(issue_plan, brief):
            return self._disabled_delegation_policy(
                "importability_blocker_low_entropy",
                existing_policy={
                    "allowed_stages": allowed_stages,
                },
            )

        explicit_policy = bool(policy) and (
            "enabled" in policy
            or "mode" in policy
            or "max_tasks" in policy
            or "parallelism" in policy
        )
        completion_like = self._completion_like(issue_plan)
        search_policy = brief.search_policy if isinstance(brief.search_policy, dict) else {}
        action = self._sync_brief_controller_action(
            brief,
            policy=search_policy,
            issue_plan=issue_plan,
            repo_context=repo_context,
        )
        mode = str(action.mode or search_policy.get("mode") or "").strip().lower()
        target_kind = (
            str(action.graph_target_kind or search_policy.get("graph_target_kind") or "")
            .strip()
            .lower()
        )
        focus_files = list(action.file_paths or brief.focus_files or [])
        symbol_count = len(list(action.symbols or []))
        edit_span_count = len(list(action.edit_spans or []))
        difficulty = float(issue_plan.difficulty_estimate or 0.0)
        complexity_score = 0.0
        if completion_like:
            complexity_score += 1.5
        if len(focus_files) >= 4:
            complexity_score += 1
        if len(issue_plan.relevant_files) >= 6:
            complexity_score += 0.5
        if symbol_count >= 2:
            complexity_score += 1
        if edit_span_count >= 2:
            complexity_score += 1
        if difficulty >= 0.65:
            complexity_score += 1
        if mode in {"dependency_trace", "invariant_guard", "api_contract"}:
            complexity_score += 1
        if target_kind in {"joint", "hypothesis"}:
            complexity_score += 1

        heuristic_enabled = (
            bool(policy.get("enabled")) if explicit_policy else complexity_score >= 2
        )
        max_tasks = policy.get("max_tasks")
        if not isinstance(max_tasks, int) or max_tasks <= 0:
            max_tasks = (
                3 if completion_like or mode in {"dependency_trace", "invariant_guard"} else 2
            )
        max_tasks = max(1, min(max_tasks, self.config.aci.max_agent_team_size))

        max_iterations = policy.get("max_iterations")
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            max_iterations = 6 if completion_like else 5
        max_iterations = max(1, min(max_iterations, self.config.aci.max_agent_team_iterations))

        reason = str(policy.get("reason") or "").strip()
        if not reason:
            if completion_like:
                reason = (
                    "Repository-completion work is broad enough to justify a bounded repair team "
                    "once the main worker identifies separable file clusters."
                )
            elif mode == "dependency_trace":
                reason = "Caller/callee tracing can be split into a few focused threads before integration."
            elif mode == "invariant_guard":
                reason = (
                    "Invariant-guard work benefits from parallel contract and regression checks."
                )
            elif len(focus_files) >= 4:
                reason = "This rollout spans multiple file clusters and can be decomposed cleanly."
            else:
                reason = "Keep this rollout single-threaded to preserve a tight patch loop."

        subtasks = self._normalize_provided_delegation_subtasks(
            issue_plan,
            brief,
            list(policy.get("subtasks") or []),
            max_tasks=max_tasks,
        ) or self._build_delegation_subtasks(
            issue_plan,
            brief,
            repo_context=repo_context,
            max_tasks=max_tasks,
        )
        split_diagnostics = self._analyze_delegation_subtasks(
            repo_context,
            subtasks,
        )
        delegation_feature_view = {
            **self._brief_action_feature_view(issue_plan, brief),
            "complexity_score": float(complexity_score),
            "completion_like": 1.0 if completion_like else 0.0,
            "graph_target_is_joint": 1.0 if target_kind == "joint" else 0.0,
            "graph_target_is_hypothesis": 1.0 if target_kind == "hypothesis" else 0.0,
            "split_confidence": float(split_diagnostics.get("split_confidence") or 0.0),
            "graph_supported": 1.0 if bool(split_diagnostics.get("graph_supported")) else 0.0,
            "owned_symbol_count": float(split_diagnostics.get("owned_symbol_count") or 0.0),
            "edit_span_count": float(split_diagnostics.get("edit_span_count") or edit_span_count),
            "candidate_max_tasks": float(max_tasks),
            "heuristic_score": 1.0 if heuristic_enabled else 0.0,
        }
        enable_evaluation = evaluate_policy_model(
            getattr(self.config, "controller_models", None),
            model_name="planning.delegation_enablement",
            features=delegation_feature_view,
            baseline_value=1.0 if heuristic_enabled else 0.0,
            lower=0.0,
            upper=1.0,
        )
        enabled = bool(policy.get("enabled")) if explicit_policy else enable_evaluation.value >= 0.5

        requested_parallelism = policy.get("parallelism")
        if not isinstance(requested_parallelism, int) or requested_parallelism <= 0:
            requested_parallelism = (
                2
                if enabled
                and max_tasks > 1
                and (
                    completion_like
                    or mode in {"dependency_trace", "invariant_guard"}
                    or symbol_count >= 2
                    or edit_span_count >= 2
                )
                else 1
            )
        requested_parallelism = max(
            1,
            min(
                requested_parallelism,
                max_tasks,
                self.config.aci.max_agent_team_parallelism,
            ),
        )
        if enabled and len(subtasks) < 2:
            return {
                "enabled": False,
                "mode": "off",
                "reason": "insufficient_disjoint_subtasks",
                "allowed_stages": allowed_stages,
                "max_tasks": 0,
                "parallelism": 0,
                "max_iterations": 0,
                "subtasks": [],
                "split_confidence": float(split_diagnostics.get("split_confidence") or 0.0),
                "bridge_files": list(split_diagnostics.get("bridge_files") or []),
                "interface_symbols": list(split_diagnostics.get("interface_symbols") or []),
                "cluster_hints": list(split_diagnostics.get("cluster_hints") or []),
                "within_cluster_weight": float(
                    split_diagnostics.get("within_cluster_weight") or 0.0
                ),
                "cross_cluster_weight": float(split_diagnostics.get("cross_cluster_weight") or 0.0),
                "graph_supported": bool(split_diagnostics.get("graph_supported")),
                "partition_basis": str(
                    split_diagnostics.get("partition_basis") or "file_affinity+symbol_surface"
                ),
                "interface_prediction_available": bool(
                    split_diagnostics.get("cluster_hints")
                    or split_diagnostics.get("interface_symbols")
                ),
            }
        if (
            enabled
            and bool(split_diagnostics.get("graph_supported"))
            and float(split_diagnostics.get("bridge_coverage") or 0.0) >= 0.95
            and float(split_diagnostics.get("cross_cluster_weight") or 0.0)
            >= (
                0.75
                * max(
                    float(split_diagnostics.get("within_cluster_weight") or 0.0),
                    1.0,
                )
            )
        ):
            return {
                "enabled": False,
                "mode": "off",
                "reason": "high_boundary_entanglement",
                "allowed_stages": allowed_stages,
                "max_tasks": 0,
                "parallelism": 0,
                "max_iterations": 0,
                "subtasks": [],
                "split_confidence": float(split_diagnostics.get("split_confidence") or 0.0),
                "bridge_files": list(split_diagnostics.get("bridge_files") or []),
                "interface_symbols": list(split_diagnostics.get("interface_symbols") or []),
                "cluster_hints": list(split_diagnostics.get("cluster_hints") or []),
                "within_cluster_weight": float(
                    split_diagnostics.get("within_cluster_weight") or 0.0
                ),
                "cross_cluster_weight": float(split_diagnostics.get("cross_cluster_weight") or 0.0),
                "graph_supported": bool(split_diagnostics.get("graph_supported")),
                "partition_basis": str(
                    split_diagnostics.get("partition_basis") or "file_affinity+symbol_surface"
                ),
                "interface_prediction_available": bool(
                    split_diagnostics.get("cluster_hints")
                    or split_diagnostics.get("interface_symbols")
                ),
            }
        if (
            enabled
            and bool(split_diagnostics.get("graph_supported"))
            and float(split_diagnostics.get("split_confidence") or 0.0)
            < float(
                getattr(
                    self._delegation_policy_config(),
                    "split_confidence_threshold",
                    0.6,
                )
                or 0.6
            )
        ):
            return {
                "enabled": False,
                "mode": "off",
                "reason": "low_split_confidence",
                "allowed_stages": allowed_stages,
                "max_tasks": 0,
                "parallelism": 0,
                "max_iterations": 0,
                "subtasks": [],
                "split_confidence": float(split_diagnostics.get("split_confidence") or 0.0),
                "bridge_files": list(split_diagnostics.get("bridge_files") or []),
                "interface_symbols": list(split_diagnostics.get("interface_symbols") or []),
                "cluster_hints": list(split_diagnostics.get("cluster_hints") or []),
                "within_cluster_weight": float(
                    split_diagnostics.get("within_cluster_weight") or 0.0
                ),
                "cross_cluster_weight": float(split_diagnostics.get("cross_cluster_weight") or 0.0),
                "graph_supported": bool(split_diagnostics.get("graph_supported")),
                "partition_basis": str(
                    split_diagnostics.get("partition_basis") or "file_affinity+symbol_surface"
                ),
                "interface_prediction_available": bool(
                    split_diagnostics.get("cluster_hints")
                    or split_diagnostics.get("interface_symbols")
                ),
            }

        effective_max_tasks = min(max_tasks, len(subtasks)) if enabled else 0
        effective_parallelism = (
            self.config.clamp_agent_team_parallelism(
                requested_parallelism,
                max_tasks=effective_max_tasks,
            )
            if enabled and effective_max_tasks > 0
            else 0
        )

        return {
            "enabled": enabled,
            "mode": "bounded_team" if enabled else "off",
            "allowed_stages": allowed_stages,
            "max_tasks": effective_max_tasks,
            "parallelism": effective_parallelism,
            "max_iterations": max_iterations if enabled else 0,
            "reason": reason,
            "subtasks": subtasks[:effective_max_tasks] if enabled else [],
            "split_confidence": float(split_diagnostics.get("split_confidence") or 0.0),
            "bridge_files": list(split_diagnostics.get("bridge_files") or []),
            "interface_symbols": list(split_diagnostics.get("interface_symbols") or []),
            "cluster_hints": list(split_diagnostics.get("cluster_hints") or []),
            "within_cluster_weight": float(split_diagnostics.get("within_cluster_weight") or 0.0),
            "cross_cluster_weight": float(split_diagnostics.get("cross_cluster_weight") or 0.0),
            "graph_supported": bool(split_diagnostics.get("graph_supported")),
            "partition_basis": str(
                split_diagnostics.get("partition_basis") or "file_affinity+symbol_surface"
            ),
            "interface_prediction_available": bool(
                split_diagnostics.get("cluster_hints") or split_diagnostics.get("interface_symbols")
            ),
        }

    def _should_disable_collection_error_fast_path_delegation(
        self,
        issue_plan: IssuePlan,
        brief: RolloutBrief,
    ) -> bool:
        search_policy = brief.search_policy if isinstance(brief.search_policy, dict) else {}
        action = brief.resolved_controller_action()
        importability_focus = str(
            action.regime_state or search_policy.get("origin_regime_state") or ""
        ).strip().lower() == "importability_blocker" or self._task_regime_probability(
            issue_plan.task_regime, "importability_blocker"
        ) >= self.regime_policy.threshold("importability_blocker")
        if not importability_focus:
            return False
        if self._should_allow_collection_error_fast_path_delegation(issue_plan, brief):
            return False
        test_context = issue_plan.test_context
        failing_test_count = max(
            int(test_context.failing_test_count or 0),
            len(test_context.failing_test_ids or []),
        )
        if test_context.passing_test_count > 0:
            return False
        if failing_test_count > 4:
            return False
        return bool(test_context.terminal_source_files or test_context.source_focus_files)

    def _should_allow_collection_error_fast_path_delegation(
        self,
        issue_plan: IssuePlan,
        brief: RolloutBrief,
    ) -> bool:
        if not self.config.planning.allow_collection_error_fast_path_delegation:
            return False

        test_context = issue_plan.test_context
        completion_like = self._completion_like(issue_plan)
        if not completion_like:
            return False

        planner_metadata = issue_plan.planner_metadata or {}
        rollout_pressure = max(
            int(issue_plan.recommended_rollouts or 0),
            int(planner_metadata.get("portfolio_rollout_floor") or 0),
            int(planner_metadata.get("portfolio_profile_count") or 0),
        )
        focus_file_count = len(
            _dedupe_preserve(
                list(brief.focus_files or [])
                + list(test_context.source_focus_files or [])
                + list(test_context.terminal_source_files or [])
            )
        )
        return bool(
            len(issue_plan.relevant_files or []) >= 6
            or len(issue_plan.risk_files or []) >= 4
            or len(test_context.incomplete_source_files or []) >= 2
            or len(test_context.terminal_source_files or []) >= 2
            or focus_file_count >= 4
            or float(issue_plan.difficulty_estimate or 0.0) >= 0.55
            or rollout_pressure >= 4
        )

    def _normalize_provided_delegation_subtasks(
        self,
        issue_plan: IssuePlan,
        brief: RolloutBrief,
        raw_subtasks: list[Any],
        *,
        max_tasks: int,
    ) -> list[dict[str, Any]]:
        if max_tasks <= 0:
            return []
        allowed_focus_files = set(
            _dedupe_preserve(
                [
                    normalized
                    for normalized in (
                        _strip_pytest_node_id(path)
                        for path in (
                            list(brief.focus_files or [])
                            + list(issue_plan.relevant_files or [])
                            + list(issue_plan.risk_files or [])
                            + list(issue_plan.test_context.source_focus_files or [])
                            + list(issue_plan.test_context.incomplete_source_files or [])
                            + list(issue_plan.test_context.focus_test_files or [])
                            + list(issue_plan.test_context.failing_test_ids or [])
                        )
                    )
                    if normalized
                ]
            )
        )
        normalized: list[dict[str, Any]] = []
        for raw_task in raw_subtasks:
            if not isinstance(raw_task, dict):
                continue
            title = str(raw_task.get("title") or "").strip()
            if not title:
                continue
            kind = str(raw_task.get("kind") or "implementation").strip().lower() or "implementation"
            owned_files = _dedupe_preserve(
                [
                    normalized_path
                    for normalized_path in (
                        _strip_pytest_node_id(path)
                        for path in list(
                            raw_task.get("owned_files") or raw_task.get("focus_files") or []
                        )
                    )
                    if normalized_path and normalized_path in allowed_focus_files
                ]
            )
            if kind != "validation" and not owned_files:
                continue
            validation_targets = _dedupe_preserve(
                [
                    str(item).strip()
                    for item in list(raw_task.get("validation_targets") or [])
                    if str(item).strip()
                ]
            )
            objective = str(raw_task.get("objective") or "").strip()
            deliverable = str(raw_task.get("deliverable") or "").strip()
            forbidden_files = _dedupe_preserve(
                [
                    _strip_pytest_node_id(path)
                    for path in list(raw_task.get("forbidden_files") or [])
                    if _strip_pytest_node_id(path) in allowed_focus_files
                ]
            )
            interface_symbols = _dedupe_preserve(
                [
                    str(item).strip()
                    for item in list(raw_task.get("interface_symbols") or [])
                    if str(item).strip()
                ]
            )
            owned_symbols = _dedupe_preserve(
                [
                    str(item).strip()
                    for item in list(raw_task.get("owned_symbols") or [])
                    if str(item).strip()
                ]
            )
            edit_spans = [
                {
                    "file_path": str(item.get("file_path") or "").strip(),
                    "symbol": str(item.get("symbol") or "").strip(),
                    "start_line": int(item.get("start_line") or 0),
                    "end_line": int(item.get("end_line") or 0),
                }
                for item in list(raw_task.get("edit_spans") or [])
                if isinstance(item, dict) and str(item.get("file_path") or "").strip()
            ]
            assumptions = _dedupe_preserve(
                [
                    str(item).strip()
                    for item in list(raw_task.get("assumptions") or [])
                    if str(item).strip()
                ]
            )
            escalation_triggers = _dedupe_preserve(
                [
                    str(item).strip()
                    for item in list(raw_task.get("escalation_triggers") or [])
                    if str(item).strip()
                ]
            )
            depends_on = _dedupe_preserve(
                [
                    str(item).strip()
                    for item in list(raw_task.get("depends_on") or [])
                    if str(item).strip()
                ]
            )
            normalized.append(
                {
                    "title": title,
                    "kind": kind,
                    "owned_files": owned_files,
                    "focus_files": owned_files,
                    "forbidden_files": forbidden_files,
                    "interface_symbols": interface_symbols,
                    "owned_symbols": owned_symbols,
                    "edit_spans": edit_spans,
                    "assumptions": assumptions,
                    "escalation_triggers": escalation_triggers,
                    "depends_on": depends_on,
                    "validation_targets": validation_targets,
                    "objective": objective,
                    "deliverable": deliverable,
                }
            )
        disjoint: list[dict[str, Any]] = []
        owned_paths: set[str] = set()
        for task in normalized:
            focus_files = list(task.get("owned_files") or task.get("focus_files") or [])
            if task.get("kind") == "validation":
                # Validation subtasks should report against targeted tests rather than
                # implicitly claiming ownership of editable files.
                focus_files = []
            focus_files = [path for path in focus_files if path not in owned_paths]
            if task.get("kind") != "validation" and not focus_files:
                continue
            task = dict(task)
            task["owned_files"] = focus_files
            task["focus_files"] = focus_files
            owned_paths.update(focus_files)
            disjoint.append(task)
        implementation_tasks = [task for task in disjoint if task.get("kind") != "validation"]
        if len(disjoint) < 2 or not implementation_tasks:
            return []
        return disjoint[:max_tasks]

    def _analyze_delegation_subtasks(
        self,
        repo_context: RepoContext,
        subtasks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        implementation_clusters = [
            _dedupe_preserve(
                [
                    str(path).strip()
                    for path in list(task.get("focus_files") or [])
                    if str(path).strip()
                ]
            )
            for task in subtasks
            if str(task.get("kind") or "implementation").strip().lower() != "validation"
        ]
        implementation_clusters = [cluster for cluster in implementation_clusters if cluster]
        flattened_files = _dedupe_preserve(
            [path for cluster in implementation_clusters for path in cluster]
        )
        if not implementation_clusters:
            return {
                "split_confidence": 0.0,
                "bridge_files": [],
                "interface_symbols": [],
                "cluster_hints": [],
                "within_cluster_weight": 0.0,
                "cross_cluster_weight": 0.0,
                "graph_supported": False,
                "partition_basis": "symbol_surface_unavailable",
            }
        if len(implementation_clusters) == 1:
            return {
                "split_confidence": 1.0,
                "bridge_files": [],
                "interface_symbols": [],
                "cluster_hints": [],
                "within_cluster_weight": 0.0,
                "cross_cluster_weight": 0.0,
                "graph_supported": bool(flattened_files),
                "partition_basis": "single_cluster",
            }
        affinity = repo_context.build_file_affinity_graph(flattened_files)
        diagnostics = self._evaluate_delegation_partition(
            repo_context,
            flattened_files,
            implementation_clusters,
            affinity,
        )
        interface_hints = repo_context.describe_partition_interfaces(implementation_clusters)
        diagnostics["bridge_files"] = _dedupe_preserve(
            list(diagnostics.get("bridge_files") or [])
            + list(interface_hints.get("bridge_files") or [])
        )
        diagnostics["bridge_coverage"] = (
            len(list(diagnostics.get("bridge_files") or [])) / len(flattened_files)
            if flattened_files
            else 0.0
        )
        diagnostics["interface_symbols"] = list(interface_hints.get("interface_symbols") or [])
        diagnostics["cluster_hints"] = list(interface_hints.get("cluster_hints") or [])
        policy = self._delegation_policy_config()
        owned_symbol_count = sum(
            len(_dedupe_preserve(list(task.get("owned_symbols") or [])))
            for task in subtasks
            if str(task.get("kind") or "implementation").strip().lower() != "validation"
        )
        edit_span_count = sum(
            len(list(task.get("edit_spans") or []))
            for task in subtasks
            if str(task.get("kind") or "implementation").strip().lower() != "validation"
        )
        confidence_bonus = 0.0
        if owned_symbol_count > 0:
            confidence_bonus += (
                float(getattr(policy, "symbol_interface_bonus", 0.12) or 0.12)
                * min(owned_symbol_count, 6)
                / 12.0
            )
        if edit_span_count > 0:
            confidence_bonus += (
                float(getattr(policy, "edit_span_bonus", 0.08) or 0.08)
                * min(edit_span_count, 6)
                / 12.0
            )
        diagnostics["split_confidence"] = max(
            0.0,
            min(1.0, float(diagnostics.get("split_confidence") or 0.0) + confidence_bonus),
        )
        diagnostics["owned_symbol_count"] = owned_symbol_count
        diagnostics["edit_span_count"] = edit_span_count
        return diagnostics

    def _delegation_file_surface(
        self,
        repo_context: RepoContext,
        path: str,
    ) -> dict[str, Any]:
        file_info = repo_context.get_file_info(path)
        if file_info is None:
            return {
                "symbol_count": 0,
                "edit_span_count": 0,
                "interface_symbol_count": 0,
                "line_count": 0,
                "import_count": 0,
            }
        owned_symbols = self._owned_symbol_names(repo_context, [path])
        edit_spans = self._owned_edit_spans(
            repo_context,
            [path],
            owned_symbols,
        )
        interface_symbols = self._collect_interface_symbols(repo_context, [path])
        return {
            "symbol_count": len(owned_symbols),
            "edit_span_count": len(edit_spans),
            "interface_symbol_count": len(interface_symbols),
            "line_count": int(file_info.line_count or 0),
            "import_count": len(file_info.imports or []),
        }

    def _delegation_cluster_surface(
        self,
        repo_context: RepoContext,
        cluster: list[str],
    ) -> dict[str, Any]:
        files = _dedupe_preserve(cluster)
        owned_symbols = self._owned_symbol_names(repo_context, files)
        edit_spans = self._owned_edit_spans(
            repo_context,
            files,
            owned_symbols,
        )
        interface_symbols = self._collect_interface_symbols(repo_context, files)
        line_count = 0
        import_count = 0
        for path in files:
            file_info = repo_context.get_file_info(path)
            if file_info is None:
                continue
            line_count += int(file_info.line_count or 0)
            import_count += len(file_info.imports or [])
        symbol_count = len(owned_symbols)
        edit_span_count = len(edit_spans)
        interface_symbol_count = len(interface_symbols)
        if symbol_count or edit_span_count or interface_symbol_count:
            work_score = (
                0.35
                + min(symbol_count / 2.5, 2.0)
                + min(edit_span_count / 2.0, 1.5)
                + min(interface_symbol_count / 2.0, 1.0)
                + min(import_count / 16.0, 0.25)
                + min(line_count / 180.0, 0.35)
            )
        else:
            work_score = 0.35 + min(line_count / 80.0, 1.0) + min(import_count / 12.0, 0.25)
        return {
            "symbol_count": symbol_count,
            "edit_span_count": edit_span_count,
            "interface_symbol_count": interface_symbol_count,
            "line_count": line_count,
            "import_count": import_count,
            "work_score": float(work_score),
        }

    def _delegation_file_work_score(
        self,
        repo_context: RepoContext,
        path: str,
    ) -> float:
        surface = self._delegation_file_surface(repo_context, path)
        if (
            surface["symbol_count"]
            or surface["edit_span_count"]
            or surface["interface_symbol_count"]
        ):
            return (
                0.35
                + min(float(surface["symbol_count"]) / 2.0, 1.75)
                + min(float(surface["edit_span_count"]) / 1.5, 1.0)
                + min(float(surface["interface_symbol_count"]) / 2.0, 0.75)
                + min(float(surface["line_count"]) / 180.0, 0.35)
                + min(float(surface["import_count"]) / 16.0, 0.20)
            )
        return (
            0.35
            + min(float(surface["line_count"]) / 80.0, 1.0)
            + min(
                float(surface["import_count"]) / 12.0,
                0.25,
            )
        )

    def _is_delegation_thin_file(
        self,
        repo_context: RepoContext,
        path: str,
    ) -> bool:
        surface = self._delegation_file_surface(repo_context, path)
        policy = self._delegation_policy_config()
        return (
            int(surface["line_count"]) <= int(getattr(policy, "thin_file_max_lines", 12) or 12)
            and int(surface["symbol_count"])
            <= int(getattr(policy, "thin_file_max_symbols", 1) or 1)
            and int(surface["edit_span_count"]) <= 1
            and int(surface["interface_symbol_count"]) <= 0
        )

    def _cluster_work_score(
        self,
        repo_context: RepoContext,
        cluster: list[str],
    ) -> float:
        return float(
            self._delegation_cluster_surface(repo_context, cluster).get("work_score") or 0.0
        )

    def _evaluate_delegation_partition(
        self,
        repo_context: RepoContext,
        files: list[str],
        clusters: list[list[str]],
        affinity: dict[str, dict[str, float]],
        *,
        protected_files: Optional[set[str]] = None,
    ) -> dict[str, Any]:
        policy = self._delegation_policy_config()
        strong_edge_threshold = 1.0
        cluster_index = {path: index for index, cluster in enumerate(clusters) for path in cluster}
        protected = {path for path in list(protected_files or set()) if path in cluster_index}
        within_weight = 0.0
        cross_weight = 0.0
        support_edges = 0
        strong_support_edges = 0
        cross_by_file: Counter[str] = Counter()
        within_by_cluster: Counter[int] = Counter()
        cross_by_cluster: Counter[int] = Counter()
        cluster_peer_weights: dict[int, Counter[int]] = {
            index: Counter() for index, _ in enumerate(clusters)
        }
        total_by_file = {
            path: sum(float(weight) for weight in affinity.get(path, {}).values()) for path in files
        }

        for index, left in enumerate(files):
            for right in files[index + 1 :]:
                weight = float(affinity.get(left, {}).get(right, 0.0) or 0.0)
                if weight <= 0:
                    continue
                support_edges += 1
                if weight >= strong_edge_threshold:
                    strong_support_edges += 1
                left_cluster = cluster_index.get(left)
                right_cluster = cluster_index.get(right)
                if left_cluster == right_cluster:
                    within_weight += weight
                    if left_cluster is not None:
                        within_by_cluster[left_cluster] += weight
                else:
                    cross_weight += weight
                    cross_by_file[left] += weight
                    cross_by_file[right] += weight
                    if left_cluster is not None:
                        cross_by_cluster[left_cluster] += weight
                    if right_cluster is not None:
                        cross_by_cluster[right_cluster] += weight
                    if left_cluster is not None and right_cluster is not None:
                        cluster_peer_weights[left_cluster][right_cluster] += weight
                        cluster_peer_weights[right_cluster][left_cluster] += weight

        total_weight = within_weight + cross_weight
        cluster_sizes = [len(cluster) for cluster in clusters if cluster]
        max_cluster = max(cluster_sizes) if cluster_sizes else 0
        min_cluster = min(cluster_sizes) if cluster_sizes else 0
        size_balance = (min_cluster / max_cluster) if max_cluster else 0.0
        cluster_surfaces = [
            self._delegation_cluster_surface(repo_context, cluster)
            for cluster in clusters
            if cluster
        ]
        cluster_work_scores = [
            float(surface.get("work_score") or 0.0) for surface in cluster_surfaces
        ]
        max_cluster_work = max(cluster_work_scores) if cluster_work_scores else 0.0
        min_cluster_work = min(cluster_work_scores) if cluster_work_scores else 0.0
        work_balance = (min_cluster_work / max_cluster_work) if max_cluster_work else 0.0
        cluster_symbol_counts = [
            int(surface.get("symbol_count") or 0) for surface in cluster_surfaces
        ]
        cluster_edit_span_counts = [
            int(surface.get("edit_span_count") or 0) for surface in cluster_surfaces
        ]
        max_cluster_symbols = max(cluster_symbol_counts) if cluster_symbol_counts else 0
        min_cluster_symbols = min(cluster_symbol_counts) if cluster_symbol_counts else 0
        max_cluster_edit_spans = max(cluster_edit_span_counts) if cluster_edit_span_counts else 0
        min_cluster_edit_spans = min(cluster_edit_span_counts) if cluster_edit_span_counts else 0
        symbol_balance = (min_cluster_symbols / max_cluster_symbols) if max_cluster_symbols else 1.0
        edit_span_balance = (
            (min_cluster_edit_spans / max_cluster_edit_spans) if max_cluster_edit_spans else 1.0
        )
        balance = (
            (0.20 * size_balance)
            + (0.35 * work_balance)
            + (0.25 * symbol_balance)
            + (0.20 * edit_span_balance)
        )
        separation = (within_weight / total_weight) if total_weight > 0 else 0.0
        connectivity_signal = (
            min(1.0, total_weight / max(4.0, float(len(files) * 2))) if total_weight > 0 else 0.0
        )
        bridge_files = [
            path
            for path in files
            if total_by_file.get(path, 0.0) > 0
            and cross_by_file.get(path, 0.0)
            >= float(getattr(policy, "bridge_weight_min", 3.0) or 3.0)
            and (cross_by_file.get(path, 0.0) / total_by_file.get(path, 1.0))
            >= float(getattr(policy, "bridge_cross_ratio", 0.45) or 0.45)
        ]
        bridge_coverage = (len(bridge_files) / len(files)) if files else 0.0
        low_leverage_cluster_indices: list[int] = []
        for index, cluster in enumerate(clusters):
            if not cluster or set(cluster) & protected:
                continue
            cluster_surface = self._delegation_cluster_surface(repo_context, cluster)
            cluster_work = float(cluster_surface.get("work_score") or 0.0)
            work_threshold = min(
                float(getattr(policy, "low_leverage_cluster_max_work", 2.6) or 2.6),
                max_cluster_work
                * float(getattr(policy, "low_leverage_cluster_work_ratio", 0.45) or 0.45),
            )
            internal_weight = float(within_by_cluster.get(index, 0.0) or 0.0)
            outbound_weight = float(cross_by_cluster.get(index, 0.0) or 0.0)
            peer_weight = max(cluster_peer_weights.get(index, Counter()).values(), default=0.0)
            total_cluster_weight = internal_weight + outbound_weight
            if outbound_weight < float(getattr(policy, "low_leverage_peer_weight_min", 1.0) or 1.0):
                continue
            if total_cluster_weight > 0 and (outbound_weight / total_cluster_weight) < float(
                getattr(policy, "low_leverage_outbound_ratio", 0.55) or 0.55
            ):
                continue
            if peer_weight < float(getattr(policy, "low_leverage_peer_weight_min", 1.0) or 1.0):
                continue
            thin_file_count = sum(
                1 for path in cluster if self._is_delegation_thin_file(repo_context, path)
            )
            thin_cluster = thin_file_count == len(cluster)
            sparse_symbol_surface = (
                int(cluster_surface.get("symbol_count") or 0)
                <= max(
                    1,
                    int(getattr(policy, "thin_file_max_symbols", 1) or 1) + 1,
                )
                and int(cluster_surface.get("edit_span_count") or 0) <= 2
                and int(cluster_surface.get("interface_symbol_count") or 0) <= 1
            )
            weak_small_cluster = (
                len(cluster) <= int(getattr(policy, "low_leverage_cluster_max_files", 2) or 2)
                and cluster_work <= work_threshold
            )
            weak_thin_cluster = thin_cluster and cluster_work <= min(
                float(getattr(policy, "thin_cluster_max_work", 1.75) or 1.75),
                max_cluster_work * float(getattr(policy, "thin_cluster_work_ratio", 0.60) or 0.60),
            )
            weak_symbol_cluster = sparse_symbol_surface and cluster_work <= work_threshold
            if not weak_small_cluster and not weak_thin_cluster and not weak_symbol_cluster:
                continue
            low_leverage_cluster_indices.append(index)
        low_leverage_penalty = float(
            getattr(policy, "low_leverage_confidence_penalty", 0.18) or 0.18
        ) * min(
            len(low_leverage_cluster_indices),
            2,
        )
        entanglement_penalty = 0.0
        if bridge_coverage >= 0.75:
            entanglement_penalty = 0.20 * min(
                1.0,
                max(0.0, (bridge_coverage - 0.75) / 0.25),
            )
        confidence = (
            0.15
            + (0.45 * separation)
            + (0.20 * balance)
            + (0.20 * connectivity_signal)
            - (0.08 * min(len(bridge_files), 3))
            - (0.26 * bridge_coverage)
            - entanglement_penalty
            - low_leverage_penalty
        )
        confidence = max(0.0, min(confidence, 1.0))
        return {
            "split_confidence": confidence,
            "bridge_files": bridge_files,
            "bridge_coverage": bridge_coverage,
            "interface_symbols": [],
            "cluster_hints": [],
            "within_cluster_weight": within_weight,
            "cross_cluster_weight": cross_weight,
            "size_balance": size_balance,
            "work_balance": work_balance,
            "symbol_balance": symbol_balance,
            "edit_span_balance": edit_span_balance,
            "low_leverage_cluster_indices": low_leverage_cluster_indices,
            "low_leverage_cluster_count": len(low_leverage_cluster_indices),
            "entanglement_penalty": entanglement_penalty,
            "supporting_edge_count": support_edges,
            "strong_supporting_edge_count": strong_support_edges,
            "graph_supported": strong_support_edges > 0,
            "partition_basis": "file_affinity+symbol_surface",
        }

    def _repair_delegation_partitions(
        self,
        repo_context: RepoContext,
        files: list[str],
        partitions: list[list[str]],
        affinity: dict[str, dict[str, float]],
        *,
        original_index: dict[str, int],
        protected_files: Optional[set[str]] = None,
    ) -> list[list[str]]:
        normalized = [
            sorted(
                _dedupe_preserve(list(cluster)),
                key=lambda path: (original_index.get(path, 0), path),
            )
            for cluster in partitions
            if cluster
        ]
        protected = set(protected_files or set())
        while len(normalized) > 1:
            diagnostics = self._evaluate_delegation_partition(
                repo_context,
                files,
                normalized,
                affinity,
                protected_files=protected,
            )
            low_indices = list(diagnostics.get("low_leverage_cluster_indices") or [])
            if not low_indices:
                break
            low_indices.sort(
                key=lambda index: (
                    self._cluster_work_score(repo_context, normalized[index]),
                    len(normalized[index]),
                    index,
                )
            )
            merged = False
            for source_index in low_indices:
                cluster = normalized[source_index]
                best_target: Optional[int] = None
                best_weight = 0.0
                for target_index, peer_cluster in enumerate(normalized):
                    if target_index == source_index:
                        continue
                    weight = sum(
                        float(affinity.get(path, {}).get(peer, 0.0) or 0.0)
                        for path in cluster
                        for peer in peer_cluster
                    )
                    if weight > best_weight:
                        best_weight = weight
                        best_target = target_index
                if best_target is None or best_weight < float(
                    getattr(self._delegation_policy_config(), "low_leverage_peer_weight_min", 1.0)
                    or 1.0
                ):
                    continue
                normalized[best_target] = sorted(
                    _dedupe_preserve(normalized[best_target] + cluster),
                    key=lambda path: (original_index.get(path, 0), path),
                )
                del normalized[source_index]
                merged = True
                break
            if not merged:
                break
        normalized.sort(key=lambda cluster: (original_index.get(cluster[0], 0), cluster[0]))
        return normalized

    def _finalize_delegation_partitions(
        self,
        repo_context: RepoContext,
        files: list[str],
        partitions: list[list[str]],
        affinity: dict[str, dict[str, float]],
        *,
        original_index: dict[str, int],
        protected_files: Optional[set[str]] = None,
    ) -> tuple[list[list[str]], dict[str, Any]]:
        repaired_partitions = self._repair_delegation_partitions(
            repo_context,
            files,
            partitions,
            affinity,
            original_index=original_index,
            protected_files=protected_files,
        )
        diagnostics = self._evaluate_delegation_partition(
            repo_context,
            files,
            repaired_partitions,
            affinity,
            protected_files=protected_files,
        )
        interface_hints = repo_context.describe_partition_interfaces(repaired_partitions)
        diagnostics["bridge_files"] = _dedupe_preserve(
            list(diagnostics.get("bridge_files") or [])
            + list(interface_hints.get("bridge_files") or [])
        )
        diagnostics["bridge_coverage"] = (
            len(list(diagnostics.get("bridge_files") or [])) / len(files) if files else 0.0
        )
        diagnostics["interface_symbols"] = list(interface_hints.get("interface_symbols") or [])
        diagnostics["cluster_hints"] = list(interface_hints.get("cluster_hints") or [])
        return repaired_partitions, diagnostics

    def _partition_delegation_source_files(
        self,
        repo_context: RepoContext,
        source_files: list[str],
        *,
        max_partitions: int,
        anchor_files: Optional[list[str]] = None,
        protected_files: Optional[list[str]] = None,
    ) -> tuple[list[list[str]], dict[str, Any]]:
        files = _dedupe_preserve(
            [path for path in source_files if path and not self._looks_like_test_path(path)]
        )
        if not files or max_partitions <= 0:
            return [], {
                "split_confidence": 0.0,
                "bridge_files": [],
                "interface_symbols": [],
                "cluster_hints": [],
                "within_cluster_weight": 0.0,
                "cross_cluster_weight": 0.0,
                "graph_supported": False,
                "partition_basis": "no_source_files",
            }
        partition_count = max(1, min(max_partitions, len(files)))
        if partition_count == 1:
            return [files], {
                "split_confidence": 1.0,
                "bridge_files": [],
                "interface_symbols": [],
                "cluster_hints": [
                    {
                        "bridge_files": [],
                        "interface_symbols": [],
                        "peer_files": [],
                    }
                ],
                "within_cluster_weight": 0.0,
                "cross_cluster_weight": 0.0,
                "graph_supported": bool(files),
                "partition_basis": "single_cluster",
            }

        affinity = repo_context.build_file_affinity_graph(files)
        strong_edge_threshold = 1.0
        file_set = set(files)
        priority_files = {path for path in list(anchor_files or []) if path in file_set}
        protected = {path for path in list(protected_files or []) if path in file_set}
        original_index = {path: index for index, path in enumerate(files)}
        weighted_degree = {
            path: sum(float(weight) for weight in affinity.get(path, {}).values()) for path in files
        }
        ordered = sorted(
            files, key=lambda path: (-weighted_degree[path], original_index[path], path)
        )
        strong_support = any(
            float(affinity.get(left, {}).get(right, 0.0) or 0.0) >= strong_edge_threshold
            for index, left in enumerate(files)
            for right in files[index + 1 :]
        )

        if not strong_support:
            chunk_size = max(1, math.ceil(len(files) / partition_count))
            partitions = [
                files[index : index + chunk_size]
                for index in range(0, len(files), chunk_size)
                if files[index : index + chunk_size]
            ]
            while len(partitions) > partition_count:
                tail = partitions.pop()
                partitions[-1].extend(tail)
            return self._finalize_delegation_partitions(
                repo_context,
                files,
                partitions[:partition_count],
                affinity,
                original_index=original_index,
                protected_files=protected,
            )

        if partition_count == 2 and len(files) <= int(
            getattr(self._delegation_policy_config(), "exhaustive_bisection_max_files", 8) or 8
        ):
            fixed = files[0]
            remaining = list(files[1:])
            best_partitions: Optional[list[list[str]]] = None
            best_diagnostics: Optional[dict[str, Any]] = None
            best_key: Optional[tuple[float, int, float, float]] = None
            for left_size in range(1, len(files)):
                for subset_rest in itertools.combinations(remaining, left_size - 1):
                    left = [fixed, *subset_rest]
                    left_set = set(left)
                    right = [path for path in files if path not in left_set]
                    if not right:
                        continue
                    partitions = [
                        sorted(left, key=lambda path: (original_index[path], path)),
                        sorted(right, key=lambda path: (original_index[path], path)),
                    ]
                    repaired_partitions, diagnostics = self._finalize_delegation_partitions(
                        repo_context,
                        files,
                        partitions,
                        affinity,
                        original_index=original_index,
                        protected_files=protected,
                    )
                    anchor_balance = 0
                    if priority_files and len(repaired_partitions) >= 2:
                        anchor_balance = min(
                            len(set(repaired_partitions[0]) & priority_files),
                            len(set(repaired_partitions[1]) & priority_files),
                        )
                    key = (
                        int(len(repaired_partitions) >= 2),
                        float(diagnostics.get("split_confidence") or 0.0),
                        anchor_balance,
                        float(diagnostics.get("within_cluster_weight") or 0.0),
                        -float(diagnostics.get("cross_cluster_weight") or 0.0),
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best_partitions = repaired_partitions
                        best_diagnostics = diagnostics
            if best_partitions is not None and best_diagnostics is not None:
                best_partitions.sort(
                    key=lambda cluster: (
                        original_index[cluster[0]],
                        cluster[0],
                    )
                )
                return best_partitions, best_diagnostics

        partitions = [[path] for path in ordered[:partition_count]]
        assigned = {path for cluster in partitions for path in cluster}
        for path in ordered:
            if path in assigned:
                continue
            best_index = max(
                range(len(partitions)),
                key=lambda index: (
                    (
                        sum(
                            float(affinity.get(path, {}).get(member, 0.0) or 0.0)
                            for member in partitions[index]
                        )
                        / max(1, len(partitions[index]))
                    )
                    - (0.40 * len(partitions[index])),
                    -len(partitions[index]),
                    -index,
                ),
            )
            partitions[best_index].append(path)
            assigned.add(path)

        normalized_partitions = [
            sorted(cluster, key=lambda path: (original_index[path], path))
            for cluster in partitions
            if cluster
        ]
        normalized_partitions.sort(key=lambda cluster: (original_index[cluster[0]], cluster[0]))
        return self._finalize_delegation_partitions(
            repo_context,
            files,
            normalized_partitions[:partition_count],
            affinity,
            original_index=original_index,
            protected_files=protected,
        )

    def _build_delegation_subtasks(
        self,
        issue_plan: IssuePlan,
        brief: RolloutBrief,
        *,
        repo_context: RepoContext,
        max_tasks: int,
    ) -> list[dict[str, Any]]:
        if max_tasks <= 1:
            return []

        search_policy = brief.search_policy if isinstance(brief.search_policy, dict) else {}
        mode = str(search_policy.get("mode") or "").strip().lower()
        source_pool = _dedupe_preserve(
            [
                path
                for path in (
                    list(brief.focus_files or [])
                    + list(issue_plan.test_context.source_focus_files or [])
                    + list(issue_plan.test_context.incomplete_source_files or [])
                    + list(issue_plan.relevant_files or [])
                )
                if path and not self._looks_like_test_path(path)
            ]
        )
        risk_files = _dedupe_preserve(
            list(issue_plan.risk_files or [])
            + list(issue_plan.test_context.terminal_source_files or [])
        )
        protected_files = _dedupe_preserve(
            risk_files
            + list(issue_plan.test_context.source_focus_files or [])
            + list(issue_plan.test_context.incomplete_source_files or [])
            + self._task_state_focus_files(issue_plan.task_state_context)
        )
        validation_targets = _dedupe_preserve(
            list(issue_plan.test_context.failing_test_ids or [])
            + list(issue_plan.test_context.focus_test_files or [])
        )
        source_partitions, partition_diagnostics = self._partition_delegation_source_files(
            repo_context,
            source_pool,
            max_partitions=max(1, max_tasks - 1),
            anchor_files=risk_files + list(brief.focus_files or []),
            protected_files=protected_files,
        )
        if len(source_partitions) < 2:
            return []
        cluster_hints = list(partition_diagnostics.get("cluster_hints") or [])
        global_interface_symbols = list(partition_diagnostics.get("interface_symbols") or [])

        subtasks: list[dict[str, Any]] = []
        for index, owned_files in enumerate(source_partitions):
            cluster_hint = cluster_hints[index] if index < len(cluster_hints) else {}
            peer_files = _dedupe_preserve(
                [
                    path
                    for path in list(cluster_hint.get("peer_files") or [])
                    if path not in set(owned_files)
                ]
            )
            bridge_files = _dedupe_preserve(
                [
                    path
                    for path in list(cluster_hint.get("bridge_files") or [])
                    if path not in set(owned_files)
                ]
            )
            owned_symbols = self._owned_symbol_names(repo_context, owned_files)
            edit_spans = self._owned_edit_spans(
                repo_context,
                owned_files,
                owned_symbols,
            )
            global_interface_set = set(global_interface_symbols)
            interface_symbols = (
                _dedupe_preserve(
                    list(cluster_hint.get("interface_symbols") or [])
                    + [symbol for symbol in owned_symbols if symbol in global_interface_set]
                )
                or global_interface_symbols[:4]
            )
            interface_symbols = _dedupe_preserve(interface_symbols)
            validation_slice = validation_targets[index : index + 1] or validation_targets[:1]
            if mode == "dependency_trace":
                objective = (
                    "Trace the caller/callee path through this owned file cluster and land the "
                    "implementation changes needed there without broad repo-wide edits."
                )
            elif mode == "invariant_guard":
                objective = (
                    "Repair this owned cluster while preserving nearby invariants and edge-case "
                    "behavior surfaced by the tests."
                )
            elif mode == "api_contract":
                objective = (
                    "Implement the missing behavior in this owned cluster while preserving the "
                    "public API and compatibility expectations."
                )
            elif mode == "test_rooted":
                objective = (
                    "Use the mapped visible-test slice to repair the owned implementation cluster "
                    "and validate the contract directly."
                )
            else:
                objective = (
                    "Own this implementation cluster, make the local code changes there, and hand "
                    "back targeted validation evidence."
                )
            assumptions: list[str] = []
            if interface_symbols:
                assumptions.append(
                    "Preserve or coordinate around interface symbols: "
                    + ", ".join(interface_symbols[:4])
                )
            if owned_symbols:
                assumptions.append("Primary owned symbols: " + ", ".join(owned_symbols[:4]))
            if bridge_files:
                assumptions.append(
                    "Boundary-sensitive neighboring files may need parent integration: "
                    + ", ".join(bridge_files[:4])
                )
            escalation_triggers = [
                "If resolving the fix requires editing any forbidden file, stop and report that need to the parent integrator.",
            ]
            if interface_symbols:
                escalation_triggers.append(
                    "If the fix changes the contract around "
                    + ", ".join(interface_symbols[:3])
                    + ", report the required integration work upward before widening scope."
                )
            subtasks.append(
                {
                    "title": f"Implementation cluster {index + 1}",
                    "kind": "implementation",
                    "owned_files": owned_files,
                    "focus_files": owned_files,
                    "forbidden_files": peer_files[:8],
                    "interface_symbols": interface_symbols[:6],
                    "owned_symbols": owned_symbols[:8],
                    "edit_spans": edit_spans[:8],
                    "assumptions": assumptions[:4],
                    "escalation_triggers": escalation_triggers[:4],
                    "depends_on": [],
                    "validation_targets": validation_slice,
                    "objective": objective,
                    "deliverable": (
                        "Return a focused patch touching the owned files and the targeted tests run "
                        "against this cluster."
                    ),
                }
            )

        if validation_targets and len(subtasks) < max_tasks:
            subtasks.append(
                {
                    "title": "Regression and integration validation",
                    "kind": "validation",
                    "owned_files": [],
                    "focus_files": [],
                    "forbidden_files": [],
                    "interface_symbols": global_interface_symbols[:6],
                    "owned_symbols": [],
                    "edit_spans": [],
                    "assumptions": [
                        "Treat child implementation patches as provisional until cross-cluster validation passes."
                    ],
                    "escalation_triggers": [
                        "Escalate any cross-cluster regressions or interface breakages back to the parent integrator."
                    ],
                    "depends_on": [
                        task["title"] for task in subtasks if task.get("kind") == "implementation"
                    ],
                    "validation_targets": validation_targets[:4],
                    "objective": (
                        "Validate the edited clusters against the highest-signal visible tests and "
                        "integration-risk files, then report remaining blockers or regressions."
                    ),
                    "deliverable": (
                        "Return validation outcomes, regressions, and any follow-up integration notes "
                        "without broad unrelated edits."
                    ),
                }
            )

        return subtasks[:max_tasks]

    def _decomposition_max_partitions(
        self,
        stub_file_count: int,
        *,
        num_rollouts: int,
    ) -> int:
        """clamp(ceil(stub_files / FILES_PER_GROUP), 2, max(num_rollouts, cap)) (T2.2)."""
        files_per_group = max(
            1,
            int(getattr(self.config.rollout, "decomposition_files_per_group", 25) or 25),
        )
        cap = max(
            2,
            int(getattr(self.config.rollout, "decomposition_max_partitions_cap", 12) or 12),
        )
        upper = max(int(num_rollouts or 0), cap)
        target = math.ceil(max(1, int(stub_file_count)) / files_per_group)
        return max(2, min(target, upper))

    def _partition_stub_surface_into_module_groups(
        self,
        repo_context: RepoContext,
        stub_files: list[str],
        *,
        num_rollouts: int,
        protected_files: Optional[list[str]] = None,
    ) -> list[ModuleGroup]:
        """Partition the repo-wide stub surface into disjoint module groups (T2.2).

        Reuses the existing scored/balanced/bridge-aware partitioner
        (:meth:`_partition_delegation_source_files`) verbatim — no new
        partition primitive. The stub surface is pre-seeded by top-level
        package dir (``statsmodels/tsa/...`` -> ``tsa``) via the anchor-file
        hint so the affinity graph refines around package boundaries while
        minimizing cross-cluster weight. Emits file-disjoint, balanced,
        bridge-aware :class:`ModuleGroup` records.
        """
        files = _dedupe_preserve(
            [path for path in stub_files if path and not self._looks_like_test_path(path)]
        )
        if len(files) < 2:
            return []
        max_partitions = self._decomposition_max_partitions(
            len(files),
            num_rollouts=num_rollouts,
        )
        # Pre-seed by top-level package dir: pick one representative file per
        # top-level dir as an anchor so the partitioner biases toward package
        # boundaries (e.g. statsmodels/tsa, statsmodels/stats, ...).
        package_anchor: dict[str, str] = {}
        for path in files:
            parts = Path(path).parts
            # Use the second component when the repo root is a single package
            # dir (statsmodels/<subpkg>/...), else the first component.
            top = parts[1] if len(parts) >= 2 else parts[0]
            package_anchor.setdefault(top, path)
        anchor_files = list(package_anchor.values())

        partitions, diagnostics = self._partition_delegation_source_files(
            repo_context,
            files,
            max_partitions=max_partitions,
            anchor_files=anchor_files,
            protected_files=list(protected_files or []),
        )
        if len(partitions) < 2:
            return []
        cluster_hints = list(diagnostics.get("cluster_hints") or [])
        global_interface_symbols = list(diagnostics.get("interface_symbols") or [])
        global_interface_set = set(global_interface_symbols)

        groups: list[ModuleGroup] = []
        for index, owned_files in enumerate(partitions):
            owned_set = set(owned_files)
            hint = cluster_hints[index] if index < len(cluster_hints) else {}
            bridge_files = _dedupe_preserve(
                [
                    path
                    for path in list(hint.get("bridge_files") or [])
                    if path not in owned_set
                ]
            )
            owned_symbols = self._owned_symbol_names(repo_context, owned_files)
            interface_symbols = _dedupe_preserve(
                list(hint.get("interface_symbols") or [])
                + [symbol for symbol in owned_symbols if symbol in global_interface_set]
            ) or list(global_interface_symbols[:4])
            groups.append(
                ModuleGroup(
                    group_id=index,
                    owned_files=list(owned_files),
                    bridge_files=bridge_files[:8],
                    interface_symbols=interface_symbols[:8],
                    expected_test_ids_subset=[],
                )
            )
        return groups

    def _module_group_rollout_brief(
        self,
        group: ModuleGroup,
        *,
        success_criteria: list[str],
        agent_mode: AgentMode,
        group_count: int,
    ) -> RolloutBrief:
        """Build one enforced-write-scope brief for a single module group (T2.3)."""
        owned = list(group.owned_files)
        graph_targets = _dedupe_preserve(owned + list(group.bridge_files))
        # The brief's search_policy carries the enforced write scope and the
        # per-group expected-id subset. ``focus_files`` narrow retrieval; the
        # graph targets + module_group_* keys narrow the discovery/localizer
        # scope (discovery_scope.py / localizer_scope.py) and drive the
        # ACI enforced write-scope wired by the engine (aci.set_write_scope
        # enforce=True / _enforce_write_scope reverts off-group edits).
        search_policy: dict[str, Any] = {
            "mode": "source_cluster",
            "verification_focus": "targeted_validation",
            "graph_target_file_paths": graph_targets[:12],
            "graph_target_symbols": list(group.interface_symbols[:6]),
            "decomposition_module_group": True,
            "module_group_id": int(group.group_id),
            "module_group_owned_files": owned,
            "module_group_bridge_files": list(group.bridge_files),
            "module_group_interface_symbols": list(group.interface_symbols),
            "module_group_expected_test_ids": list(group.expected_test_ids_subset),
            "enforce_module_group_write_scope": True,
        }
        return RolloutBrief(
            title=f"Module group {group.group_id + 1}/{group_count}",
            goal=(
                "Implement this owned module group's missing behavior end-to-end. "
                "Only edit files inside the group; coordinate around shared bridge "
                "files without rewriting them."
            ),
            focus_files=owned[:8],
            hypotheses=[
                "This module group is a tractable, file-disjoint slice of the library.",
                "The group's tests encode the contract its owned files must satisfy.",
            ],
            success_criteria=success_criteria,
            prompt_hint=(
                "Stay strictly within the owned files for this module group. "
                "Off-group edits will be reverted."
            ),
            agent_mode=agent_mode,
            search_policy=search_policy,
        )

    def _build_module_group_rollout_briefs(
        self,
        issue_plan: "IssuePlan",
        repo_context: RepoContext,
        *,
        success_criteria: list[str],
        num_rollouts: int,
        stub_files: Optional[list[str]] = None,
        expected_id_mapper: Optional[Any] = None,
        expected_id_partitioner: Optional[Any] = None,
    ) -> tuple[list[RolloutBrief], list[ModuleGroup]]:
        """One brief per disjoint module group for decomposition-scale repos (T2.3).

        Returns ``([], [])`` (so the caller keeps today's exact behavior) unless
        the repo trips :func:`repo_is_decomposition_scale` AND the stub surface
        partitions into >= 2 disjoint groups.

        ``expected_id_partitioner`` is the preferred Layer-B callback
        ``(groups_owned_files) -> [[node_id], ...]`` (T2.4): one global call that
        assigns each expected id to exactly ONE group, so the per-group subsets
        are DISJOINT. ``expected_id_mapper`` (``(owned_files) -> list[node_id]``)
        is the legacy per-group fallback; it double-assigns when groups co-own a
        subpackage, so it is used only when no partitioner is supplied.
        """
        if stub_files is None:
            stub_files = self._scan_repo_stub_surface(repo_context)
        if not repo_is_decomposition_scale(
            issue_plan,
            repo_context,
            config=self.config,
            stub_file_count=len(stub_files),
        ):
            return [], []
        protected = _dedupe_preserve(
            list(issue_plan.risk_files or [])
            + list(issue_plan.test_context.terminal_source_files or [])
        )
        groups = self._partition_stub_surface_into_module_groups(
            repo_context,
            stub_files,
            num_rollouts=num_rollouts,
            protected_files=protected,
        )
        if len(groups) < 2:
            return [], []
        assigned = False
        if expected_id_partitioner is not None:
            # Preferred: one global partition keeps the subsets disjoint.
            try:
                subsets = list(
                    expected_id_partitioner([list(g.owned_files) for g in groups]) or []
                )
            except Exception as exc:  # noqa: BLE001 - fall back to per-group mapper
                logger.warning("Module-group partitioner failed; using per-group mapper: %s", exc)
                subsets = []
            if len(subsets) == len(groups):
                for group, subset in zip(groups, subsets):
                    group.expected_test_ids_subset = list(subset or [])
                assigned = True
        if not assigned and expected_id_mapper is not None:
            for group in groups:
                try:
                    subset = list(expected_id_mapper(list(group.owned_files)) or [])
                except Exception:
                    subset = []
                group.expected_test_ids_subset = subset
        agent_mode = self._choose_agent_mode(repo_context, issue_plan.relevant_files)
        briefs = [
            self._module_group_rollout_brief(
                group,
                success_criteria=success_criteria,
                agent_mode=agent_mode,
                group_count=len(groups),
            )
            for group in groups
        ]
        return briefs, groups

    def _scan_repo_stub_surface(self, repo_context: RepoContext) -> list[str]:
        """Repo-wide stub-surface scan (T2.1), tolerant of missing workspace."""
        repo_path = str(getattr(repo_context, "repo_path", "") or "").strip()
        if not repo_path:
            return []
        try:
            from ..core.test_runners import detect_adapter
        except Exception:
            detect_adapter = None  # type: ignore[assignment]
        adapter = None
        if detect_adapter is not None:
            try:
                adapter = detect_adapter(Path(repo_path))
            except Exception:
                adapter = None
        try:
            return scan_repo_for_stub_surface(Path(repo_path), adapter=adapter)
        except Exception:
            return []

    def _source_file_candidates(self, relevant_files: list[str]) -> list[str]:
        source_files = [path for path in relevant_files if not self._looks_like_test_path(path)]
        return _dedupe_preserve(source_files)

    def _fallback_relevant_files(self, repo_context: RepoContext) -> list[str]:
        source_files = [
            file_info.path
            for file_info in repo_context.files
            if not self._looks_like_test_path(file_info.path)
        ]
        test_files = [
            file_info.path
            for file_info in repo_context.files
            if self._looks_like_test_path(file_info.path)
        ]
        return _dedupe_preserve(source_files[:6] + test_files[:2])[
            : self.config.planning.max_relevant_files
        ]

    def _build_source_windows(
        self,
        file_paths: list[str],
        *,
        max_windows: int,
    ) -> list[list[str]]:
        files = _dedupe_preserve(file_paths)
        if not files:
            return []
        window_size = 2 if len(files) <= 4 else 3 if len(files) <= 8 else 4
        stride = max(1, window_size - 1)
        windows: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for start in range(0, len(files), stride):
            window = files[start : start + window_size]
            if not window:
                continue
            signature = tuple(window)
            if signature in seen:
                continue
            seen.add(signature)
            windows.append(window)
            if len(windows) >= max_windows:
                break
        tail = tuple(files[-window_size:])
        if tail and tail not in seen and len(windows) < max_windows:
            windows.append(list(tail))
        return windows or [files[:window_size]]

    def _apply_search_policy_focus(
        self,
        mode: str,
        *,
        current_focus: list[str],
        source_window: list[str],
        test_files: list[str],
        risk_files: list[str],
        repo_context: RepoContext,
    ) -> list[str]:
        if mode == "test_rooted":
            return _dedupe_preserve(
                test_files[:2] + current_focus[:2] + source_window[:3] + risk_files[:1]
            )
        if mode == "dependency_trace":
            neighbors = repo_context.get_dependency_neighbors(source_window[:2], max_neighbors=3)
            return _dedupe_preserve(
                current_focus[:2] + source_window[:3] + neighbors[:3] + test_files[:1]
            )
        if mode == "invariant_guard":
            return _dedupe_preserve(risk_files[:3] + source_window[:3] + test_files[:2])
        if mode == "source_cluster":
            return _dedupe_preserve(source_window[:4] + current_focus[:2] + test_files[:1])
        if mode == "api_contract":
            return _dedupe_preserve(
                test_files[:2] + source_window[:3] + risk_files[:2] + current_focus[:2]
            )
        if mode == "agentless_pipeline":
            return _dedupe_preserve(
                test_files[:2] + risk_files[:3] + current_focus[:3] + source_window[:3]
            )
        return _dedupe_preserve(
            current_focus[:3] + source_window[:3] + risk_files[:2] + test_files[:1]
        )

    def _variant_hypothesis(
        self,
        mode: str,
        source_window: list[str],
        variant_index: int,
    ) -> str:
        focus_hint = (
            ", ".join(Path(path).name for path in source_window[:2]) or "a distinct code cluster"
        )
        if mode == "dependency_trace":
            return f"Variant {variant_index + 1}: a neighboring dependency around {focus_hint} may hide the real root cause."
        if mode == "test_rooted":
            return f"Variant {variant_index + 1}: a different visible-test slice around {focus_hint} may reveal the intended contract."
        if mode == "api_contract":
            return f"Variant {variant_index + 1}: public behavior near {focus_hint} may still be incomplete."
        if mode == "agentless_pipeline":
            return f"Variant {variant_index + 1}: a linear failure-to-source pass around {focus_hint} may produce the simplest validated repair."
        return f"Variant {variant_index + 1}: {focus_hint} may contain a distinct implementation path worth checking."

    def _choose_agent_mode(
        self,
        repo_context: RepoContext,
        relevant_files: list[str],
    ) -> AgentMode:
        explicit = self.config.rollout.agent_mode
        if explicit != AgentMode.ADAPTIVE:
            return explicit
        if len(repo_context.files) <= 40 and len(relevant_files) <= 8:
            return AgentMode.FULL_SOLVER
        if len(relevant_files) >= 10 or len(repo_context.files) >= 150:
            return AgentMode.SCAFFOLDED
        return AgentMode.ADAPTIVE

    def _agent_mode_for_primitives(self, primitives: list[Primitive]) -> AgentMode:
        if primitives == [Primitive.REACT]:
            return AgentMode.FULL_SOLVER
        return AgentMode.SCAFFOLDED

    def _primitive_prompt_hint(self, primitives: list[Primitive]) -> str:
        if primitives == [Primitive.REACT]:
            return "Treat this as a direct fix: inspect, patch, validate, and finish."
        labels = ", ".join(primitive.value for primitive in primitives)
        if Primitive.MCTS in primitives:
            return (
                f"Follow a staged workflow aligned with: {labels}. "
                "Checkpoint promising states and backtrack when a change dead-ends."
            )
        return f"Follow a staged workflow aligned with: {labels}."

    def _compose_prompt_hint(self, existing_hint: str, primitives: list[Primitive]) -> str:
        primitive_hint = self._primitive_prompt_hint(primitives)
        existing = (existing_hint or "").strip()
        if not existing:
            return primitive_hint
        if primitive_hint in existing:
            return existing
        return f"{existing} {primitive_hint}".strip()

    def _resolve_plan_rollout_inputs(
        self,
        issue_description: str,
        repo_context: RepoContext,
        *,
        rollout_count: Optional[int],
        difficulty: Optional[float],
    ) -> tuple[int, Optional[float]]:
        effective_rollout_count = rollout_count
        effective_difficulty = difficulty

        if effective_rollout_count is None:
            decision = self.build_execution_strategy(
                issue_description,
                repo_context,
            )
            effective_rollout_count = decision.rollout_count
            if effective_difficulty is None:
                effective_difficulty = decision.difficulty_estimate
            return effective_rollout_count, effective_difficulty

        if effective_difficulty is None:
            effective_difficulty = self.estimate_difficulty(issue_description, repo_context)
        component_ablation = component_ablation_assignment_for_task(
            config=self.config,
            issue_description=issue_description,
            repo_label=str(getattr(repo_context, "repo_path", "") or ""),
        )
        if component_disabled(component_ablation, "multi_rollout"):
            return 1, effective_difficulty
        return max(1, effective_rollout_count), effective_difficulty

    def _requested_rollout_budget(self, rollout_count: Optional[int]) -> int:
        if rollout_count is not None:
            return max(1, int(rollout_count))
        if self.config.rollout.enable_adaptive_allocation:
            return max(self.config.rollout.min_rollouts, self.config.rollout.max_rollouts)
        return max(1, int(self.config.rollout.num_rollouts))

    def _configured_rollout_buckets(self) -> list[int]:
        raw_buckets = self.config.rollout.rollout_buckets
        return sorted({int(bucket) for bucket in raw_buckets if int(bucket) > 0})

    def _eligible_rollout_buckets(self, minimum: int, maximum: int) -> list[int]:
        configured = self._configured_rollout_buckets()
        if not configured:
            return []
        eligible = [bucket for bucket in configured if minimum <= bucket <= maximum]
        if eligible:
            return eligible
        return sorted({minimum, maximum})

    def _clamp_rollout_bucket(self, requested: int) -> int:
        minimum = self.config.rollout.min_rollouts
        maximum = self.config.rollout.max_rollouts
        eligible = self._eligible_rollout_buckets(minimum, maximum)
        if not eligible:
            return max(minimum, min(requested, maximum))
        within_budget = [bucket for bucket in eligible if bucket >= requested]
        if within_budget:
            return within_budget[0]
        return eligible[-1]

    def _brief_family_count(
        self,
        requested_rollouts: int,
        *,
        planning_mode: str = "direct",
        llm_config: Optional[LLMConfig] = None,
    ) -> int:
        family_count = max(
            1,
            min(
                int(requested_rollouts),
                int(self.config.planning.max_rollout_brief_families),
            ),
        )
        if llm_config is not None and llm_config.is_cli_backend and self.config.use_concise_prompts:
            if planning_mode == "coarse":
                family_count = min(family_count, 4)
            else:
                family_count = min(family_count, 5)
        return max(1, family_count)

    def _compact_delegation_policy(
        self,
        policy: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(policy, dict):
            return {}
        compacted = dict(policy)
        compacted["reason"] = _truncate_words(
            compacted.get("reason"),
            max_words=20,
            max_chars=180,
        )
        compacted["subtasks"] = [
            {
                **dict(raw_task),
                "title": _truncate_words(
                    dict(raw_task).get("title"),
                    max_words=8,
                    max_chars=72,
                ),
                "objective": _truncate_words(
                    dict(raw_task).get("objective"),
                    max_words=28,
                    max_chars=220,
                ),
                "deliverable": _truncate_words(
                    dict(raw_task).get("deliverable"),
                    max_words=20,
                    max_chars=180,
                ),
                "owned_files": _dedupe_preserve(list(dict(raw_task).get("owned_files") or []))[:6],
                "focus_files": _dedupe_preserve(list(dict(raw_task).get("focus_files") or []))[:6],
                "forbidden_files": _dedupe_preserve(
                    list(dict(raw_task).get("forbidden_files") or [])
                )[:6],
                "interface_symbols": _dedupe_preserve(
                    list(dict(raw_task).get("interface_symbols") or [])
                )[:6],
                "assumptions": _compact_string_list(
                    list(dict(raw_task).get("assumptions") or []),
                    max_items=2,
                    max_words=18,
                    max_chars=160,
                ),
                "escalation_triggers": _compact_string_list(
                    list(dict(raw_task).get("escalation_triggers") or []),
                    max_items=2,
                    max_words=18,
                    max_chars=180,
                ),
                "depends_on": _dedupe_preserve(list(dict(raw_task).get("depends_on") or []))[:4],
                "validation_targets": _dedupe_preserve(
                    list(dict(raw_task).get("validation_targets") or [])
                )[:3],
            }
            for raw_task in list(compacted.get("subtasks") or [])
            if isinstance(raw_task, dict)
        ]
        return compacted

    def _compact_planner_brief(
        self,
        brief: RolloutBrief,
    ) -> RolloutBrief:
        brief.title = (
            _truncate_words(
                brief.title,
                max_words=8,
                max_chars=72,
            )
            or "Focused rollout"
        )
        brief.goal = (
            _truncate_words(
                brief.goal,
                max_words=28,
                max_chars=220,
            )
            or "Investigate the owned focus and land the strongest fix."
        )
        brief.focus_files = _dedupe_preserve(list(brief.focus_files or []))[:6]
        brief.hypotheses = _compact_string_list(
            brief.hypotheses,
            max_items=3,
            max_words=16,
            max_chars=140,
        )
        brief.success_criteria = _compact_string_list(
            brief.success_criteria,
            max_items=3,
            max_words=16,
            max_chars=140,
        )
        brief.prompt_hint = _truncate_words(
            brief.prompt_hint,
            max_words=20,
            max_chars=180,
        )
        brief.delegation_policy = self._compact_delegation_policy(
            brief.delegation_policy if isinstance(brief.delegation_policy, dict) else {}
        )
        return brief

    def _planner_hard_timeout_seconds(
        self,
        issue_description: str,
        repo_context: RepoContext,
        heuristic: IssuePlan,
        llm_config: LLMConfig,
    ) -> Optional[int]:
        configured = self.config.planning.planner_timeout_seconds
        if not llm_config.is_cli_backend or not isinstance(configured, int) or configured <= 0:
            return None
        return configured

    def _render_planner_baseline_signal_block(
        self,
        repo_context: RepoContext,
        baseline_result: Optional[Any],
        *,
        concise: bool,
    ) -> list[str]:
        if baseline_result is None:
            return []

        output = _baseline_output(baseline_result)
        failing_tests = _compact_string_list(
            _baseline_test_ids(baseline_result, "failing_tests"),
            max_items=4,
            max_words=8,
            max_chars=120,
        )
        traceback_signal = self._extract_traceback_signal(repo_context, output)
        terminal_source_files = _compact_string_list(
            traceback_signal.terminal_source_files,
            max_items=4,
            max_words=8,
            max_chars=140,
        )
        source_files = _compact_string_list(
            traceback_signal.source_files,
            max_items=4,
            max_words=8,
            max_chars=140,
        )
        exception_summaries = _compact_string_list(
            traceback_signal.exception_summaries,
            max_items=2,
            max_words=18,
            max_chars=180,
        )
        excerpt = self._truncate_diagnostic_block(
            output,
            max_lines=8 if concise else 12,
        )
        if not any(
            (
                failing_tests,
                terminal_source_files,
                source_files,
                exception_summaries,
                excerpt,
            )
        ):
            return []

        lines = ["# Baseline Failure Signal"]
        if failing_tests:
            lines.extend(
                [
                    "Failing tests:",
                    "\n".join(f"- {item}" for item in failing_tests),
                ]
            )
        if terminal_source_files:
            lines.extend(
                [
                    "Terminal source files:",
                    "\n".join(f"- {item}" for item in terminal_source_files),
                ]
            )
        elif source_files:
            lines.extend(
                [
                    "Traceback source files:",
                    "\n".join(f"- {item}" for item in source_files),
                ]
            )
        if exception_summaries:
            lines.extend(
                [
                    "Exceptions:",
                    "\n".join(f"- {item}" for item in exception_summaries),
                ]
            )
        if excerpt:
            lines.extend(
                [
                    "Traceback excerpt:",
                    excerpt,
                ]
            )
        lines.append("")
        return lines

    def _build_baseline_signal_metadata(
        self,
        repo_context: RepoContext,
        baseline_result: Optional[Any],
    ) -> dict[str, Any]:
        if baseline_result is None:
            return {}

        output = _baseline_output(baseline_result)
        traceback_signal = self._extract_traceback_signal(repo_context, output)
        lowered = output.lower()
        if not output:
            quality = "empty"
        elif traceback_signal.terminal_source_files or traceback_signal.exception_summaries:
            quality = "traceback_localized"
        elif _looks_like_collection_failure_output(output) or "collected 0 items /" in lowered:
            quality = "collection_summary_only"
        else:
            quality = "raw_failure_text"

        return {
            "baseline_signal_quality": quality,
            "baseline_signal_has_collection_signature": bool(
                _looks_like_collection_failure_output(output) or "collected 0 items /" in lowered
            ),
            "baseline_output_line_count": len(output.splitlines()),
            "baseline_terminal_source_files": list(traceback_signal.terminal_source_files[:4]),
            "baseline_exception_summaries": list(traceback_signal.exception_summaries[:4]),
        }

    def _render_seed_plan_block(
        self,
        issue_plan: IssuePlan,
        *,
        max_briefs: int,
    ) -> str:
        lines = []
        for index, brief in enumerate(list(issue_plan.rollout_briefs or [])[:max_briefs], start=1):
            mode = self._normalize_brief_search_policy(brief).get("mode")
            focus_files = ", ".join(list(brief.focus_files or [])[:3]) or "<none>"
            delegation = "enabled" if brief.delegation_enabled("patcher") else "disabled"
            lines.extend(
                [
                    f"{index}. {brief.title}",
                    f"   Goal: {_truncate_words(brief.goal, max_words=18, max_chars=140)}",
                    f"   Files: {focus_files} | Mode: {mode} | Delegation: {delegation}",
                ]
            )
        return "\n".join(lines).strip() or "No prior plan sketch is available."

    def _build_single_agent_portfolio_brief(
        self,
        heuristic: IssuePlan,
    ) -> RolloutBrief:
        focus_files = list(heuristic.risk_files or heuristic.relevant_files)[:6]
        brief = RolloutBrief(
            title="Single-threaded root-cause repair",
            goal=(
                "Stay single-threaded, localize the highest-probability root cause, "
                "and validate the result without delegation."
            ),
            focus_files=focus_files,
            hypotheses=[
                "A focused single-agent pass should remain available as a hedge against a weak split."
            ],
            success_criteria=list(
                heuristic.success_criteria or ["Run targeted validation before finishing."]
            ),
            prompt_hint="Keep this family non-delegated and integration-first.",
            delegation_policy={
                "enabled": False,
                "mode": "off",
                "reason": "Preserve an explicit single-agent fallback family in the portfolio.",
                "allowed_stages": ["patcher"],
            },
        )
        brief.search_policy = self._normalize_brief_search_policy(brief)
        return brief

    def _build_agentless_pipeline_portfolio_brief(
        self,
        heuristic: IssuePlan,
    ) -> RolloutBrief:
        focus_files = _dedupe_preserve(
            list(heuristic.test_context.focus_test_files or [])[:2]
            + list(heuristic.risk_files or [])[:4]
            + list(heuristic.relevant_files or [])[:6]
        )[:8]
        brief = RolloutBrief(
            title="Agentless pipeline repair",
            goal=(
                "Run a simple failure-driven localize, patch, validate, and rerank-style "
                "repair pass without delegation."
            ),
            focus_files=focus_files,
            hypotheses=[
                "Visible failures, stack traces, and repo-map neighbors identify a direct source repair.",
                "A linear edit loop should avoid orchestration overhead on straightforward contract gaps.",
            ],
            success_criteria=list(
                heuristic.success_criteria or ["Run targeted then broad validation before finishing."]
            ),
            prompt_hint=(
                "Use a linear pipeline: inspect failure evidence, patch source, run targeted "
                "validation, then broaden only after the focused check improves."
            ),
            agent_mode=AgentMode.CLI_AGENT,
            search_policy={
                "mode": "agentless_pipeline",
                "verification_focus": "targeted_then_broad_validation",
                "cli_agent_use_masai_preround": "structured_prompt",
                "origin": "agentless_portfolio_lane",
                "preserve_agent_mode": True,
            },
            delegation_policy={
                "enabled": False,
                "mode": "off",
                "reason": "Agentless-style portfolio lane uses a linear single-agent repair loop.",
                "allowed_stages": ["patcher"],
            },
        )
        brief.search_policy = self._normalize_brief_search_policy(brief)
        return brief

    def _ensure_portfolio_briefs(
        self,
        base_briefs: list[RolloutBrief],
        *,
        heuristic: IssuePlan,
        family_cap: int,
    ) -> list[RolloutBrief]:
        if not base_briefs:
            return base_briefs

        merged = [RolloutBrief.from_dict(brief.to_dict()) for brief in base_briefs]
        for brief in merged:
            brief.search_policy = self._normalize_brief_search_policy(brief)

        if (
            self.config.planning.enable_plan_portfolio
            and self.config.planning.always_include_single_agent_family
            and not self._plan_has_explicit_single_agent_family(merged)
        ):
            merged.append(self._build_single_agent_portfolio_brief(heuristic))

        if self.config.planning.enable_plan_portfolio:
            existing_modes = {
                str((brief.search_policy or {}).get("mode") or "").strip().lower()
                for brief in merged
            }
            if "test_rooted" not in existing_modes:
                fallback = next(
                    (
                        RolloutBrief.from_dict(brief.to_dict())
                        for brief in heuristic.rollout_briefs
                        if self._normalize_brief_search_policy(brief).get("mode") == "test_rooted"
                    ),
                    None,
                )
                if fallback is not None:
                    fallback.delegation_policy = {
                        "enabled": False,
                        "mode": "off",
                        "reason": "Preserve a non-delegated validation-rooted family in the plan portfolio.",
                        "allowed_stages": ["patcher"],
                    }
                    fallback.search_policy = self._normalize_brief_search_policy(fallback)
                    merged.append(fallback)

        unique: list[RolloutBrief] = []
        seen: set[tuple[str, tuple[str, ...], str]] = set()
        for brief in merged:
            key = (
                brief.title.strip().lower(),
                tuple(list(brief.focus_files or [])[:6]),
                brief.goal.strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(brief)
            if len(unique) >= family_cap:
                break
        if (
            self.config.planning.enable_plan_portfolio
            and self.config.planning.always_include_agentless_pipeline_family
            and family_cap >= 3
            and not self._plan_has_agentless_pipeline_family(unique)
        ):
            agentless = self._build_agentless_pipeline_portfolio_brief(heuristic)
            if len(unique) >= family_cap:
                unique[-1] = agentless
            else:
                unique.append(agentless)
        return unique

    def _llm_refine_plan(
        self,
        issue_description: str,
        repo_context: RepoContext,
        heuristic: IssuePlan,
        rollout_count: Optional[int] = None,
        difficulty: Optional[float] = None,
        hard_timeout_seconds: Optional[int] = None,
        llm_config_override: Optional[LLMConfig] = None,
        seed_plan: Optional[IssuePlan] = None,
        baseline_result: Optional[Any] = None,
        planning_mode: str = "direct",
    ) -> IssuePlan:
        effective_rollout_count = rollout_count or self.config.rollout.num_rollouts
        llm_config = llm_config_override or self.config.get_planner_llm()
        brief_family_count = self._brief_family_count(
            effective_rollout_count,
            planning_mode=planning_mode,
            llm_config=llm_config,
        )
        effective_hard_timeout_seconds = hard_timeout_seconds
        if effective_hard_timeout_seconds is None:
            effective_hard_timeout_seconds = self._planner_hard_timeout_seconds(
                issue_description,
                repo_context,
                heuristic,
                llm_config,
            )
        codex_cli_planner = llm_config.backend == LLMBackend.CODEX_CLI
        candidate_file_limit = min(
            len(heuristic.relevant_files),
            self.config.planning.max_relevant_files,
        )
        if planning_mode == "coarse":
            candidate_file_limit = max(8, min(candidate_file_limit, 10))
        elif seed_plan is not None:
            candidate_file_limit = max(10, min(candidate_file_limit, 12))
        elif codex_cli_planner:
            candidate_file_limit = max(10, min(candidate_file_limit, 12))
        elif self.config.use_concise_prompts:
            candidate_file_limit = max(14, candidate_file_limit)
        else:
            candidate_file_limit = max(18, candidate_file_limit)
        prompt_lines = [
            "# Issue",
            self._truncate_block(
                _truncate_words(
                    issue_description,
                    max_words=180 if codex_cli_planner else 260,
                    max_chars=1200 if codex_cli_planner else 1800,
                ),
                max_lines=12 if codex_cli_planner else 18,
            ),
            "",
            "# Heuristic Summary",
            _truncate_words(
                heuristic.summary,
                max_words=70,
                max_chars=520,
            ),
            "",
            "# Candidate Files",
            "\n".join(f"- {path}" for path in heuristic.relevant_files[:candidate_file_limit]),
            "",
            "# Risk Files",
            "\n".join(f"- {path}" for path in heuristic.risk_files[:4]),
            "",
            "# Success Criteria",
            "\n".join(
                f"- {criterion}"
                for criterion in _compact_string_list(
                    heuristic.success_criteria,
                    max_items=4,
                    max_words=18,
                    max_chars=140,
                )
            ),
            "",
        ]
        prompt_lines.extend(
            self._render_planner_baseline_signal_block(
                repo_context,
                baseline_result,
                concise=bool(codex_cli_planner or self.config.use_concise_prompts),
            )
        )
        if seed_plan is not None:
            prompt_lines.extend(
                [
                    "# Existing Plan Sketch",
                    self._render_seed_plan_block(
                        seed_plan,
                        max_briefs=max(1, brief_family_count),
                    ),
                    "",
                ]
            )
        elif not codex_cli_planner and planning_mode != "coarse":
            prompt_lines.extend(
                [
                    "# Focus Repo Map",
                    self._truncate_block(
                        heuristic.repo_focus_map,
                        max_lines=48 if self.config.use_concise_prompts else 120,
                    ),
                    "",
                ]
            )
        prompt_lines.extend(
            [
                (
                    f"Build a plan sketch covering {effective_rollout_count} rollouts."
                    if effective_rollout_count > brief_family_count
                    else f"Build a plan for {effective_rollout_count} rollouts."
                ),
                (
                    "Produce a fast first-pass plan. Prefer robust rollout families over brittle decomposition."
                    if planning_mode == "coarse"
                    else "Critique and refine the existing plan sketch rather than restarting from scratch."
                    if seed_plan is not None
                    else "Return the strongest rollout-family plan you can from the provided context."
                ),
                (
                    f"Return at most {brief_family_count} diverse rollout brief families. "
                    "Each family should represent a materially different search strategy, not a wording variant."
                ),
                "The runtime will expand these families into the full rollout budget locally.",
                (
                    "Use only file paths from the provided candidate list."
                    if codex_cli_planner
                    else "Use only file paths from the provided candidate list and repo map."
                ),
                "Prioritize plans that separate direct-fix, validation, and regression-hardening search modes.",
                "Keep each rollout family materially distinct instead of producing minor wording variants.",
                "Keep titles terse, goals to one short sentence, prompt_hint to one short sentence, and list fields concise.",
                "Use at most 3 short hypotheses and 3 short success criteria per family.",
                "Prefer omission over verbose prose for optional fields.",
                (
                    "Ensure at least one explicitly single-threaded, non-delegated family remains available as a fallback baseline."
                    if self.config.planning.enable_plan_portfolio
                    else ""
                ),
                (
                    "If a rollout should use bounded multi-agent execution, include delegation_policy "
                    "with an explicit subtask split, owned file clusters per subtask, and one integration "
                    "or validation-oriented subtask."
                    if self.config.rollout.enable_orchestrated_multi_agent
                    else "Do not rely on subagents or delegation in the plan."
                ),
                (
                    "Only decompose when the owned file clusters are genuinely separable by dependency or "
                    "call-graph structure. If the boundaries feel ambiguous, leave delegation disabled for "
                    "that rollout instead of forcing a weak split."
                    if self.config.rollout.enable_orchestrated_multi_agent
                    else ""
                ),
                (
                    "For delegated subtasks, include owned_files, forbidden_files, interface_symbols, assumptions, "
                    "escalation_triggers, and depends_on when they improve coordination."
                    if self.config.rollout.enable_orchestrated_multi_agent
                    else ""
                ),
                (
                    "Use runtime stage names in delegation_policy.allowed_stages: patcher, localizer, "
                    "reproducer, test_writer. For implementation or integration-validation delegation, "
                    "use patcher."
                    if self.config.rollout.enable_orchestrated_multi_agent
                    else ""
                ),
                (
                    "Return the plan via submit_plan."
                    if self.config.enable_planning_tool
                    else "Return a JSON object that matches the planning schema."
                ),
            ]
        )
        prompt = "\n".join(prompt_lines)
        payload, tokens_used = self._run_plan_prompt(
            llm_config,
            prompt,
            working_dir=repo_context.repo_path,
            hard_timeout_seconds=effective_hard_timeout_seconds,
        )
        candidate_files = set(
            _dedupe_preserve(
                list(heuristic.relevant_files)
                + list(heuristic.risk_files)
                + _repo_map_files(heuristic.repo_focus_map)
            )
        )
        briefs_payload = payload.get("rollout_briefs") or []
        if not briefs_payload:
            raise RuntimeError("Planner returned no rollout briefs.")

        base_briefs: list[RolloutBrief] = []
        for brief_payload in briefs_payload[:brief_family_count]:
            brief = RolloutBrief.from_dict(brief_payload)
            brief.focus_files = self._validate_files(
                brief.focus_files, candidate_files, heuristic.relevant_files
            )
            if not brief.success_criteria:
                brief.success_criteria = list(heuristic.success_criteria)
            brief.search_policy = self._normalize_brief_search_policy(brief)
            brief = self._compact_planner_brief(brief)
            base_briefs.append(brief)
        if not base_briefs:
            raise RuntimeError("Planner returned no valid rollout brief families.")
        authored_brief_family_count = len(base_briefs)
        base_briefs = self._ensure_portfolio_briefs(
            base_briefs,
            heuristic=heuristic,
            family_cap=brief_family_count,
        )
        portfolio_brief_family_count = len(base_briefs)

        relevant_files = self._validate_files(
            payload.get("relevant_files") or heuristic.relevant_files,
            candidate_files,
            heuristic.relevant_files,
        )
        risk_files = self._validate_files(
            payload.get("risk_files") or heuristic.risk_files,
            candidate_files,
            heuristic.risk_files,
        )
        refined_briefs = self._expand_rollout_briefs(
            base_briefs=base_briefs,
            issue_description=issue_description,
            repo_context=repo_context,
            relevant_files=relevant_files,
            requested_rollouts=effective_rollout_count,
        )
        planner_source = "llm_hierarchical" if effective_rollout_count > len(base_briefs) else "llm"

        return IssuePlan(
            summary=payload.get("summary") or heuristic.summary,
            keywords=list(heuristic.keywords),
            relevant_files=relevant_files,
            risk_files=risk_files,
            success_criteria=payload.get("success_criteria") or heuristic.success_criteria,
            rollout_briefs=refined_briefs,
            repo_focus_map=repo_context.build_context_pack(
                relevant_files[: self.config.planning.max_repo_map_files],
                max_symbols_per_file=8,
                seed_symbols=list(heuristic.keywords)
                + list(heuristic.test_context.terminal_reference_symbols or []),
            ),
            planner_source=planner_source,
            planner_tokens=tokens_used,
            difficulty_estimate=difficulty,
            recommended_rollouts=effective_rollout_count,
            test_context=heuristic.test_context,
            planner_metadata={
                "requested_rollouts": effective_rollout_count,
                "brief_family_count": authored_brief_family_count,
                "portfolio_brief_family_count": portfolio_brief_family_count,
                "family_cap": brief_family_count,
                "planning_mode": planning_mode,
                "seed_plan_used": bool(seed_plan is not None),
                "expansion_mode": (
                    "llm_family_expansion"
                    if effective_rollout_count > portfolio_brief_family_count
                    else "llm_direct"
                ),
            },
        )

    def _run_plan_prompt(
        self,
        llm_config: LLMConfig,
        prompt: str,
        working_dir: str,
        *,
        hard_timeout_seconds: Optional[int] = None,
    ) -> tuple[dict[str, Any], int]:
        attempted_fingerprints: set[tuple[str, str, str, str]] = set()
        last_exc: Optional[Exception] = None
        while True:
            resolved_llm_config, routing = resolve_available_llm_config(
                llm_config,
                self.config.llm_configs,
                exclude_fingerprints=attempted_fingerprints,
                purpose="planner",
            )
            resolved_fingerprint = llm_backend_fingerprint(resolved_llm_config)
            current_reason = llm_backend_unavailable_reason(resolved_llm_config)
            if current_reason:
                if last_exc is not None:
                    raise last_exc
                raise RuntimeError(current_reason)
            if resolved_fingerprint in attempted_fingerprints:
                break
            if routing.get("fallback_applied"):
                logger.info(
                    "Planner rerouted from %s/%s to %s/%s (%s)",
                    routing.get("requested_backend") or "unknown",
                    routing.get("requested_model") or "unknown",
                    routing.get("resolved_backend") or "unknown",
                    routing.get("resolved_model") or "unknown",
                    routing.get("fallback_kind") or "fallback",
                )
            attempted_fingerprints.add(resolved_fingerprint)
            try:
                return self._run_plan_prompt_once(
                    resolved_llm_config,
                    prompt,
                    working_dir,
                    hard_timeout_seconds=hard_timeout_seconds,
                )
            except Exception as exc:
                reason = record_llm_backend_failure(resolved_llm_config, exc)
                invocation_failover_reason = (
                    "" if reason else classify_llm_call_failover_failure(exc)
                )
                if not reason and not invocation_failover_reason:
                    raise
                last_exc = exc
                if reason:
                    logger.warning(
                        "Planner backend %s/%s became unavailable (%s); retrying alternate backend if configured.",
                        routing.get("resolved_backend") or "unknown",
                        routing.get("resolved_model") or "unknown",
                        reason,
                    )
                else:
                    logger.warning(
                        "Planner backend %s/%s hit invocation-local failure (%s); retrying alternate backend if configured without globally marking the backend unavailable.",
                        routing.get("resolved_backend") or "unknown",
                        routing.get("resolved_model") or "unknown",
                        invocation_failover_reason,
                    )
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(
            llm_backend_unavailable_reason(llm_config) or "Planner LLM is unavailable."
        )

    def _run_plan_prompt_once(
        self,
        llm_config: LLMConfig,
        prompt: str,
        working_dir: str,
        *,
        hard_timeout_seconds: Optional[int] = None,
    ) -> tuple[dict[str, Any], int]:
        system_prompt = (
            "You are a software engineering manager preparing diverse rollout briefs "
            "for a coding agent orchestrator."
        )
        configured_planner_timeout_seconds = self.config.planning.planner_timeout_seconds
        effective_hard_timeout_seconds = (
            hard_timeout_seconds
            if isinstance(hard_timeout_seconds, int) and hard_timeout_seconds > 0
            else configured_planner_timeout_seconds
        )
        effective_hard_timeout_seconds = (
            effective_hard_timeout_seconds
            if llm_config.is_cli_backend
            and isinstance(effective_hard_timeout_seconds, int)
            and effective_hard_timeout_seconds > 0
            else None
        )

        if llm_config.is_cli_backend:
            prompt = prompt.replace(
                "Return the plan via submit_plan.",
                "Return the plan as a JSON object matching the planning schema.",
            )
            schema = _CLI_PLAN_SCHEMA
            use_inline_schema = llm_config.backend == LLMBackend.CODEX_CLI
            schema_text = (
                "{\n"
                '  "summary": string,\n'
                '  "relevant_files": [string],\n'
                '  "risk_files": [string],\n'
                '  "success_criteria": [string],\n'
                '  "rollout_briefs": [\n'
                "    {\n"
                '      "title": string,\n'
                '      "goal": string,\n'
                '      "focus_files": [string],\n'
                '      "hypotheses": [string],\n'
                '      "success_criteria": [string],\n'
                '      "prompt_hint": string,\n'
                '      "agent_mode": "full_solver" | "scaffolded" | "adaptive",\n'
                '      "delegation_policy": {\n'
                '        "enabled": boolean,\n'
                '        "allowed_stages": [string],\n'
                '        "max_tasks": integer,\n'
                '        "parallelism": integer,\n'
                '        "reason": string,\n'
                '        "subtasks": [\n'
                "          {\n"
                '            "title": string,\n'
                '            "kind": string,\n'
                '            "objective": string,\n'
                '            "owned_files": [string],\n'
                '            "forbidden_files": [string],\n'
                '            "depends_on": [string],\n'
                '            "deliverable": string\n'
                "          }\n"
                "        ]\n"
                "      }\n"
                "    }\n"
                "  ]\n"
                "}"
            )
            if not use_inline_schema:
                schema_text = json.dumps(schema, indent=2, sort_keys=True)
            primary_prompt = (
                prompt
                + "\n\nRespond with JSON only. No prose, markdown, or code fences."
                + "\nKeep text fields terse. Omit optional keys when they are empty."
            )
            if use_inline_schema:
                primary_prompt += (
                    "\nReturn a single JSON object matching this schema:\n" + schema_text
                )
            else:
                primary_prompt += "\nReturn a JSON object matching the provided schema."
            logger.info(
                "Planner CLI attempt 1 starting (backend=%s model=%s prompt_chars=%s inline_schema=%s timeout=%s)",
                llm_config.backend.value,
                llm_config.model,
                len(primary_prompt),
                use_inline_schema,
                effective_hard_timeout_seconds or llm_config.cli_timeout,
            )
            result = CLIModelClient(llm_config).run_structured_prompt(
                prompt=primary_prompt,
                working_dir=working_dir,
                schema=None if use_inline_schema else schema,
                system_prompt=system_prompt,
                allow_edits=False,
                hard_timeout_seconds=effective_hard_timeout_seconds,
            )
            if result.success and result.parsed_json:
                validation_error = (
                    _validate_planner_parsed_json(
                        result.parsed_json,
                        schema=schema,
                        required_keys=["summary", "relevant_files", "rollout_briefs"],
                    )
                    if self.config.planning.enable_planner_output_validation
                    else None
                )
                if not validation_error:
                    logger.info(
                        "Planner CLI attempt 1 succeeded in %.1fs (tokens=%s)",
                        result.duration_seconds,
                        extract_total_tokens(result.usage),
                    )
                    return result.parsed_json, extract_total_tokens(result.usage)
                logger.warning(
                    "Planner CLI attempt 1 produced schema-invalid output (%s); "
                    "retrying rather than accepting the degraded plan",
                    validation_error,
                )
            if result.error and _looks_like_timeout_or_stall_error(result.error):
                logger.warning("Planner CLI attempt 1 stalled: %s", result.error)
                raise RuntimeError(result.error)

            retry_prompt = primary_prompt
            if use_inline_schema:
                retry_prompt += (
                    "\nThe response must be valid JSON with top-level keys: "
                    "summary, relevant_files, risk_files, success_criteria, rollout_briefs."
                )
            logger.info(
                "Planner CLI attempt 1 returned unparseable output after %.1fs; retrying once",
                result.duration_seconds,
            )
            logger.info(
                "Planner CLI attempt 2 starting (backend=%s model=%s prompt_chars=%s inline_schema=%s timeout=%s)",
                llm_config.backend.value,
                llm_config.model,
                len(retry_prompt),
                use_inline_schema,
                effective_hard_timeout_seconds or llm_config.cli_timeout,
            )
            retry_result = CLIModelClient(llm_config).run_structured_prompt(
                prompt=retry_prompt,
                working_dir=working_dir,
                schema=None if use_inline_schema else schema,
                system_prompt=system_prompt,
                allow_edits=False,
                hard_timeout_seconds=effective_hard_timeout_seconds,
            )
            if retry_result.success and retry_result.parsed_json:
                retry_validation_error = (
                    _validate_planner_parsed_json(
                        retry_result.parsed_json,
                        schema=schema,
                        required_keys=["summary", "relevant_files", "rollout_briefs"],
                    )
                    if self.config.planning.enable_planner_output_validation
                    else None
                )
                if not retry_validation_error:
                    logger.info(
                        "Planner CLI attempt 2 succeeded in %.1fs (tokens=%s)",
                        retry_result.duration_seconds,
                        extract_total_tokens(retry_result.usage),
                    )
                    return retry_result.parsed_json, extract_total_tokens(retry_result.usage)
                logger.warning(
                    "Planner CLI attempt 2 produced schema-invalid output (%s); "
                    "falling through to heuristic fallback",
                    retry_validation_error,
                )
                raise RuntimeError(
                    "Planner CLI returned schema-invalid JSON after retry: "
                    + retry_validation_error
                )
            if retry_result.error and _looks_like_timeout_or_stall_error(retry_result.error):
                logger.warning("Planner CLI attempt 2 stalled: %s", retry_result.error)
                raise RuntimeError(retry_result.error)

            raw_preview = (
                retry_result.raw_output or result.raw_output or retry_result.text or result.text
            )[:1200]
            logger.warning("Planner CLI returned unparseable output: %s", raw_preview)
            raise RuntimeError(
                retry_result.error or result.error or "Planner CLI did not return structured JSON."
            )

        llm = LLMClient(llm_config, temperature_override=0.0)
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=prompt),
        ]
        if self.config.enable_planning_tool:
            response = llm.chat(
                messages=messages,
                tools=[_PLAN_TOOL],
                temperature=0.0,
            )
            if not response.tool_calls or response.tool_calls[0].name != "submit_plan":
                raise RuntimeError("Planner did not return a structured plan.")
            return response.tool_calls[0].arguments, llm.total_tokens_used

        response = llm.chat(
            messages=messages,
            temperature=0.0,
        )
        payload = self._parse_plan_json(response.content or "")
        if not payload:
            raise RuntimeError("Planner did not return JSON content.")
        return payload, llm.total_tokens_used

    def _validate_files(
        self,
        candidate_paths: list[str],
        allowed: set[str],
        fallback: list[str],
    ) -> list[str]:
        validated = [path for path in candidate_paths if path in allowed]
        if validated:
            return list(dict.fromkeys(validated))
        return list(dict.fromkeys(fallback))

    def _truncate_block(self, text: str, max_lines: int) -> str:
        content = text.strip()
        if not self.config.use_concise_prompts or not content:
            return content
        lines = content.splitlines()
        if len(lines) <= max_lines:
            return content
        remaining = len(lines) - max_lines
        return "\n".join(lines[:max_lines] + [f"... ({remaining} more lines omitted)"])

    def _truncate_diagnostic_block(self, text: str, max_lines: int) -> str:
        content = normalize_terminal_output(text).strip()
        if not self.config.use_concise_prompts or not content:
            return content
        lines = content.splitlines()
        if len(lines) <= max_lines:
            return content

        informative_indices = [
            index
            for index, line in enumerate(lines)
            if self._looks_informative_diagnostic_line(line)
        ]
        if informative_indices:
            end = informative_indices[-1] + 1
            start = max(0, end - max_lines)
            excerpt = list(lines[start:end])
            if start > 0:
                excerpt.insert(0, f"... ({start} earlier lines omitted)")
            remaining = len(lines) - end
            if remaining > 0:
                excerpt.append(f"... ({remaining} more lines omitted)")
            return "\n".join(excerpt)

        remaining = len(lines) - max_lines
        return "\n".join(lines[:max_lines] + [f"... ({remaining} more lines omitted)"])

    def _looks_informative_diagnostic_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        lowered = stripped.lower()
        if ".py:" in stripped or stripped.startswith(("E   ", 'File "')):
            return True
        return any(
            token in lowered
            for token in (
                "traceback",
                "error collecting",
                "importerror while loading conftest",
                "importerror while importing test module",
                "syntaxerror:",
                "nameerror:",
                "typeerror:",
                "attributeerror:",
                "assertionerror:",
                "valueerror:",
                "keyerror:",
                "indexerror:",
                "runtimeerror:",
                "modulenotfounderror:",
                "notimplementederror:",
                "exception:",
            )
        )

    def _parse_plan_json(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        if not stripped:
            return {}

        candidates = [stripped]
        if stripped.startswith("```"):
            stripped_fence = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", stripped)
            stripped_fence = re.sub(r"\n?```$", "", stripped_fence).strip()
            candidates.append(stripped_fence)

        first_brace = stripped.find("{")
        last_brace = stripped.rfind("}")
        if 0 <= first_brace < last_brace:
            candidates.append(stripped[first_brace : last_brace + 1])

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return {}
