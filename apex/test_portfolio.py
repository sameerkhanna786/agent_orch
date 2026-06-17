"""
Normalization and promotion heuristics for synthetic test portfolios.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Optional

from .controller_policy import default_test_inventory_language, infer_test_inventory_framework
from .core.generated_tests import normalize_generated_test_content

_KNOWN_TEST_STRATEGIES = {
    "regression",
    "contract",
    "edge",
    "negative",
    "property",
    "metamorphic",
    "differential",
    "fuzz_seed",
    "exploratory",
}

_KNOWN_MATERIALIZATION_MODES = {
    "auto",
    "append",
    "replace",
}

_STRATEGY_ALIASES = {
    "api": "contract",
    "api_contract": "contract",
    "boundary": "edge",
    "contract_api": "contract",
    "differential_test": "differential",
    "edge_case": "edge",
    "error": "negative",
    "failure": "negative",
    "fuzz": "fuzz_seed",
    "fuzzing": "fuzz_seed",
    "metamorphic_relation": "metamorphic",
    "negative_case": "negative",
    "property_based": "property",
    "property_test": "property",
    "public_contract": "contract",
    "reference": "differential",
    "repro": "regression",
    "reproduction": "regression",
}

_CONTRACT_AXIS_ALIASES = {
    "boundary": "missing_boundary",
    "canonical": "positive_path",
    "empty": "missing_boundary",
    "error": "negative_malformed",
    "happy_path": "positive_path",
    "invalid": "negative_malformed",
    "malformed": "negative_malformed",
    "missing": "missing_boundary",
    "multi": "multi_ordering",
    "multiple": "multi_ordering",
    "negative": "negative_malformed",
    "none": "missing_boundary",
    "null": "missing_boundary",
    "order": "multi_ordering",
    "ordering": "multi_ordering",
    "positive": "positive_path",
    "success": "positive_path",
}

_KNOWN_CONTRACT_AXES = {
    "positive_path",
    "missing_boundary",
    "negative_malformed",
    "multi_ordering",
    "property",
    "metamorphic",
    "differential",
    "fuzz_seed",
}

_DEFAULT_TEST_GENERATION_PIPELINE_STAGES = [
    "context_hypothesis",
    "pass_then_invert",
    "execution_feedback",
    "mutation_discrimination",
    "dual_version_verification",
]

_SOURCE_ALIASES = {
    "bug_report": "issue",
    "doc": "docs",
    "documentation": "docs",
    "example": "examples",
    "existing_test": "existing_tests",
    "existing_tests": "existing_tests",
    "failing_tests": "traceback",
    "localizer": "localization",
    "public_api": "types",
    "reproducer": "reproduction",
    "stacktrace": "traceback",
    "test_context": "existing_tests",
    "type_hints": "types",
}

_CONTRACT_AXIS_PATTERNS: dict[str, re.Pattern[str]] = {
    "missing_boundary": re.compile(
        r"\b(missing|absent|empty|null|none|not\s+present|not\s+found|default|fallback|zero)\b|\[\]",
        re.IGNORECASE,
    ),
    "negative_malformed": re.compile(
        r"\b(malformed|invalid|error|exception|reject|skip|ignore|fail(?:ed)?\s+closed|wrong\s+shape|negative)\b",
        re.IGNORECASE,
    ),
    "multi_ordering": re.compile(
        r"\b(multi|multiple|ordered|ordering|preserve\s+order|sequence|all\s+valid\s+values|all\s+values|list[-\s]?valued)\b",
        re.IGNORECASE,
    ),
}

_POSITIVE_PATH_PATTERN = re.compile(
    r"\b("
    r"accept(?:s|ed)?"
    r"|allow(?:s|ed)?"
    r"|canonical"
    r"|emit(?:s|ted)?"
    r"|expose(?:s|d)?"
    r"|happy[_\s-]?path"
    r"|pass(?:es|ed)?\s+through"
    r"|positive"
    r"|preserve(?:s|d)?"
    r"|reach(?:es|ed)?"
    r"|return(?:s|ed)?"
    r"|success(?:ful(?:ly)?)?"
    r"|unfiltered"
    r"|valid"
    r")\b",
    re.IGNORECASE,
)

_SCHEMA_DISCRIMINATOR_TOKENS = frozenset(
    {
        "discriminator",
        "format",
        "kind",
        "mode",
        "sentinel",
        "tag",
        "type",
        "variant",
    }
)

_SCHEMA_CODELIKE_TOKEN_PATTERN = re.compile(
    r"""
    (?:
        ['"`](?P<quoted>[A-Za-z_][A-Za-z0-9_]*)['"`]\s*:
      | \.(?P<dotted>[A-Za-z_][A-Za-z0-9_]*)
      | \b(?P<bare>[A-Za-z_][A-Za-z0-9_]*)\s*:
    )
    """,
    re.VERBOSE,
)

_SCHEMA_TEXT_TOKEN_PATTERN = re.compile(
    r"""
    `(?P<backtick>[A-Za-z_][A-Za-z0-9_]*)`
    | (?:[A-Za-z_][A-Za-z0-9_]*\.)+(?P<field>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE,
)

_FIELD_PATH_TEXT_PATTERN = re.compile(
    r"""
    `(?P<backtick>(?:[a-z][a-z0-9_]*\.)+[a-z][a-z0-9_]*)`
    | (?<![\w/])(?P<plain>(?:[a-z][a-z0-9_]*\.){1,3}[a-z][a-z0-9_]*)(?![\w/])
    """,
    re.VERBOSE,
)

_FIELD_PATH_CONTEXT_PATTERN = re.compile(
    r"\b(field|leaf|path|payload|shape|object|dict|json|container|content|missing|malformed|nested)\b|\b(?:sibling|leaf|container)\s+key\b",
    re.IGNORECASE,
)

_WEAK_FIELD_PATH_CONTEXT_PATTERN = re.compile(
    r"\b(field|leaf|payload|shape|object|dict|json|container|content|missing|malformed|nested|response|request|body|data)\b|\b(?:sibling|leaf|container)\s+key\b",
    re.IGNORECASE,
)

_FIELD_PATH_SELF_CONTEXT_PATTERN = re.compile(
    r"\b(value|content|payload|data|body|fields?|json|dict|object|container|nested|response|request|config|metadata|meta)\b",
    re.IGNORECASE,
)

_WEAK_FIELD_PATH_TERMINALS = frozenset(
    {
        "count",
        "index",
        "key",
        "keys",
        "length",
        "name",
        "names",
        "path",
        "paths",
        "sep",
        "size",
        "type",
        "value",
        "values",
    }
)

_FIELD_PATH_ROOT_ALIASES = {
    "req": "request",
    "request": "request",
    "res": "response",
    "resp": "response",
    "response": "response",
}

_UNEXPECTED_SIBLING_KEY_HINT_PATTERN = re.compile(
    r"\b("
    r"wrong[_\s-]?key"
    r"|unexpected\s+(?:sibling\s+)?key"
    r"|unexpected\s+field"
    r"|other[_\s-]?key"
    r"|invalid[_\s-]?key"
    r"|unknown[_\s-]?key"
    r"|extra[_\s-]?key"
    r"|bad[_\s-]?key"
    r")\b",
    re.IGNORECASE,
)

_KNOWN_SOURCE_FILE_EXTENSIONS = {
    ".bash",
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
    ".m",
    ".mm",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".scala",
    ".sh",
    ".swift",
    ".ts",
    ".tsx",
    ".zsh",
}

_ISSUE_INTERFACE_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?im)^\s*Name:\s*`([^`]+)`\s*$"),
    re.compile(r"(?i)\b(?:method|function|api|command|endpoint|class|property|field)\s+`([^`]+)`"),
    re.compile(
        r"(?i)\b(?:method|function|api|command|endpoint|class|property|field)\s+named\s+`([^`]+)`"
    ),
    re.compile(r"(?i)`([^`]+)`\s+(?:method|function|api|command|endpoint|property|field)\b"),
    re.compile(
        r"(?i)\b(?:provide|expose|introduce|add|implement)(?:s|d)?\s+(?:a|an)\s+"
        r"(?:method|function|api|command|endpoint|class|property|field)\s+named\s+`([^`]+)`"
    ),
)

_ISSUE_INTERFACE_CONTAINER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\binside\s+the\s+`([^`]+)`\s+(?:class|struct|interface|type)\b"),
    re.compile(r"(?i)\bon\s+the\s+`([^`]+)`\s+(?:class|struct|interface|type)\b"),
    re.compile(r"(?i)\bwithin\s+the\s+`([^`]+)`\s+(?:class|struct|interface|type)\b"),
)
_REQUIREMENT_LINE_PATTERN = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(.+?)\s*$")
_OBJECTIVE_STATUS_ORDER = {
    "planned": 0,
    "draft": 1,
    "candidate": 2,
    "verified": 3,
}
_PASS_THEN_INVERT_COMPLETE_STATUSES = frozenset(
    {"complete", "done", "pass_then_invert", "verified"}
)
_DESIGN_MILESTONE_TITLES = {
    "milestone_direct_regression": "Direct Regression Capture",
    "milestone_contract_hardening": "Contract Hardening",
}


def _dedupe_strings(values: list[Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        ordered.append(text)
        seen.add(text)
    return ordered


def _clean_issue_contract_symbol(value: Any) -> str:
    text = str(value or "").strip().strip("`'\"")
    if not text:
        return ""
    text = re.sub(r"\(.*\)$", "", text).strip()
    if not text or "/" in text:
        return ""
    lowered = text.lower()
    if lowered in {
        "method",
        "function",
        "api",
        "command",
        "endpoint",
        "class",
        "property",
        "field",
    }:
        return ""
    return text


def extract_issue_contract_targets(issue_description: str) -> list[str]:
    text = str(issue_description or "")
    if not text.strip():
        return []

    containers = _dedupe_strings(
        [
            cleaned
            for pattern in _ISSUE_INTERFACE_CONTAINER_PATTERNS
            for match in pattern.findall(text)
            for cleaned in [_clean_issue_contract_symbol(match)]
            if cleaned
        ]
    )
    names = _dedupe_strings(
        [
            cleaned
            for pattern in _ISSUE_INTERFACE_NAME_PATTERNS
            for match in pattern.findall(text)
            for cleaned in [_clean_issue_contract_symbol(match)]
            if cleaned
        ]
    )
    if not names:
        return []

    candidates: list[str] = []
    for name in names:
        if not any(token in name for token in (".", "::", "#")):
            candidates.extend(f"{container}.{name}" for container in containers[:2])
        candidates.append(name)

    ordered: list[str] = []
    seen_normalized: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_contract_target(candidate)
        if not normalized or normalized in seen_normalized:
            continue
        ordered.append(candidate)
        seen_normalized.add(normalized)
    return ordered


def _sanitize_design_identifier(value: Any, *, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    return normalized or fallback


def _normalize_requirement_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    bullet_match = _REQUIREMENT_LINE_PATTERN.match(text)
    if bullet_match:
        text = bullet_match.group(1).strip()
    text = re.sub(
        r"^(problem statement|acceptance requirements?|interface specification)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip(" -")


def _extract_requirement_lines(text: str) -> list[str]:
    requirements: list[str] = []
    for raw_line in str(text or "").splitlines():
        match = _REQUIREMENT_LINE_PATTERN.match(raw_line)
        if not match:
            continue
        candidate = _normalize_requirement_text(match.group(1))
        if not candidate:
            continue
        lowered = candidate.lower()
        if lowered.startswith(("title:", "location:", "name:", "type:", "repository:")):
            continue
        if len(candidate) < 8:
            continue
        requirements.append(candidate)
    return _dedupe_strings(requirements)


def _first_problem_statement(text: str) -> str:
    for raw_line in str(text or "").splitlines():
        candidate = raw_line.strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        if lowered.startswith(
            ("#", "problem statement", "acceptance requirements", "interface specification")
        ):
            continue
        if lowered.startswith(
            ("name:", "type:", "location:", "repository:", "repository language:")
        ):
            continue
        return candidate
    return ""


def _axis_requirement_line(axis: str, interface_targets: list[str]) -> str:
    surface = ", ".join(interface_targets[:2]) if interface_targets else "the named interface"
    human_axis = str(axis or "").replace("_", " ").strip() or "contract"
    return f"Cover {human_axis} behavior on {surface}."


def _merge_task_contract_payload(
    primary: dict[str, Any] | None,
    fallback: dict[str, Any] | None,
) -> dict[str, Any]:
    primary_payload = dict(primary or {})
    fallback_payload = dict(fallback or {})
    problem_statement = str(
        primary_payload.get("problem_statement")
        or primary_payload.get("summary")
        or fallback_payload.get("problem_statement")
        or fallback_payload.get("summary")
        or ""
    ).strip()
    acceptance_requirements = _dedupe_strings(
        list(primary_payload.get("acceptance_requirements") or [])
        + list(primary_payload.get("requirements") or [])
        + list(fallback_payload.get("acceptance_requirements") or [])
        + list(fallback_payload.get("requirements") or [])
    )
    interface_specification = _dedupe_strings(
        list(primary_payload.get("interface_specification") or [])
        + list(primary_payload.get("interface_targets") or [])
        + list(fallback_payload.get("interface_specification") or [])
        + list(fallback_payload.get("interface_targets") or [])
    )
    if not problem_statement and not acceptance_requirements and not interface_specification:
        return {}
    return {
        "problem_statement": problem_statement,
        "acceptance_requirements": acceptance_requirements,
        "interface_specification": interface_specification,
    }


def _normalize_task_contract(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        text = str(raw).strip()
        if not text:
            return {}
        return {
            "problem_statement": text,
            "acceptance_requirements": [],
            "interface_specification": [],
        }
    payload = dict(raw or {}) if isinstance(raw, dict) else {}
    return _merge_task_contract_payload(payload, {})


def derive_test_generation_plan(
    issue_description: str,
    *,
    issue_summary: str = "",
    success_criteria: Optional[list[str]] = None,
    behavioral_obligations: Optional[list[str]] = None,
    interface_targets: Optional[list[str]] = None,
    required_axes: Optional[list[str]] = None,
) -> dict[str, Any]:
    normalized_targets = _dedupe_strings(
        list(interface_targets or []) or extract_issue_contract_targets(issue_description)
    )
    normalized_axes = _normalize_required_contract_axes(required_axes)
    acceptance_requirements = _dedupe_strings(
        [_normalize_requirement_text(value) for value in list(behavioral_obligations or [])]
        + _extract_requirement_lines(issue_description)
        + [_normalize_requirement_text(value) for value in list(success_criteria or [])]
    )
    if not acceptance_requirements and normalized_axes:
        acceptance_requirements = [
            _axis_requirement_line(axis, normalized_targets) for axis in normalized_axes
        ]
    problem_statement = str(issue_summary or "").strip() or _first_problem_statement(
        issue_description
    )
    if not problem_statement and acceptance_requirements:
        problem_statement = acceptance_requirements[0]

    task_contract = {
        "problem_statement": problem_statement,
        "acceptance_requirements": acceptance_requirements,
        "interface_specification": normalized_targets,
    }

    derived_objectives: list[dict[str, Any]] = []
    requirement_values = acceptance_requirements or (
        [problem_statement] if problem_statement else []
    )
    for index, requirement in enumerate(requirement_values, start=1):
        objective_axes = _normalize_required_contract_axes(
            infer_required_contract_axes_from_texts([requirement])
        )
        if not objective_axes:
            objective_axes = (
                [normalized_axes[0]] if index == 1 and normalized_axes else list(normalized_axes)
            )
        if not objective_axes:
            objective_axes = ["positive_path"]
        milestone_id = (
            "milestone_contract_hardening"
            if any(axis != "positive_path" for axis in objective_axes)
            else "milestone_direct_regression"
        )
        derived_objectives.append(
            {
                "objective_id": f"objective_{index}",
                "milestone_id": milestone_id,
                "objective": requirement,
                "acceptance_requirements": [requirement],
                "interface_specification": list(normalized_targets),
                "contract_targets": list(normalized_targets),
                "contract_axes": list(objective_axes),
                "objective_status": "planned",
            }
        )

    grouped_objectives: dict[str, list[dict[str, Any]]] = {}
    for objective in derived_objectives:
        grouped_objectives.setdefault(
            str(objective.get("milestone_id") or "milestone_direct_regression"),
            [],
        ).append(objective)

    milestones: list[dict[str, Any]] = []
    for milestone_id, objectives in grouped_objectives.items():
        title = (
            _DESIGN_MILESTONE_TITLES.get(milestone_id)
            or milestone_id.replace(
                "_",
                " ",
            ).title()
        )
        milestone_requirements = _dedupe_strings(
            [
                requirement
                for objective in objectives
                for requirement in list(objective.get("acceptance_requirements") or [])
            ]
        )
        milestones.append(
            {
                "milestone_id": milestone_id,
                "title": title,
                "summary": title,
                "acceptance_requirements": milestone_requirements,
                "objective_ids": [
                    str(objective.get("objective_id") or "").strip()
                    for objective in objectives
                    if str(objective.get("objective_id") or "").strip()
                ],
                "validation_level": "strict",
                "pipeline_stages": list(_DEFAULT_TEST_GENERATION_PIPELINE_STAGES),
            }
        )

    return {
        "task_contract": task_contract if any(task_contract.values()) else {},
        "milestones": milestones,
        "test_objectives": derived_objectives,
    }


def normalize_test_generation_design_payload(
    payload: Any,
    *,
    issue_description: str,
    issue_summary: str = "",
    success_criteria: Optional[list[str]] = None,
    behavioral_obligations: Optional[list[str]] = None,
    interface_targets: Optional[list[str]] = None,
    required_axes: Optional[list[str]] = None,
) -> dict[str, Any]:
    raw_payload = dict(payload or {}) if isinstance(payload, dict) else {}
    derived_plan = derive_test_generation_plan(
        issue_description,
        issue_summary=issue_summary,
        success_criteria=list(success_criteria or []),
        behavioral_obligations=list(behavioral_obligations or []),
        interface_targets=list(interface_targets or []),
        required_axes=list(required_axes or []),
    )
    task_contract = _merge_task_contract_payload(
        _normalize_task_contract(raw_payload.get("task_contract") or raw_payload.get("contract")),
        dict(derived_plan.get("task_contract") or {}),
    )
    test_objectives = [
        _normalize_test_generation_objective(
            item,
            index=index,
            task_contract=task_contract,
        )
        for index, item in enumerate(
            list(
                raw_payload.get("test_objectives")
                or raw_payload.get("objectives")
                or derived_plan.get("test_objectives")
                or []
            ),
            start=1,
        )
    ]
    derived_milestones_by_id = {
        str(item.get("milestone_id") or "").strip(): dict(item)
        for item in list(derived_plan.get("milestones") or [])
        if isinstance(item, dict) and str(item.get("milestone_id") or "").strip()
    }
    raw_milestones = list(raw_payload.get("milestones") or [])
    milestones = [
        _normalize_test_generation_milestone(item, index=index)
        for index, item in enumerate(
            raw_milestones or list(derived_plan.get("milestones") or []),
            start=1,
        )
    ]
    milestone_order = [
        str(item.get("milestone_id") or "").strip()
        for item in milestones
        if str(item.get("milestone_id") or "").strip()
    ]
    milestones_by_id = {
        str(item.get("milestone_id") or "").strip(): dict(item)
        for item in milestones
        if str(item.get("milestone_id") or "").strip()
    }
    for objective in test_objectives:
        milestone_id = str(objective.get("milestone_id") or "").strip()
        if not milestone_id:
            continue
        if milestone_id not in milestones_by_id:
            fallback = dict(derived_milestones_by_id.get(milestone_id) or {})
            fallback.setdefault("milestone_id", milestone_id)
            fallback.setdefault("objective_ids", [])
            fallback.setdefault("acceptance_requirements", [])
            normalized = _normalize_test_generation_milestone(
                fallback or {"milestone_id": milestone_id},
                index=len(milestone_order) + 1,
            )
            milestones_by_id[milestone_id] = normalized
            milestone_order.append(milestone_id)
        milestone = milestones_by_id[milestone_id]
        milestone["objective_ids"] = _dedupe_strings(
            list(milestone.get("objective_ids") or [])
            + [str(objective.get("objective_id") or "").strip()]
        )
        milestone["acceptance_requirements"] = _dedupe_strings(
            list(milestone.get("acceptance_requirements") or [])
            + list(objective.get("acceptance_requirements") or [])
        )
        if not str(milestone.get("validation_level") or "").strip():
            milestone["validation_level"] = "strict"
        if not list(milestone.get("pipeline_stages") or []):
            milestone["pipeline_stages"] = list(_DEFAULT_TEST_GENERATION_PIPELINE_STAGES)
    normalized_milestones = [
        milestones_by_id[milestone_id]
        for milestone_id in milestone_order
        if milestone_id in milestones_by_id
    ]
    return {
        "task_contract": task_contract,
        "milestones": normalized_milestones,
        "test_objectives": test_objectives,
    }


def _normalize_pass_then_invert(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, bool):
        return {
            "attempted": bool(raw),
            "status": "complete" if raw else "not_attempted",
        }
    if isinstance(raw, str):
        status = str(raw).strip().lower()
        if not status:
            return {}
        return {
            "attempted": status not in {"", "not_attempted", "none"},
            "status": status,
        }
    if not isinstance(raw, dict):
        return {}
    payload = dict(raw)
    status = str(payload.get("status") or "").strip().lower()
    passing_variant_summary = str(
        payload.get("passing_variant_summary")
        or payload.get("pass_summary")
        or payload.get("passing_summary")
        or ""
    ).strip()
    inversion_summary = str(
        payload.get("inversion_summary")
        or payload.get("invert_summary")
        or payload.get("failing_oracle_summary")
        or ""
    ).strip()
    execution_feedback_summary = str(
        payload.get("execution_feedback_summary") or payload.get("feedback_summary") or ""
    ).strip()
    attempted = bool(
        payload.get("attempted")
        or status
        or passing_variant_summary
        or inversion_summary
        or execution_feedback_summary
    )
    normalized = {
        "attempted": attempted,
        "status": status or ("complete" if attempted else "not_attempted"),
        "passing_variant_summary": passing_variant_summary,
        "inversion_summary": inversion_summary,
        "execution_feedback_summary": execution_feedback_summary,
    }
    if not attempted and not any(value for key, value in normalized.items() if key != "attempted"):
        return {}
    return normalized


def _pass_then_invert_complete(value: Any) -> bool:
    normalized = _normalize_pass_then_invert(value)
    if not normalized:
        return False
    status = str(normalized.get("status") or "").strip().lower()
    execution_feedback_summary = str(normalized.get("execution_feedback_summary") or "").strip()
    return bool(
        execution_feedback_summary
        and (
            status in _PASS_THEN_INVERT_COMPLETE_STATUSES
            or (normalized.get("passing_variant_summary") and normalized.get("inversion_summary"))
        )
    )


def _design_metadata_source_is_explicit(
    entry: dict[str, Any],
    field_name: str,
) -> bool:
    sources = dict(entry.get("design_metadata_sources") or {})
    return str(sources.get(field_name) or "").strip().lower() == "explicit"


def _artifact_design_metadata_gaps(entry: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    if not str(entry.get("milestone_id") or "").strip():
        gaps.append("milestone_id")
    elif not _design_metadata_source_is_explicit(entry, "milestone_id"):
        gaps.append("milestone_id")
    if not str(entry.get("objective_id") or "").strip():
        gaps.append("objective_id")
    elif not _design_metadata_source_is_explicit(entry, "objective_id"):
        gaps.append("objective_id")
    if not str(entry.get("objective") or "").strip():
        gaps.append("objective")
    elif not _design_metadata_source_is_explicit(entry, "objective"):
        gaps.append("objective")
    if not list(entry.get("acceptance_requirements") or []):
        gaps.append("acceptance_requirements")
    elif not _design_metadata_source_is_explicit(entry, "acceptance_requirements"):
        gaps.append("acceptance_requirements")
    if not list(entry.get("interface_specification") or []):
        gaps.append("interface_specification")
    elif not _design_metadata_source_is_explicit(entry, "interface_specification"):
        gaps.append("interface_specification")
    if not str(entry.get("oracle_origin") or "").strip():
        gaps.append("oracle_origin")
    elif not _design_metadata_source_is_explicit(entry, "oracle_origin"):
        gaps.append("oracle_origin")

    pass_then_invert = _normalize_pass_then_invert(entry.get("pass_then_invert"))
    if not pass_then_invert:
        gaps.append("pass_then_invert")
        return gaps

    if not str(pass_then_invert.get("passing_variant_summary") or "").strip():
        gaps.append("pass_then_invert.passing_variant_summary")
    if not str(pass_then_invert.get("inversion_summary") or "").strip():
        gaps.append("pass_then_invert.inversion_summary")
    if not str(pass_then_invert.get("execution_feedback_summary") or "").strip():
        gaps.append("pass_then_invert.execution_feedback_summary")
    return gaps


def _default_artifact_objective_id(
    artifact_id: str,
    contract_targets: list[str],
    *,
    index: int,
) -> str:
    for target in contract_targets:
        normalized = _normalize_contract_target(target)
        if normalized:
            token = _sanitize_design_identifier(
                normalized,
                fallback=f"artifact_{index}",
            )
            return f"objective_{token}"
    return f"objective_{_sanitize_design_identifier(artifact_id, fallback=f'artifact_{index}')}"


def _default_artifact_milestone_id(contract_axes: list[str]) -> str:
    normalized_axes = _normalize_required_contract_axes(contract_axes)
    if any(axis != "positive_path" for axis in normalized_axes):
        return "milestone_contract_hardening"
    return "milestone_direct_regression"


def _normalize_test_generation_objective(
    raw: Any,
    *,
    index: int,
    task_contract: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(raw or {}) if isinstance(raw, dict) else {"objective": str(raw or "")}
    objective = str(
        payload.get("objective") or payload.get("summary") or payload.get("title") or ""
    ).strip()
    acceptance_requirements = _dedupe_strings(
        list(payload.get("acceptance_requirements") or [])
        + ([payload.get("acceptance_requirement")] if payload.get("acceptance_requirement") else [])
    )
    if not acceptance_requirements and objective:
        acceptance_requirements = [objective]
    interface_specification = _dedupe_strings(
        list(payload.get("interface_specification") or [])
        + list(payload.get("contract_targets") or [])
        + list(task_contract.get("interface_specification") or [])
    )
    contract_targets = _dedupe_strings(
        list(payload.get("contract_targets") or []) + list(interface_specification)
    )
    contract_axes = _normalize_contract_axes(
        list(payload.get("contract_axes") or [])
        or infer_required_contract_axes_from_texts(acceptance_requirements or [objective])
    )
    objective_id = str(payload.get("objective_id") or "").strip() or f"objective_{index}"
    milestone_id = str(payload.get("milestone_id") or "").strip() or _default_artifact_milestone_id(
        contract_axes
    )
    return {
        "objective_id": objective_id,
        "milestone_id": milestone_id,
        "objective": objective or (acceptance_requirements[0] if acceptance_requirements else ""),
        "acceptance_requirements": acceptance_requirements,
        "interface_specification": interface_specification,
        "contract_targets": contract_targets,
        "contract_axes": contract_axes,
        "artifact_ids": _dedupe_strings(list(payload.get("artifact_ids") or [])),
        "objective_status": str(
            payload.get("objective_status") or payload.get("status") or "planned"
        )
        .strip()
        .lower(),
    }


def _normalize_test_generation_milestone(
    raw: Any,
    *,
    index: int,
) -> dict[str, Any]:
    payload = dict(raw or {}) if isinstance(raw, dict) else {"title": str(raw or "")}
    milestone_id = str(payload.get("milestone_id") or "").strip() or f"milestone_{index}"
    title = str(payload.get("title") or payload.get("summary") or "").strip()
    if not title:
        title = (
            _DESIGN_MILESTONE_TITLES.get(milestone_id)
            or milestone_id.replace(
                "_",
                " ",
            ).title()
        )
    return {
        "milestone_id": milestone_id,
        "title": title,
        "summary": str(payload.get("summary") or title).strip(),
        "acceptance_requirements": _dedupe_strings(
            list(payload.get("acceptance_requirements") or [])
        ),
        "objective_ids": _dedupe_strings(list(payload.get("objective_ids") or [])),
        "validation_level": str(payload.get("validation_level") or "strict").strip().lower(),
        "pipeline_stages": _dedupe_strings(
            list(payload.get("pipeline_stages") or _DEFAULT_TEST_GENERATION_PIPELINE_STAGES)
        ),
    }


def _objective_status_rank(value: Any) -> int:
    return _OBJECTIVE_STATUS_ORDER.get(str(value or "").strip().lower(), 0)


def _aggregate_objective_status(values: list[str]) -> str:
    normalized = [str(value or "").strip().lower() for value in values if str(value or "").strip()]
    if not normalized:
        return "planned"
    if all(value == "verified" for value in normalized):
        return "verified"
    if all(
        _objective_status_rank(value) >= _OBJECTIVE_STATUS_ORDER["candidate"]
        for value in normalized
    ):
        return "candidate"
    if any(
        _objective_status_rank(value) >= _OBJECTIVE_STATUS_ORDER["draft"] for value in normalized
    ):
        return "draft"
    return "planned"


def _entry_dual_version_verified(entry: dict[str, Any]) -> bool:
    validation = dict(entry.get("validation") or {})
    return bool(validation.get("dual_version_verified") or entry.get("dual_version_verified"))


def _entry_mutation_discrimination_passed(entry: dict[str, Any]) -> bool:
    validation = dict(entry.get("validation") or {})
    if "mutation_discrimination_passed" in validation:
        return bool(validation.get("mutation_discrimination_passed"))
    plausible_mutant_count = int(validation.get("plausible_mutant_count") or 0)
    if plausible_mutant_count > 0:
        return (
            bool(validation.get("mutation_signal_measured"))
            and int(validation.get("plausible_mutant_survived_count") or 0) == 0
        )
    normalized_language = (
        str(validation.get("language") or entry.get("language") or "").strip().lower()
    )
    if (
        normalized_language == "python"
        and bool(validation.get("execution_succeeded"))
        and bool(validation.get("execution_targeted_supported"))
    ):
        return False
    if bool(validation.get("mutation_signal_measured")):
        return _clamp(validation.get("mutation_signal")) > 0.0
    return _entry_dual_version_verified(entry)


def _entry_effective_objective_status(entry: dict[str, Any]) -> str:
    explicit = str(entry.get("objective_status") or "").strip().lower()
    validation = dict(entry.get("validation") or {})
    promotion_status = str(entry.get("promotion_status") or "").strip().lower()
    if (
        _entry_dual_version_verified(entry)
        and _entry_mutation_discrimination_passed(entry)
        and not _artifact_design_metadata_gaps(entry)
        and promotion_status == "promoted"
    ):
        return "verified"
    if promotion_status in {"promoted", "candidate_public"} or validation.get("baseline_preserved"):
        return "candidate"
    if explicit:
        return explicit
    if str(entry.get("content") or "").strip():
        return "draft"
    return "planned"


def _clamp(value: Any, *, lower: float = 0.0, upper: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lower
    return max(lower, min(upper, number))


def _normalize_relative_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return ""
    if text.startswith("/") or re.match(r"^[A-Za-z]:/", text):
        return ""
    parts = [part for part in Path(text).parts if part not in {"", "."}]
    if any(part == ".." for part in parts):
        return ""
    return Path(*parts).as_posix() if parts else ""


def _normalize_strategy(value: Any, *, raw: Optional[dict[str, Any]] = None) -> str:
    payload = dict(raw or {})
    for candidate in (
        value,
        payload.get("strategy"),
        payload.get("category"),
        payload.get("generator_role"),
        payload.get("summary"),
    ):
        token = str(candidate or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not token:
            continue
        if token in _STRATEGY_ALIASES:
            token = _STRATEGY_ALIASES[token]
        if token in _KNOWN_TEST_STRATEGIES:
            return token
        for alias, normalized in sorted(
            _STRATEGY_ALIASES.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if alias in token:
                return normalized
        for strategy in sorted(_KNOWN_TEST_STRATEGIES, key=len, reverse=True):
            if strategy in token:
                return strategy
    if payload.get("properties"):
        return "property"
    if payload.get("metamorphic_relations"):
        return "metamorphic"
    if payload.get("reference_targets") or payload.get("reference_target"):
        return "differential"
    if payload.get("fuzz_seeds") or payload.get("fuzz_seed"):
        return "fuzz_seed"
    return "regression"


def _normalize_materialization_mode(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if token in {"file", "new_file", "write"}:
        token = "replace"
    if token in _KNOWN_MATERIALIZATION_MODES:
        return token
    return "auto"


def _normalize_contract_sources(values: list[Any]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not token:
            continue
        normalized.append(_SOURCE_ALIASES.get(token, token))
    return _dedupe_strings(normalized)


def _normalize_contract_axes(values: list[Any]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not token:
            continue
        if token in _CONTRACT_AXIS_ALIASES:
            normalized.append(_CONTRACT_AXIS_ALIASES[token])
            continue
        if token in _KNOWN_CONTRACT_AXES:
            normalized.append(token)
            continue
        for alias, axis in sorted(
            _CONTRACT_AXIS_ALIASES.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if alias in token:
                normalized.append(axis)
                break
    return _dedupe_strings(normalized)


def _normalize_required_contract_axes(values: Optional[list[Any]]) -> list[str]:
    return sorted({axis for axis in _normalize_contract_axes(list(values or [])) if axis})


def infer_required_contract_axes_from_texts(values: list[Any]) -> list[str]:
    text = "\n".join(
        str(value or "").strip() for value in values if str(value or "").strip()
    ).strip()
    axes: set[str] = set()
    for axis, pattern in _CONTRACT_AXIS_PATTERNS.items():
        if pattern.search(text):
            axes.add(axis)
    if _POSITIVE_PATH_PATTERN.search(text) or not axes:
        axes.add("positive_path")
    return sorted(axes)


def _schema_indicator_tokens_in_code(text: Any) -> set[str]:
    tokens: set[str] = set()
    for match in _SCHEMA_CODELIKE_TOKEN_PATTERN.finditer(str(text or "")):
        token = (
            match.group("quoted") or match.group("dotted") or match.group("bare") or ""
        ).lower()
        if token in _SCHEMA_DISCRIMINATOR_TOKENS:
            tokens.add(token)
    return tokens


def _schema_indicator_tokens_in_text(text: Any) -> set[str]:
    tokens: set[str] = set()
    for match in _SCHEMA_TEXT_TOKEN_PATTERN.finditer(str(text or "")):
        token = (match.group("backtick") or match.group("field") or "").lower()
        if token in _SCHEMA_DISCRIMINATOR_TOKENS:
            tokens.add(token)
    return tokens


def _entry_unjustified_schema_discriminator_tokens(entry: dict[str, Any]) -> set[str]:
    declared_text = "\n".join(
        [
            str(entry.get("summary") or ""),
            str(entry.get("justification") or ""),
            *[str(item) for item in list(entry.get("test_descriptions") or [])],
            *[str(item) for item in list(entry.get("properties") or [])],
            *[str(item) for item in list(entry.get("metamorphic_relations") or [])],
        ]
    )
    return _schema_indicator_tokens_in_code(
        entry.get("content")
    ) - _schema_indicator_tokens_in_text(declared_text)


def extract_data_contract_field_paths(values: list[Any]) -> list[str]:
    paths: list[str] = []
    for value in values:
        text = str(value or "")
        if not text.strip():
            continue
        for match in _FIELD_PATH_TEXT_PATTERN.finditer(text):
            candidate = (match.group("backtick") or match.group("plain") or "").strip().lower()
            if not candidate:
                continue
            is_backtick_match = bool(match.group("backtick"))
            segments = candidate.split(".")
            if len(segments) < 2 or len(segments) > 4:
                continue
            if any(not segment or not segment[0].islower() for segment in segments):
                continue
            prefix_char = text[match.start() - 1] if match.start() > 0 else ""
            if prefix_char == "@":
                continue
            next_index = match.end()
            while next_index < len(text) and text[next_index].isspace():
                next_index += 1
            next_char = text[next_index] if next_index < len(text) else ""
            if next_char == "(":
                continue
            if next_char == "." and not is_backtick_match:
                after_dot_index = next_index + 1
                while after_dot_index < len(text) and text[after_dot_index].isspace():
                    after_dot_index += 1
                after_dot = text[after_dot_index] if after_dot_index < len(text) else ""
                if after_dot and (after_dot.isalpha() or after_dot in {"_", "$"}):
                    continue
            window_start = max(0, match.start() - 12)
            window_end = min(len(text), match.end() + 12)
            surrounding_context = (
                text[window_start : match.start()] + " " + text[match.end() : window_end]
            )
            if segments[0] in _FIELD_PATH_ROOT_ALIASES:
                segments[0] = _FIELD_PATH_ROOT_ALIASES[segments[0]]
                candidate = ".".join(segments)
            candidate_context = candidate.replace(".", " ")
            if segments[
                -1
            ] in _WEAK_FIELD_PATH_TERMINALS and not _WEAK_FIELD_PATH_CONTEXT_PATTERN.search(
                surrounding_context
            ):
                continue
            if not (
                _FIELD_PATH_CONTEXT_PATTERN.search(surrounding_context)
                or _FIELD_PATH_SELF_CONTEXT_PATTERN.search(candidate_context)
            ):
                continue
            paths.append(candidate)
    return _dedupe_strings(paths)


def _field_path_entry_text(entry: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(entry.get("summary") or ""),
            str(entry.get("justification") or ""),
            str(entry.get("content") or ""),
            *[str(item) for item in list(entry.get("test_descriptions") or [])],
            *[str(item) for item in list(entry.get("properties") or [])],
            *[str(item) for item in list(entry.get("metamorphic_relations") or [])],
        ]
    )


def _field_path_missing_container_case(text: str, *, root: str) -> bool:
    if re.search(
        rf"\b(?:missing|without|omit(?:s|ted)?)\s+`?{re.escape(root)}`?\b",
        text,
        re.IGNORECASE,
    ):
        return True
    return bool(re.search(r"[\[,]\s*\{\s*\}\s*[,}\]]", text))


def _field_path_missing_leaf_case(text: str, *, parent: str, leaf: str, path: str) -> bool:
    if re.search(
        rf"\b(?:missing|without|omit(?:s|ted)?)\s+`?(?:{re.escape(path)}|{re.escape(leaf)})`?\b",
        text,
        re.IGNORECASE,
    ):
        return True
    parent_pattern = re.compile(
        rf"(?:['\"`]{re.escape(parent)}['\"`]|\b{re.escape(parent)}\b)\s*:\s*\{{(?P<body>[^{{}}]*)\}}",
        re.IGNORECASE | re.DOTALL,
    )
    leaf_key_pattern = re.compile(
        rf"(?:['\"`]{re.escape(leaf)}['\"`]|\b{re.escape(leaf)}\b)\s*:",
        re.IGNORECASE,
    )
    for match in parent_pattern.finditer(text):
        body = match.group("body") or ""
        if not leaf_key_pattern.search(body):
            return True
    return False


def evaluate_field_path_negative_shape_coverage(
    entry: dict[str, Any],
    field_paths: list[str],
) -> dict[str, Any]:
    normalized_paths = _dedupe_strings(
        [str(path).strip().lower() for path in field_paths if str(path).strip()]
    )
    if not normalized_paths:
        return {
            "field_paths": [],
            "covered_shapes": [],
            "gaps": [],
            "coverage_ratio": 1.0,
        }

    text = _field_path_entry_text(entry)
    covered_shapes: set[str] = set()
    for path in normalized_paths:
        segments = path.split(".")
        if len(segments) < 2:
            continue
        root = segments[0]
        parent = segments[-2]
        leaf = segments[-1]
        if _field_path_missing_container_case(text, root=root):
            covered_shapes.add("missing_container_key")
        if _UNEXPECTED_SIBLING_KEY_HINT_PATTERN.search(text):
            covered_shapes.add("unexpected_sibling_key")
        if _field_path_missing_leaf_case(text, parent=parent, leaf=leaf, path=path):
            covered_shapes.add("missing_leaf_key")

    required_shapes = [
        "missing_container_key",
        "unexpected_sibling_key",
        "missing_leaf_key",
    ]
    gaps = [shape for shape in required_shapes if shape not in covered_shapes]
    coverage_ratio = (
        len(covered_shapes.intersection(required_shapes)) / len(required_shapes)
        if required_shapes
        else 1.0
    )
    return {
        "field_paths": normalized_paths,
        "covered_shapes": sorted(covered_shapes),
        "gaps": gaps,
        "coverage_ratio": round(coverage_ratio, 4),
    }


def default_generated_test_path(
    framework: str,
    *,
    language: str = "",
    strategy: str = "regression",
    index: int = 1,
) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", strategy.strip().lower()).strip("_") or "generated"
    normalized_framework = infer_test_inventory_framework(
        explicit_framework=framework,
        test_command=None,
    )
    normalized_language = (
        (language or default_test_inventory_language(normalized_framework)).strip().lower()
    )
    if normalized_framework in {"pytest", "unittest"}:
        return f"tests/test_apex_{token}_{index}.py"
    if normalized_framework == "jest":
        suffix = ".test.ts" if normalized_language in {"typescript", "ts"} else ".test.js"
        return f"__tests__/apex_{token}_{index}{suffix}"
    if normalized_framework == "vitest":
        suffix = ".test.ts" if normalized_language in {"typescript", "ts"} else ".test.js"
        return f"tests/apex_{token}_{index}{suffix}"
    if normalized_framework == "go_test":
        return f"tests/apex_{token}_{index}_test.go"
    if normalized_framework == "cargo_test":
        return f"tests/apex_{token}_{index}.rs"
    if normalized_framework == "rspec":
        return f"spec/apex_{token}_{index}_spec.rb"
    if normalized_framework == "phpunit":
        return f"tests/Apex{token.title().replace('_', '')}{index}Test.php"
    if normalized_framework == "junit":
        return f"tests/Apex{token.title().replace('_', '')}{index}Test.java"
    if normalized_framework == "dotnet_test":
        return f"tests/Apex{token.title().replace('_', '')}{index}Tests.cs"
    if normalized_framework == "ctest":
        return f"tests/apex_{token}_{index}.cmake"
    suffix = ".ts" if normalized_language == "typescript" else ".txt"
    return f"tests/apex_{token}_{index}{suffix}"


def normalize_test_portfolio_entry(
    raw: Any,
    *,
    index: int,
    default_framework: str = "",
    default_language: str = "",
    default_generator_vendor: str = "",
    default_generator_role: str = "generator",
    test_command: str = "",
) -> dict[str, Any]:
    payload = dict(raw or {}) if isinstance(raw, dict) else {"content": str(raw or "")}
    framework = infer_test_inventory_framework(
        explicit_framework=str(
            payload.get("framework") or payload.get("test_framework") or default_framework
        ),
        test_command=str(payload.get("test_command") or test_command or "").strip(),
    )
    language = str(
        payload.get("language") or payload.get("test_language") or default_language
    ).strip().lower() or default_test_inventory_language(framework)
    strategy = _normalize_strategy(payload.get("strategy") or payload.get("category"), raw=payload)
    raw_path_value = payload.get("path") or payload.get("test_path")
    raw_path_text = str(raw_path_value or "").strip()
    path = _normalize_relative_path(raw_path_value)
    unsafe_path_rejected = bool(raw_path_text and not path)
    if not path:
        path = default_generated_test_path(
            framework,
            language=language,
            strategy=strategy,
            index=index,
        )
    content = normalize_generated_test_content(
        payload.get("content") or payload.get("test_code") or ""
    )
    summary = str(payload.get("summary") or "").strip()
    descriptions = _dedupe_strings(
        list(payload.get("test_descriptions") or [])
        + list(payload.get("descriptions") or [])
        + list(payload.get("checks") or [])
    )
    focus_files = _dedupe_strings(
        list(payload.get("focus_files") or []) + list(payload.get("target_files") or [])
    )
    focus_tests = _dedupe_strings(
        list(payload.get("focus_tests") or []) + list(payload.get("target_tests") or [])
    )
    contract_sources = _normalize_contract_sources(
        list(payload.get("contract_sources") or [])
        + list(payload.get("evidence_sources") or [])
        + list(payload.get("independent_sources") or [])
    )
    contract_targets = _dedupe_strings(
        list(payload.get("contract_targets") or [])
        + ([payload.get("contract_target")] if payload.get("contract_target") else [])
        + (
            [payload.get("primary_contract_target")]
            if payload.get("primary_contract_target")
            else []
        )
    )
    contract_axes = _normalize_contract_axes(
        list(payload.get("contract_axes") or [])
        + list(payload.get("coverage_axes") or [])
        + ([payload.get("coverage_axis")] if payload.get("coverage_axis") else [])
    )
    justification = str(
        payload.get("justification")
        or payload.get("rationale")
        or payload.get("independent_justification")
        or summary
        or ""
    ).strip()
    materialization_mode = _normalize_materialization_mode(
        payload.get("materialization_mode") or payload.get("content_mode")
    )
    generator_role = (
        str(payload.get("generator_role") or default_generator_role or "generator").strip().lower()
    )
    generator_vendor = (
        str(payload.get("generator_vendor") or default_generator_vendor or "").strip().lower()
    )
    adjudicator_vendor = str(payload.get("adjudicator_vendor") or "").strip().lower()
    reference_targets = _dedupe_strings(
        list(payload.get("reference_targets") or [])
        + ([payload.get("reference_target")] if payload.get("reference_target") else [])
    )
    properties = _dedupe_strings(list(payload.get("properties") or []))
    metamorphic_relations = _dedupe_strings(list(payload.get("metamorphic_relations") or []))
    fuzz_seeds = _dedupe_strings(
        list(payload.get("fuzz_seeds") or [])
        + ([payload.get("fuzz_seed")] if payload.get("fuzz_seed") else [])
    )
    coverage_signal = _clamp(payload.get("coverage_signal") or payload.get("coverage_score"))
    mutation_signal = _clamp(payload.get("mutation_signal") or payload.get("mutation_score"))
    flake_signal = _clamp(payload.get("flake_signal"))
    patch_overfit_risk = _clamp(payload.get("patch_overfit_risk"))
    artifact_digest = hashlib.sha1(
        (f"{path}\n{strategy}\n{summary}\n" + content[:512]).encode("utf-8")
    ).hexdigest()[:12]
    artifact_id = str(payload.get("artifact_id") or f"test_{index}_{artifact_digest}").strip()
    validation = dict(payload.get("validation") or {})
    if unsafe_path_rejected:
        validation["unsafe_path_rejected"] = True
        validation["unsafe_path_original"] = raw_path_text
        validation.setdefault("materialization_safe", False)
    acceptance_requirements = _dedupe_strings(
        list(payload.get("acceptance_requirements") or [])
        + ([payload.get("acceptance_requirement")] if payload.get("acceptance_requirement") else [])
    )
    explicit_acceptance_requirements = bool(
        list(payload.get("acceptance_requirements") or [])
        or str(payload.get("acceptance_requirement") or "").strip()
    )
    objective = str(
        payload.get("objective")
        or payload.get("objective_title")
        or payload.get("objective_summary")
        or summary
        or ""
    ).strip()
    explicit_objective = any(
        str(payload.get(key) or "").strip()
        for key in ("objective", "objective_title", "objective_summary")
    )
    if not acceptance_requirements and objective:
        acceptance_requirements = [objective]
    interface_specification = _dedupe_strings(
        list(payload.get("interface_specification") or [])
        + list(payload.get("interface_targets") or [])
        + list(contract_targets)
    )
    explicit_interface_specification = bool(
        list(payload.get("interface_specification") or [])
        or list(payload.get("interface_targets") or [])
    )
    if not interface_specification and contract_targets:
        interface_specification = list(contract_targets)
    explicit_objective_id = bool(str(payload.get("objective_id") or "").strip())
    objective_id = str(payload.get("objective_id") or "").strip() or _default_artifact_objective_id(
        artifact_id,
        contract_targets,
        index=index,
    )
    explicit_milestone_id = bool(str(payload.get("milestone_id") or "").strip())
    milestone_id = str(payload.get("milestone_id") or "").strip() or _default_artifact_milestone_id(
        contract_axes
    )
    pass_then_invert = _normalize_pass_then_invert(payload.get("pass_then_invert"))
    raw_pass_then_invert = payload.get("pass_then_invert")
    raw_pass_then_invert_payload = (
        dict(raw_pass_then_invert) if isinstance(raw_pass_then_invert, dict) else {}
    )
    oracle_origin = str(payload.get("oracle_origin") or "").strip().lower()
    explicit_oracle_origin = bool(str(payload.get("oracle_origin") or "").strip())
    if not oracle_origin and pass_then_invert.get("attempted"):
        oracle_origin = "pass_then_invert"
    expected_fixed_behavior = str(payload.get("expected_fixed_behavior") or "").strip()
    if not expected_fixed_behavior:
        expected_fixed_behavior = str(
            payload.get("fixed_behavior")
            or payload.get("objective")
            or payload.get("summary")
            or ""
        ).strip()
    expected_broken_failure_mode = str(
        payload.get("expected_broken_failure_mode") or payload.get("broken_failure_mode") or ""
    ).strip()
    if not expected_broken_failure_mode:
        expected_broken_failure_mode = str(
            pass_then_invert.get("inversion_summary") or payload.get("justification") or ""
        ).strip()
    authoritative_source = str(
        payload.get("authoritative_source") or payload.get("contract_source") or ""
    ).strip()
    if not authoritative_source:
        authoritative_source = str(payload.get("oracle_origin") or "").strip() or "; ".join(
            contract_sources[:2]
        )
    public_surface = str(
        payload.get("public_surface")
        or payload.get("exact_public_surface")
        or payload.get("assertion_surface")
        or ""
    ).strip()
    if not public_surface:
        public_surface = str(contract_targets[0] if contract_targets else "").strip()
    objective_status = str(payload.get("objective_status") or "").strip().lower() or (
        "draft" if content.strip() else "planned"
    )
    return {
        "artifact_id": artifact_id,
        "path": path,
        "content": content,
        "framework": framework,
        "language": language,
        "strategy": strategy,
        "summary": summary,
        "test_descriptions": descriptions,
        "focus_files": focus_files,
        "focus_tests": focus_tests,
        "contract_sources": contract_sources,
        "contract_targets": contract_targets,
        "contract_axes": contract_axes,
        "justification": justification,
        "materialization_mode": materialization_mode,
        "generator_role": generator_role,
        "generator_vendor": generator_vendor,
        "adjudicator_vendor": adjudicator_vendor,
        "reference_targets": reference_targets,
        "properties": properties,
        "metamorphic_relations": metamorphic_relations,
        "fuzz_seeds": fuzz_seeds,
        "coverage_signal": coverage_signal,
        "mutation_signal": mutation_signal,
        "flake_signal": flake_signal,
        "patch_overfit_risk": patch_overfit_risk,
        "validation": validation,
        "milestone_id": milestone_id,
        "objective_id": objective_id,
        "objective": objective,
        "acceptance_requirements": acceptance_requirements,
        "interface_specification": interface_specification,
        "oracle_origin": oracle_origin,
        "expected_fixed_behavior": expected_fixed_behavior,
        "expected_broken_failure_mode": expected_broken_failure_mode,
        "authoritative_source": authoritative_source,
        "public_surface": public_surface,
        "pass_then_invert": pass_then_invert,
        "design_metadata_sources": {
            "milestone_id": "explicit" if explicit_milestone_id else "derived",
            "objective_id": "explicit" if explicit_objective_id else "derived",
            "objective": "explicit" if explicit_objective else "derived",
            "acceptance_requirements": (
                "explicit" if explicit_acceptance_requirements else "derived"
            ),
            "interface_specification": (
                "explicit" if explicit_interface_specification else "derived"
            ),
            "oracle_origin": "explicit" if explicit_oracle_origin else "derived",
            "expected_fixed_behavior": (
                "explicit"
                if str(payload.get("expected_fixed_behavior") or "").strip()
                else ("derived" if expected_fixed_behavior else "missing")
            ),
            "expected_broken_failure_mode": (
                "explicit"
                if str(payload.get("expected_broken_failure_mode") or "").strip()
                else ("derived" if expected_broken_failure_mode else "missing")
            ),
            "authoritative_source": (
                "explicit"
                if str(payload.get("authoritative_source") or "").strip()
                else ("derived" if authoritative_source else "missing")
            ),
            "public_surface": (
                "explicit"
                if str(payload.get("public_surface") or "").strip()
                else ("derived" if public_surface else "missing")
            ),
            "pass_then_invert": (
                "explicit"
                if raw_pass_then_invert is not None and bool(pass_then_invert)
                else "missing"
            ),
            "pass_then_invert.passing_variant_summary": (
                "explicit"
                if str(
                    raw_pass_then_invert_payload.get("passing_variant_summary")
                    or raw_pass_then_invert_payload.get("pass_summary")
                    or raw_pass_then_invert_payload.get("passing_summary")
                    or ""
                ).strip()
                else "missing"
            ),
            "pass_then_invert.inversion_summary": (
                "explicit"
                if str(
                    raw_pass_then_invert_payload.get("inversion_summary")
                    or raw_pass_then_invert_payload.get("invert_summary")
                    or raw_pass_then_invert_payload.get("failing_oracle_summary")
                    or ""
                ).strip()
                else "missing"
            ),
            "pass_then_invert.execution_feedback_summary": (
                "explicit"
                if str(
                    raw_pass_then_invert_payload.get("execution_feedback_summary")
                    or raw_pass_then_invert_payload.get("feedback_summary")
                    or ""
                ).strip()
                else "missing"
            ),
        },
        "dual_version_verified": bool(payload.get("dual_version_verified")),
        "objective_status": objective_status,
        "promotion_status": str(payload.get("promotion_status") or "pending").strip().lower(),
        "promotion_score": _clamp(payload.get("promotion_score")),
        "promotion_reasons": _dedupe_strings(list(payload.get("promotion_reasons") or [])),
    }


def normalize_test_suite_artifact_payload(
    payload: Any,
    *,
    default_framework: str = "",
    default_language: str = "",
    default_generator_vendor: str = "",
    test_command: str = "",
) -> dict[str, Any]:
    raw_payload = dict(payload or {}) if isinstance(payload, dict) else {}
    framework = infer_test_inventory_framework(
        explicit_framework=str(
            raw_payload.get("framework") or raw_payload.get("test_framework") or default_framework
        ),
        test_command=str(raw_payload.get("test_command") or test_command or "").strip(),
    )
    language = str(
        raw_payload.get("language") or raw_payload.get("test_language") or default_language
    ).strip().lower() or default_test_inventory_language(framework)
    raw_entries = list(raw_payload.get("test_artifacts") or [])
    if not raw_entries and (
        raw_payload.get("test_code")
        or raw_payload.get("test_descriptions")
        or raw_payload.get("summary")
    ):
        raw_entries = [
            {
                "content": raw_payload.get("test_code") or "",
                "path": raw_payload.get("test_path") or "",
                "summary": raw_payload.get("summary") or "",
                "test_descriptions": list(raw_payload.get("test_descriptions") or []),
                "strategy": raw_payload.get("strategy") or "regression",
                "framework": framework,
                "language": language,
                "contract_sources": list(raw_payload.get("contract_sources") or []),
            }
        ]
    entries = [
        normalize_test_portfolio_entry(
            entry,
            index=index,
            default_framework=framework,
            default_language=language,
            default_generator_vendor=default_generator_vendor,
            test_command=test_command,
        )
        for index, entry in enumerate(raw_entries, start=1)
    ]
    design_payload = normalize_test_generation_design_payload(
        raw_payload,
        issue_description=str(
            raw_payload.get("summary") or raw_payload.get("portfolio_summary") or ""
        ),
        issue_summary=str(raw_payload.get("summary") or "").strip(),
        success_criteria=list(raw_payload.get("test_descriptions") or []),
        behavioral_obligations=list(raw_payload.get("contract_hypotheses") or []),
        interface_targets=_dedupe_strings(
            list(raw_payload.get("reference_targets") or [])
            + [value for entry in entries for value in list(entry.get("contract_targets") or [])]
        ),
        required_axes=list(raw_payload.get("required_contract_axes") or []),
    )
    return {
        "summary": str(raw_payload.get("summary") or "").strip(),
        "framework": framework,
        "language": language,
        "test_code": normalize_generated_test_content(raw_payload.get("test_code") or ""),
        "test_descriptions": _dedupe_strings(list(raw_payload.get("test_descriptions") or [])),
        "test_artifacts": entries,
        "portfolio_summary": str(raw_payload.get("portfolio_summary") or "").strip(),
        "promotion_summary": str(raw_payload.get("promotion_summary") or "").strip(),
        "validation_summary": dict(raw_payload.get("validation_summary") or {}),
        "contract_hypotheses": _dedupe_strings(list(raw_payload.get("contract_hypotheses") or [])),
        "reference_targets": _dedupe_strings(list(raw_payload.get("reference_targets") or [])),
        "task_contract": dict(design_payload.get("task_contract") or {}),
        "milestones": [dict(item) for item in list(design_payload.get("milestones") or [])],
        "test_objectives": [
            dict(item) for item in list(design_payload.get("test_objectives") or [])
        ],
        "regression_suite_summary": dict(raw_payload.get("regression_suite_summary") or {}),
        "minimization_summary": dict(raw_payload.get("minimization_summary") or {}),
        "promoted_artifact_ids": _dedupe_strings(
            list(raw_payload.get("promoted_artifact_ids") or [])
        ),
        "candidate_artifact_ids": _dedupe_strings(
            list(raw_payload.get("candidate_artifact_ids") or [])
        ),
        "exploratory_artifact_ids": _dedupe_strings(
            list(raw_payload.get("exploratory_artifact_ids") or [])
        ),
        "public_signal": dict(raw_payload.get("public_signal") or {}),
    }


def select_cross_validation_test_artifacts(payload: Any) -> list[dict[str, Any]]:
    normalized = normalize_test_suite_artifact_payload(payload)
    issue_required_targets = _issue_required_targets_from_payload(normalized)

    def _cross_validation_eligible(item: dict[str, Any]) -> bool:
        validation = dict(item.get("validation") or {})
        if not validation:
            return True
        if "baseline_preserved" in validation and not bool(validation.get("baseline_preserved")):
            return False
        if "execution_targeted_supported" in validation:
            if bool(validation.get("execution_targeted_supported")):
                return bool(validation.get("execution_succeeded"))
            return bool(
                validation.get("collection_succeeded") and validation.get("artifact_discovered")
            )
        return True

    if issue_required_targets:
        preferred_promoted = [
            dict(item)
            for item in list(normalized.get("test_artifacts") or [])
            if str(item.get("promotion_status") or "").strip().lower() == "promoted"
            and _entry_issue_target_alignment(dict(item), issue_required_targets) > 0.0
            and not _entry_is_supplemental_issue_coverage(dict(item), issue_required_targets)
            and _cross_validation_eligible(dict(item))
        ]
        if preferred_promoted:
            preferred_promoted.sort(key=lambda item: -_clamp(item.get("promotion_score")))
            return preferred_promoted
        preferred_candidate = [
            dict(item)
            for item in list(normalized.get("test_artifacts") or [])
            if str(item.get("promotion_status") or "").strip().lower() == "candidate_public"
            and _entry_issue_target_alignment(dict(item), issue_required_targets) > 0.0
            and not _entry_is_supplemental_issue_coverage(dict(item), issue_required_targets)
            and _cross_validation_eligible(dict(item))
        ]
        if preferred_candidate:
            preferred_candidate.sort(key=lambda item: -_clamp(item.get("promotion_score")))
            return preferred_candidate
        return []
    promoted = [
        dict(item)
        for item in list(normalized.get("test_artifacts") or [])
        if str(item.get("promotion_status") or "").strip().lower() == "promoted"
        and _cross_validation_eligible(dict(item))
    ]
    if promoted:
        return promoted
    return [
        dict(item)
        for item in list(normalized.get("test_artifacts") or [])
        if str(item.get("promotion_status") or "").strip().lower() == "candidate_public"
        and _cross_validation_eligible(dict(item))
    ]


def portfolio_test_descriptions(
    payload: Any,
    *,
    promoted_first: bool = True,
) -> list[str]:
    normalized = normalize_test_suite_artifact_payload(payload)
    entries = list(normalized.get("test_artifacts") or [])
    if promoted_first:
        entries.sort(
            key=lambda item: (
                0 if str(item.get("promotion_status") or "").strip().lower() == "promoted" else 1,
                0
                if str(item.get("promotion_status") or "").strip().lower() == "candidate_public"
                else 1,
                -_clamp(item.get("promotion_score")),
            )
        )
    values = list(normalized.get("test_descriptions") or [])
    for entry in entries:
        values.extend(list(entry.get("test_descriptions") or []))
    return _dedupe_strings(values)


def portfolio_primary_test_code(payload: Any) -> str:
    normalized = normalize_test_suite_artifact_payload(payload)
    direct = str(normalized.get("test_code") or "").strip()
    if direct:
        return direct
    for entry in list(normalized.get("test_artifacts") or []):
        if str(entry.get("promotion_status") or "").strip().lower() != "promoted":
            continue
        content = str(entry.get("content") or "").strip()
        if content:
            return content
    return ""


def _entry_focus_fingerprint(
    entry: dict[str, Any],
) -> tuple[str, tuple[str, ...], tuple[str, ...], str]:
    return (
        str(entry.get("strategy") or "").strip().lower(),
        tuple(sorted(_dedupe_strings(list(entry.get("focus_files") or []))[:3])),
        tuple(sorted(_dedupe_strings(list(entry.get("focus_tests") or []))[:3])),
        re.sub(r"\s+", " ", str(entry.get("summary") or "").strip().lower())[:96],
    )


def _looks_like_source_file_reference(text: str) -> bool:
    head = str(text or "").strip().replace("\\", "/").split("::", 1)[0].strip()
    if not head:
        return False
    suffix = Path(head).suffix.lower()
    if suffix not in _KNOWN_SOURCE_FILE_EXTENSIONS:
        return False
    if "/" in head or re.match(r"^[A-Za-z]:/", head):
        return True
    return head.count(".") <= 2


def _clean_contract_target_segment(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "", str(value or "").strip())
    return cleaned.strip("_")


def _normalize_contract_target(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if "::test_" in lowered or "/tests/" in lowered or lowered.startswith("tests/"):
        return ""
    text = re.sub(r"\(.*\)$", "", text).strip()
    if "::" not in text and _looks_like_source_file_reference(text):
        return ""
    if "::" in text:
        _, _, symbol = text.partition("::")
        text = symbol.strip()
    if not text:
        return ""
    text = text.replace("#", ".")
    parts = [
        cleaned
        for part in re.split(r"[.\s]+", text)
        for cleaned in [_clean_contract_target_segment(part)]
        if cleaned
    ]
    if not parts:
        return ""
    if len(parts) >= 2:
        return ".".join(part.lower() for part in parts[-2:])
    return parts[-1].lower()


def _contract_target_leaf(value: Any) -> str:
    normalized = _normalize_contract_target(value)
    if not normalized:
        return ""
    return normalized.rsplit(".", 1)[-1]


def _entry_targets_cover_required_target(
    entry_targets: set[str],
    required_target: str,
) -> bool:
    normalized_required = _normalize_contract_target(required_target)
    if not normalized_required:
        return False
    if normalized_required in entry_targets:
        return True
    if "." in normalized_required:
        return False
    return any(target.endswith("." + normalized_required) for target in entry_targets)


def _contract_target_specificity(target: str) -> tuple[int, int]:
    segments = [part for part in str(target or "").split(".") if part]
    return (1 if len(segments) >= 2 else 0, len(segments))


def _prefer_more_specific_contract_targets(targets: list[str]) -> list[str]:
    ordered = list(dict.fromkeys(str(target).strip() for target in targets if str(target).strip()))
    if not ordered:
        return []

    target_set = set(ordered)
    filtered: list[str] = []
    for target in ordered:
        if any(other != target and other.endswith("." + target) for other in target_set):
            continue
        filtered.append(target)
    return filtered


def _normalized_required_contract_targets(values: Optional[list[str]]) -> list[str]:
    normalized_targets = [
        target
        for target in (_normalize_contract_target(value) for value in list(values or []))
        if target
    ]
    return _prefer_more_specific_contract_targets(normalized_targets)


def _issue_required_targets_from_payload(normalized: dict[str, Any]) -> list[str]:
    validation_summary = dict(normalized.get("validation_summary") or {})
    return _normalized_required_contract_targets(
        list(validation_summary.get("issue_contract_targets") or [])
        + list(normalized.get("required_contract_targets") or [])
    )


def _entry_explicit_contract_targets(entry: dict[str, Any]) -> set[str]:
    return {
        target
        for target in (
            _normalize_contract_target(value) for value in list(entry.get("contract_targets") or [])
        )
        if target
    }


def _entry_reference_contract_targets(entry: dict[str, Any]) -> set[str]:
    return {
        target
        for target in (
            _normalize_contract_target(value)
            for value in list(entry.get("reference_targets") or [])
        )
        if target
    }


def _entry_issue_target_alignment(
    entry: dict[str, Any],
    required_targets: list[str],
) -> float:
    normalized_required = _normalized_required_contract_targets(required_targets)
    if not normalized_required:
        return 0.0
    direct_targets = _entry_explicit_contract_targets(entry)
    if direct_targets:
        entry_targets = direct_targets
        scale = 1.0
    else:
        entry_targets = _entry_reference_contract_targets(entry)
        scale = 0.5
    if not entry_targets:
        return 0.0
    covered = sum(
        1
        for required_target in normalized_required
        if _entry_targets_cover_required_target(entry_targets, required_target)
    )
    if covered <= 0:
        return 0.0
    return round(scale * covered / len(normalized_required), 4)


def _entry_is_supplemental_issue_coverage(
    entry: dict[str, Any],
    required_targets: list[str],
) -> bool:
    normalized_required = set(_normalized_required_contract_targets(required_targets))
    if not normalized_required:
        return False
    direct_targets = _entry_explicit_contract_targets(entry)
    if direct_targets:
        matched_direct_targets = {
            target
            for target in direct_targets
            if any(
                _entry_targets_cover_required_target({target}, required_target)
                for required_target in normalized_required
            )
        }
        if not matched_direct_targets:
            return False
        return bool(direct_targets.difference(matched_direct_targets))

    entry_targets = _entry_reference_contract_targets(entry)
    matched_reference_targets = {
        target
        for target in entry_targets
        if any(
            _entry_targets_cover_required_target({target}, required_target)
            for required_target in normalized_required
        )
    }
    if not matched_reference_targets:
        return False
    if entry_targets.difference(matched_reference_targets):
        return True
    text = "\n".join(
        [
            str(entry.get("summary") or ""),
            str(entry.get("justification") or ""),
            *[str(item) for item in list(entry.get("contract_sources") or [])],
        ]
    ).lower()
    return any(
        token in text
        for token in (
            "alias",
            "compat",
            "parity",
            "legacy",
            "wrapper",
            "same behavior",
        )
    )


def _portfolio_primary_contract_targets(
    normalized: dict[str, Any],
    *,
    required_targets: Optional[list[str]] = None,
) -> list[str]:
    normalized_required_targets = _normalized_required_contract_targets(required_targets)
    if normalized_required_targets:
        return normalized_required_targets

    candidates: list[tuple[int, int, str]] = []
    position = 0
    entries = list(normalized.get("test_artifacts") or [])
    axis_scoped_entries = [
        entry
        for entry in entries
        if _normalize_contract_axes(list(entry.get("contract_axes") or []))
    ]
    candidate_entries = axis_scoped_entries or entries
    for entry in candidate_entries:
        for candidate in list(entry.get("contract_targets") or []):
            candidates.append((0, position, str(candidate)))
            position += 1
        for candidate in list(entry.get("reference_targets") or []):
            candidates.append((1, position, str(candidate)))
            position += 1
    if not axis_scoped_entries:
        for candidate in list(normalized.get("reference_targets") or []):
            candidates.append((2, position, str(candidate)))
            position += 1

    normalized_candidates: list[tuple[int, int, str]] = []
    for priority, order, candidate in candidates:
        target = _normalize_contract_target(candidate)
        if not target:
            continue
        normalized_candidates.append((priority, order, target))

    if not normalized_candidates:
        return []

    all_targets = {target for _, _, target in normalized_candidates}
    filtered_candidates = [
        (priority, order, target)
        for priority, order, target in normalized_candidates
        if not any(other != target and other.startswith(target + ".") for other in all_targets)
    ]
    filtered_candidates.sort(
        key=lambda item: (
            item[0],
            -_contract_target_specificity(item[2])[0],
            -_contract_target_specificity(item[2])[1],
            item[1],
        )
    )

    ordered: list[str] = []
    seen: set[str] = set()
    for _, _, target in filtered_candidates:
        if target in seen:
            continue
        ordered.append(target)
        seen.add(target)
        if len(ordered) >= 2:
            break
    return ordered


def _infer_required_contract_axes(normalized: dict[str, Any]) -> set[str]:
    values: list[str] = [
        str(normalized.get("summary") or ""),
        str(normalized.get("portfolio_summary") or ""),
        str(normalized.get("promotion_summary") or ""),
    ]
    values.extend(list(normalized.get("contract_hypotheses") or []))
    values.extend(list(normalized.get("test_descriptions") or []))
    for entry in list(normalized.get("test_artifacts") or []):
        values.append(str(entry.get("summary") or ""))
        values.extend(list(entry.get("test_descriptions") or []))
        values.extend(list(entry.get("properties") or []))
        values.extend(list(entry.get("metamorphic_relations") or []))
    return set(infer_required_contract_axes_from_texts(values))


def _entry_contract_targets(entry: dict[str, Any]) -> set[str]:
    direct_targets = _entry_explicit_contract_targets(entry)
    if direct_targets:
        return direct_targets
    return _entry_reference_contract_targets(entry)


def _entry_contract_axes(entry: dict[str, Any]) -> set[str]:
    explicit_axes: set[str] = set(_normalize_contract_axes(list(entry.get("contract_axes") or [])))
    axes: set[str] = set(explicit_axes)
    values: list[str] = [
        str(entry.get("summary") or ""),
        str(entry.get("justification") or ""),
        str(entry.get("content") or ""),
    ]
    values.extend(list(entry.get("test_descriptions") or []))
    values.extend(list(entry.get("properties") or []))
    values.extend(list(entry.get("metamorphic_relations") or []))
    text = "\n".join(value for value in values if value).strip()
    inferred_non_positive_axes: set[str] = set()
    for axis, pattern in _CONTRACT_AXIS_PATTERNS.items():
        if pattern.search(text):
            inferred_non_positive_axes.add(axis)
    axes.update(inferred_non_positive_axes)
    if (
        "positive_path" in explicit_axes
        or _POSITIVE_PATH_PATTERN.search(text)
        or (text and not explicit_axes and not inferred_non_positive_axes)
    ):
        axes.add("positive_path")
    return axes


def _public_artifact_status_rank(entry: dict[str, Any]) -> int:
    status = str(entry.get("promotion_status") or "").strip().lower()
    if status == "promoted":
        return 2
    if status == "candidate_public":
        return 1
    return 0


def _public_artifact_sort_key(
    entry: dict[str, Any],
    *,
    index: int,
) -> tuple[int, int, int, int, int, int, int, int, int, int, int]:
    validation = dict(entry.get("validation") or {})
    return (
        _public_artifact_status_rank(entry),
        1 if not bool(entry.get("supplemental_issue_coverage")) else 0,
        int(round(10000.0 * _clamp(entry.get("issue_target_alignment")))),
        1
        if bool(entry.get("dual_version_verified") or validation.get("dual_version_verified"))
        else 0,
        1 if _entry_mutation_discrimination_passed(entry) else 0,
        1 if bool(validation.get("execution_succeeded")) else 0,
        1 if bool(validation.get("rerun_consistent")) else 0,
        1 if bool(validation.get("baseline_preserved")) else 0,
        int(round(10000.0 * _clamp(entry.get("promotion_score")))),
        len(_entry_contract_axes(entry)),
        -index,
    )


def _artifact_focus_overlap(
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
) -> bool:
    candidate_focus_files = set(_dedupe_strings(list(candidate.get("focus_files") or [])))
    incumbent_focus_files = set(_dedupe_strings(list(incumbent.get("focus_files") or [])))
    if (
        candidate_focus_files
        and incumbent_focus_files
        and candidate_focus_files.intersection(incumbent_focus_files)
    ):
        return True

    candidate_focus_tests = set(_dedupe_strings(list(candidate.get("focus_tests") or [])))
    incumbent_focus_tests = set(_dedupe_strings(list(incumbent.get("focus_tests") or [])))
    if (
        candidate_focus_tests
        and incumbent_focus_tests
        and candidate_focus_tests.intersection(incumbent_focus_tests)
    ):
        return True

    if (
        not candidate_focus_files
        and not incumbent_focus_files
        and not candidate_focus_tests
        and not incumbent_focus_tests
    ):
        return _entry_focus_fingerprint(candidate) == _entry_focus_fingerprint(incumbent)
    return False


def _artifact_redundant_public_coverage(
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
) -> bool:
    candidate_objective_id = str(candidate.get("objective_id") or "").strip()
    incumbent_objective_id = str(incumbent.get("objective_id") or "").strip()
    if not candidate_objective_id or not incumbent_objective_id:
        return False
    if candidate_objective_id != incumbent_objective_id:
        return False

    if bool(candidate.get("supplemental_issue_coverage")) is False and bool(
        incumbent.get("supplemental_issue_coverage")
    ):
        return False
    if _clamp(candidate.get("issue_target_alignment")) > _clamp(
        incumbent.get("issue_target_alignment")
    ):
        return False

    candidate_targets = _entry_contract_targets(candidate)
    incumbent_targets = _entry_contract_targets(incumbent)
    if candidate_targets:
        if not incumbent_targets or not candidate_targets.issubset(incumbent_targets):
            return False
    elif incumbent_targets:
        return False

    candidate_axes = _entry_contract_axes(candidate)
    incumbent_axes = _entry_contract_axes(incumbent)
    if candidate_axes:
        if not incumbent_axes or not candidate_axes.issubset(incumbent_axes):
            return False
    elif incumbent_axes:
        return False

    if candidate_targets == incumbent_targets and candidate_axes == incumbent_axes:
        return True
    return _artifact_focus_overlap(candidate, incumbent)


def _minimize_public_test_artifacts(
    scored_entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    minimized_entries = [dict(entry) for entry in scored_entries]
    kept_by_objective: dict[str, list[dict[str, Any]]] = {}
    kept_public_artifact_ids: list[str] = []
    pruned_public_artifact_ids: list[str] = []
    pruned_public_artifact_paths: list[str] = []
    minimization_decisions: list[dict[str, Any]] = []

    ranked_entries = sorted(
        enumerate(minimized_entries),
        key=lambda item: _public_artifact_sort_key(item[1], index=item[0]),
        reverse=True,
    )
    for index, entry in ranked_entries:
        if _public_artifact_status_rank(entry) <= 0:
            continue
        objective_id = str(entry.get("objective_id") or "").strip()
        if not objective_id:
            artifact_id = str(entry.get("artifact_id") or "").strip()
            if artifact_id:
                kept_public_artifact_ids.append(artifact_id)
            continue

        redundant_against = next(
            (
                kept
                for kept in kept_by_objective.get(objective_id, [])
                if _artifact_redundant_public_coverage(entry, kept)
            ),
            None,
        )
        if redundant_against is None:
            kept_by_objective.setdefault(objective_id, []).append(dict(entry))
            artifact_id = str(entry.get("artifact_id") or "").strip()
            if artifact_id:
                kept_public_artifact_ids.append(artifact_id)
            continue

        updated_entry = dict(minimized_entries[index])
        original_status = str(updated_entry.get("promotion_status") or "").strip().lower()
        updated_entry["pre_minimization_promotion_status"] = original_status
        updated_entry["promotion_status"] = "exploratory"
        updated_entry["promotion_reasons"] = _dedupe_strings(
            list(updated_entry.get("promotion_reasons") or [])
            + ["suite_minimization_redundant_public_coverage"]
        )
        updated_entry["minimization"] = {
            **dict(updated_entry.get("minimization") or {}),
            "demoted": True,
            "reason": "redundant_public_coverage",
            "demoted_by_artifact_id": str(redundant_against.get("artifact_id") or "").strip(),
            "demoted_by_path": str(redundant_against.get("path") or "").strip(),
            "objective_id": objective_id,
        }
        minimized_entries[index] = updated_entry

        artifact_id = str(updated_entry.get("artifact_id") or "").strip()
        artifact_path = str(updated_entry.get("path") or "").strip()
        if artifact_id:
            pruned_public_artifact_ids.append(artifact_id)
        if artifact_path:
            pruned_public_artifact_paths.append(artifact_path)
        minimization_decisions.append(
            {
                "artifact_id": artifact_id,
                "path": artifact_path,
                "objective_id": objective_id,
                "demoted_by_artifact_id": str(redundant_against.get("artifact_id") or "").strip(),
                "demoted_by_path": str(redundant_against.get("path") or "").strip(),
                "reason": "redundant_public_coverage",
            }
        )

    objective_ids = {
        str(entry.get("objective_id") or "").strip()
        for entry in minimized_entries
        if str(entry.get("objective_id") or "").strip()
    }
    kept_public_artifact_ids = _dedupe_strings(kept_public_artifact_ids)
    pruned_public_artifact_ids = _dedupe_strings(pruned_public_artifact_ids)
    pruned_public_artifact_paths = _dedupe_strings(pruned_public_artifact_paths)
    kept_public_artifact_count = len(kept_public_artifact_ids)
    objective_count = len(objective_ids)
    return (
        minimized_entries,
        {
            "kept_artifact_count": kept_public_artifact_count,
            "objective_count": objective_count,
            "redundant_artifact_count": len(pruned_public_artifact_ids),
            "objective_artifact_ratio": round(
                kept_public_artifact_count / max(objective_count, 1),
                4,
            ),
            "public_pruned_artifact_count": len(pruned_public_artifact_ids),
            "pruned_public_artifact_ids": pruned_public_artifact_ids,
            "pruned_public_artifact_paths": pruned_public_artifact_paths,
            "kept_public_artifact_ids": kept_public_artifact_ids,
            "decisions": minimization_decisions,
        },
    )


def _evaluate_contract_matrix(
    normalized: dict[str, Any],
    *,
    required_targets: Optional[list[str]] = None,
    required_axes: Optional[list[str]] = None,
) -> dict[str, Any]:
    normalized_required_targets = _normalized_required_contract_targets(required_targets)
    primary_targets = _portfolio_primary_contract_targets(
        normalized,
        required_targets=normalized_required_targets,
    )
    normalized_required_axes = _normalize_required_contract_axes(required_axes)
    required_axes = normalized_required_axes or sorted(_infer_required_contract_axes(normalized))
    if not primary_targets:
        return {
            "primary_targets": [],
            "required_axes": required_axes,
            "covered_axes_by_target": {},
            "coverage_ratio_by_target": {},
            "gaps_by_target": {},
            "gap_count": 0,
            "fully_covered_targets": [],
        }

    covered_axes_by_target: dict[str, list[str]] = {}
    coverage_ratio_by_target: dict[str, float] = {}
    gaps_by_target: dict[str, list[str]] = {}
    gap_count = 0
    fully_covered_targets: list[str] = []
    entries = list(normalized.get("test_artifacts") or [])
    for target in primary_targets:
        covered_axes: set[str] = set()
        for entry in entries:
            entry_targets = _entry_explicit_contract_targets(entry)
            if not entry_targets and not normalized_required_targets:
                entry_targets = _entry_reference_contract_targets(entry)
            if not _entry_targets_cover_required_target(entry_targets, target):
                continue
            if (
                normalized_required_targets
                and target in set(normalized_required_targets)
                and _entry_is_supplemental_issue_coverage(entry, normalized_required_targets)
            ):
                continue
            covered_axes.update(_entry_contract_axes(entry))
        covered_axes_by_target[target] = sorted(covered_axes)
        gaps = [axis for axis in required_axes if axis not in covered_axes]
        coverage_ratio_by_target[target] = round(
            (len(covered_axes.intersection(required_axes)) / len(required_axes))
            if required_axes
            else 1.0,
            4,
        )
        if gaps:
            gaps_by_target[target] = gaps
            gap_count += len(gaps)
        else:
            fully_covered_targets.append(target)
    return {
        "primary_targets": primary_targets,
        "required_axes": required_axes,
        "covered_axes_by_target": covered_axes_by_target,
        "coverage_ratio_by_target": coverage_ratio_by_target,
        "gaps_by_target": gaps_by_target,
        "gap_count": gap_count,
        "fully_covered_targets": fully_covered_targets,
    }


def _rollup_test_generation_design(
    normalized: dict[str, Any],
    *,
    scored_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    task_contract = _normalize_task_contract(normalized.get("task_contract"))
    seed_objectives = [
        _normalize_test_generation_objective(
            item,
            index=index,
            task_contract=task_contract,
        )
        for index, item in enumerate(list(normalized.get("test_objectives") or []), start=1)
    ]
    seed_milestones = [
        _normalize_test_generation_milestone(item, index=index)
        for index, item in enumerate(list(normalized.get("milestones") or []), start=1)
    ]

    objective_map = {
        str(item.get("objective_id") or "").strip(): dict(item)
        for item in seed_objectives
        if str(item.get("objective_id") or "").strip()
    }
    milestone_map = {
        str(item.get("milestone_id") or "").strip(): dict(item)
        for item in seed_milestones
        if str(item.get("milestone_id") or "").strip()
    }

    updated_entries: list[dict[str, Any]] = []
    objective_entries: dict[str, list[dict[str, Any]]] = {}
    artifact_metadata_gaps: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(scored_entries, start=1):
        entry = dict(raw_entry)
        artifact_id = str(entry.get("artifact_id") or f"artifact_{index}").strip()
        contract_targets = _dedupe_strings(list(entry.get("contract_targets") or []))
        objective_id = str(
            entry.get("objective_id") or ""
        ).strip() or _default_artifact_objective_id(
            artifact_id,
            contract_targets,
            index=index,
        )
        milestone_id = str(
            entry.get("milestone_id") or ""
        ).strip() or _default_artifact_milestone_id(list(entry.get("contract_axes") or []))
        objective = str(entry.get("objective") or entry.get("summary") or "").strip()
        acceptance_requirements = _dedupe_strings(
            list(entry.get("acceptance_requirements") or []) + ([objective] if objective else [])
        )
        interface_specification = _dedupe_strings(
            list(entry.get("interface_specification") or [])
            + contract_targets
            + list(task_contract.get("interface_specification") or [])
        )
        pass_then_invert = _normalize_pass_then_invert(entry.get("pass_then_invert"))
        oracle_origin = str(entry.get("oracle_origin") or "").strip().lower()
        if not oracle_origin and pass_then_invert.get("attempted"):
            oracle_origin = "pass_then_invert"
        dual_version_verified = _entry_dual_version_verified(entry)
        mutation_discrimination_passed = _entry_mutation_discrimination_passed(entry)
        objective_status = _entry_effective_objective_status(entry)
        entry.update(
            {
                "objective_id": objective_id,
                "milestone_id": milestone_id,
                "objective": objective
                or (acceptance_requirements[0] if acceptance_requirements else ""),
                "acceptance_requirements": acceptance_requirements,
                "interface_specification": interface_specification,
                "pass_then_invert": pass_then_invert,
                "oracle_origin": oracle_origin,
                "dual_version_verified": dual_version_verified,
                "mutation_discrimination_passed": mutation_discrimination_passed,
                "objective_status": objective_status,
            }
        )
        metadata_gap_fields = _artifact_design_metadata_gaps(entry)
        entry["design_metadata_complete"] = not metadata_gap_fields
        entry["design_metadata_gap_fields"] = list(metadata_gap_fields)
        if metadata_gap_fields:
            artifact_metadata_gaps.append(
                {
                    "path": str(entry.get("path") or "").strip(),
                    "missing_fields": list(metadata_gap_fields),
                }
            )
        updated_entries.append(entry)
        objective_entries.setdefault(objective_id, []).append(entry)

    for objective_id, entries in objective_entries.items():
        existing = dict(objective_map.get(objective_id) or {})
        first_entry = entries[0]
        objective_map[objective_id] = {
            "objective_id": objective_id,
            "milestone_id": str(
                existing.get("milestone_id") or first_entry.get("milestone_id") or ""
            ).strip()
            or "milestone_direct_regression",
            "objective": str(
                existing.get("objective")
                or first_entry.get("objective")
                or first_entry.get("summary")
                or ""
            ).strip(),
            "acceptance_requirements": _dedupe_strings(
                list(existing.get("acceptance_requirements") or [])
                + [
                    requirement
                    for entry in entries
                    for requirement in list(entry.get("acceptance_requirements") or [])
                ]
            ),
            "interface_specification": _dedupe_strings(
                list(existing.get("interface_specification") or [])
                + [
                    value
                    for entry in entries
                    for value in list(entry.get("interface_specification") or [])
                ]
                + list(task_contract.get("interface_specification") or [])
            ),
            "contract_targets": _dedupe_strings(
                list(existing.get("contract_targets") or [])
                + [
                    value
                    for entry in entries
                    for value in list(entry.get("contract_targets") or [])
                ]
            ),
            "contract_axes": _normalize_contract_axes(
                list(existing.get("contract_axes") or [])
                + [value for entry in entries for value in list(entry.get("contract_axes") or [])]
            ),
            "artifact_ids": _dedupe_strings(
                list(existing.get("artifact_ids") or [])
                + [str(entry.get("artifact_id") or "").strip() for entry in entries]
            ),
            "objective_status": _aggregate_objective_status(
                [str(entry.get("objective_status") or "").strip() for entry in entries]
            ),
            "dual_version_verified": any(
                bool(entry.get("dual_version_verified")) for entry in entries
            ),
            "mutation_discrimination_passed": all(
                _entry_mutation_discrimination_passed(entry) for entry in entries
            ),
            "pass_then_invert_complete": any(
                _pass_then_invert_complete(entry.get("pass_then_invert")) for entry in entries
            ),
            "baseline_preserved": any(
                bool(dict(entry.get("validation") or {}).get("baseline_preserved"))
                for entry in entries
            ),
            "promoted_artifact_ids": _dedupe_strings(
                [
                    str(entry.get("artifact_id") or "").strip()
                    for entry in entries
                    if str(entry.get("promotion_status") or "").strip().lower() == "promoted"
                ]
            ),
            "candidate_artifact_ids": _dedupe_strings(
                [
                    str(entry.get("artifact_id") or "").strip()
                    for entry in entries
                    if str(entry.get("promotion_status") or "").strip().lower()
                    in {"promoted", "candidate_public"}
                ]
            ),
            "design_metadata_complete": all(
                bool(entry.get("design_metadata_complete")) for entry in entries
            ),
            "design_metadata_complete_count": sum(
                1 for entry in entries if bool(entry.get("design_metadata_complete"))
            ),
        }

    for objective in objective_map.values():
        milestone_id = (
            str(objective.get("milestone_id") or "").strip() or "milestone_direct_regression"
        )
        milestone = dict(milestone_map.get(milestone_id) or {})
        objective_ids = _dedupe_strings(
            list(milestone.get("objective_ids") or [])
            + [str(objective.get("objective_id") or "").strip()]
        )
        milestone_map[milestone_id] = {
            "milestone_id": milestone_id,
            "title": str(milestone.get("title") or "").strip()
            or _DESIGN_MILESTONE_TITLES.get(milestone_id)
            or milestone_id.replace("_", " ").title(),
            "summary": str(milestone.get("summary") or "").strip()
            or _DESIGN_MILESTONE_TITLES.get(
                milestone_id,
                milestone_id.replace("_", " ").title(),
            ),
            "acceptance_requirements": _dedupe_strings(
                list(milestone.get("acceptance_requirements") or [])
                + list(objective.get("acceptance_requirements") or [])
            ),
            "objective_ids": objective_ids,
            "validation_level": str(milestone.get("validation_level") or "strict").strip().lower(),
            "pipeline_stages": _dedupe_strings(
                list(milestone.get("pipeline_stages") or [])
                or [
                    "context_hypothesis",
                    "pass_then_invert",
                    "execution_feedback",
                    "mutation_discrimination",
                    "dual_version_verification",
                ]
            ),
        }

    ordered_objectives = sorted(
        objective_map.values(),
        key=lambda item: (
            str(item.get("milestone_id") or ""),
            str(item.get("objective_id") or ""),
        ),
    )
    ordered_milestones: list[dict[str, Any]] = []
    for milestone_id, milestone in sorted(milestone_map.items()):
        related_objectives = [
            objective
            for objective in ordered_objectives
            if str(objective.get("milestone_id") or "") == milestone_id
        ]
        milestone_status = _aggregate_objective_status(
            [
                str(objective.get("objective_status") or "").strip()
                for objective in related_objectives
            ]
        )
        milestone["objective_status"] = milestone_status
        milestone["strict_validation_ready"] = bool(related_objectives) and all(
            bool(objective.get("dual_version_verified"))
            and bool(objective.get("mutation_discrimination_passed"))
            and bool(objective.get("baseline_preserved"))
            and bool(objective.get("pass_then_invert_complete"))
            and bool(objective.get("design_metadata_complete"))
            for objective in related_objectives
        )
        ordered_milestones.append(milestone)

    promoted_entries = [
        entry
        for entry in updated_entries
        if str(entry.get("promotion_status") or "").strip().lower() == "promoted"
    ]
    public_entries = [
        entry
        for entry in updated_entries
        if str(entry.get("promotion_status") or "").strip().lower()
        in {"promoted", "candidate_public"}
    ]
    dual_version_verified_count = sum(
        1 for entry in updated_entries if bool(entry.get("dual_version_verified"))
    )
    pass_then_invert_count = sum(
        1 for entry in updated_entries if _pass_then_invert_complete(entry.get("pass_then_invert"))
    )
    regression_suite_summary = {
        **dict(normalized.get("regression_suite_summary") or {}),
        "artifact_paths": _dedupe_strings(
            [str(entry.get("path") or "").strip() for entry in public_entries]
        ),
        "promoted_artifact_ids": _dedupe_strings(
            [str(entry.get("artifact_id") or "").strip() for entry in promoted_entries]
        ),
        "candidate_artifact_ids": _dedupe_strings(
            [str(entry.get("artifact_id") or "").strip() for entry in public_entries]
        ),
        "objective_ids": _dedupe_strings(
            [str(objective.get("objective_id") or "").strip() for objective in ordered_objectives]
        ),
        "baseline_preserving_artifact_count": sum(
            1
            for entry in updated_entries
            if bool(dict(entry.get("validation") or {}).get("baseline_preserved"))
        ),
        "dual_version_verified_artifact_count": dual_version_verified_count,
        "strict_ready": bool(ordered_milestones)
        and all(bool(milestone.get("strict_validation_ready")) for milestone in ordered_milestones),
        "design_artifact_metadata_gap_count": len(artifact_metadata_gaps),
        "design_artifact_metadata_gaps": artifact_metadata_gaps,
    }
    existing_minimization_summary = dict(normalized.get("minimization_summary") or {})
    pruned_public_artifact_ids = _dedupe_strings(
        list(existing_minimization_summary.get("pruned_public_artifact_ids") or [])
    )
    pruned_public_artifact_paths = _dedupe_strings(
        list(existing_minimization_summary.get("pruned_public_artifact_paths") or [])
    )
    kept_public_artifact_ids = _dedupe_strings(
        list(existing_minimization_summary.get("kept_public_artifact_ids") or [])
    )
    kept_artifact_count = int(
        existing_minimization_summary.get("kept_artifact_count") or len(public_entries)
    )
    objective_count = int(
        existing_minimization_summary.get("objective_count") or len(ordered_objectives)
    )
    minimization_summary = {
        **existing_minimization_summary,
        "kept_artifact_count": kept_artifact_count,
        "objective_count": objective_count,
        "redundant_artifact_count": int(
            existing_minimization_summary.get("redundant_artifact_count")
            or len(pruned_public_artifact_ids)
            or max(0, kept_artifact_count - len(ordered_objectives))
        ),
        "objective_artifact_ratio": round(
            float(
                existing_minimization_summary.get("objective_artifact_ratio")
                or (
                    kept_artifact_count / max(objective_count, 1)
                    if objective_count > 0
                    else float(kept_artifact_count)
                )
            ),
            4,
        ),
        "total_artifact_count": len(updated_entries),
        "public_artifact_count": len(public_entries),
        "pruned_artifact_count": len(
            list(
                dict(normalized.get("issue_surface_cleanup") or {}).get("pruned_artifact_paths")
                or []
            )
        ),
        "public_pruned_artifact_count": int(
            existing_minimization_summary.get("public_pruned_artifact_count")
            or len(pruned_public_artifact_ids)
        ),
        "pruned_public_artifact_ids": pruned_public_artifact_ids,
        "pruned_public_artifact_paths": pruned_public_artifact_paths,
        "kept_public_artifact_ids": kept_public_artifact_ids,
    }
    return {
        "test_artifacts": updated_entries,
        "test_objectives": ordered_objectives,
        "milestones": ordered_milestones,
        "regression_suite_summary": regression_suite_summary,
        "minimization_summary": minimization_summary,
        "dual_version_verified_artifact_count": dual_version_verified_count,
        "pass_then_invert_artifact_count": pass_then_invert_count,
        "design_artifact_metadata_gap_count": len(artifact_metadata_gaps),
        "design_artifact_metadata_gaps": artifact_metadata_gaps,
    }


def apply_test_portfolio_promotion(
    payload: Any,
    *,
    relevant_files: Optional[list[str]] = None,
    focus_test_files: Optional[list[str]] = None,
    failing_test_ids: Optional[list[str]] = None,
    localization_files: Optional[list[str]] = None,
    localization_symbols: Optional[list[str]] = None,
    changed_files: Optional[list[str]] = None,
    required_contract_targets: Optional[list[str]] = None,
    required_contract_axes: Optional[list[str]] = None,
    promoted_threshold: float = 0.74,
    candidate_threshold: float = 0.58,
) -> dict[str, Any]:
    normalized = normalize_test_suite_artifact_payload(payload)
    existing_validation_summary = dict(normalized.get("validation_summary") or {})
    normalized_required_targets = _normalized_required_contract_targets(required_contract_targets)
    normalized_required_axes = _normalize_required_contract_axes(
        list(existing_validation_summary.get("required_contract_axes") or [])
        + list(required_contract_axes or [])
    )
    primary_targets = _portfolio_primary_contract_targets(
        normalized,
        required_targets=normalized_required_targets,
    )
    required_axes = normalized_required_axes or sorted(_infer_required_contract_axes(normalized))
    portfolio_field_paths = extract_data_contract_field_paths(
        [
            str(normalized.get("summary") or ""),
            str(normalized.get("portfolio_summary") or ""),
            str(normalized.get("promotion_summary") or ""),
            *[str(item) for item in list(normalized.get("contract_hypotheses") or [])],
            *[str(item) for item in list(normalized.get("test_descriptions") or [])],
            *[
                str(item)
                for entry in list(normalized.get("test_artifacts") or [])
                for item in (
                    str(entry.get("summary") or ""),
                    str(entry.get("justification") or ""),
                    *[str(value) for value in list(entry.get("test_descriptions") or [])],
                    *[str(value) for value in list(entry.get("properties") or [])],
                )
            ],
        ]
    )
    relevant = set(_dedupe_strings(list(relevant_files or []) + list(localization_files or [])))
    focus_tests = set(_dedupe_strings(list(focus_test_files or []) + list(failing_test_ids or [])))
    changed = set(_dedupe_strings(list(changed_files or [])))
    symbols = {
        str(symbol).strip() for symbol in list(localization_symbols or []) if str(symbol).strip()
    }

    seen_fingerprints: dict[tuple[str, tuple[str, ...], tuple[str, ...], str], int] = {}
    promoted_ids: list[str] = []
    exploratory_ids: list[str] = []
    candidate_ids: list[str] = []
    vendor_families: set[str] = set()
    scored_entries: list[dict[str, Any]] = []

    for entry in list(normalized.get("test_artifacts") or []):
        scored = dict(entry)
        validation = dict(scored.get("validation") or {})
        focus_files = set(_dedupe_strings(list(scored.get("focus_files") or [])))
        focus_entry_tests = set(_dedupe_strings(list(scored.get("focus_tests") or [])))
        descriptions = list(scored.get("test_descriptions") or [])
        recognized_sources = set(
            _normalize_contract_sources(list(scored.get("contract_sources") or []))
        )
        direct_contract_targets = _entry_explicit_contract_targets(scored)
        supporting_contract_targets = _entry_contract_targets(scored)
        direct_contract_axes = _entry_contract_axes(scored)
        issue_target_alignment = _entry_issue_target_alignment(
            scored,
            normalized_required_targets,
        )
        supplemental_issue_coverage = _entry_is_supplemental_issue_coverage(
            scored,
            normalized_required_targets,
        )
        unjustified_schema_tokens = (
            _entry_unjustified_schema_discriminator_tokens(scored)
            if direct_contract_targets
            and issue_target_alignment > 0.0
            and not supplemental_issue_coverage
            else set()
        )
        schema_constraint_drift_risk = (
            min(1.0, 0.45 * len(unjustified_schema_tokens)) if unjustified_schema_tokens else 0.0
        )
        field_path_shape_signal = (
            evaluate_field_path_negative_shape_coverage(scored, portfolio_field_paths)
            if (
                portfolio_field_paths
                and direct_contract_targets
                and issue_target_alignment > 0.0
                and not supplemental_issue_coverage
                and "negative_malformed" in direct_contract_axes
            )
            else {
                "field_paths": [],
                "covered_shapes": [],
                "gaps": [],
                "coverage_ratio": 1.0,
            }
        )
        field_path_negative_shape_gap_count = len(list(field_path_shape_signal.get("gaps") or []))
        field_path_negative_shape_penalty = 0.05 * field_path_negative_shape_gap_count
        vendor = str(scored.get("generator_vendor") or "").strip().lower()
        if vendor:
            vendor_families.add(vendor)
        adjudicator_vendor = str(scored.get("adjudicator_vendor") or "").strip().lower()
        if adjudicator_vendor:
            vendor_families.add(adjudicator_vendor)

        independence = 0.0
        if str(scored.get("justification") or "").strip():
            independence += 0.18
        independence += min(0.36, 0.10 * len(recognized_sources))
        if relevant and focus_files.intersection(relevant):
            independence += 0.16
        if focus_tests and focus_entry_tests.intersection(focus_tests):
            independence += 0.16
        if symbols and any(
            symbol in " ".join(descriptions + [str(scored.get("summary") or "")])
            for symbol in symbols
        ):
            independence += 0.08
        if str(scored.get("strategy") or "") in {
            "property",
            "metamorphic",
            "differential",
            "fuzz_seed",
        }:
            independence += 0.08
        if issue_target_alignment > 0.0:
            independence += min(0.10, 0.10 * issue_target_alignment)
        contract_target_focus = (
            1.0
            if primary_targets
            and any(
                _entry_targets_cover_required_target(direct_contract_targets, target)
                for target in primary_targets
            )
            else (
                1.0
                if direct_contract_targets and not primary_targets
                else (
                    0.35
                    if primary_targets
                    and not direct_contract_targets
                    and any(
                        _entry_targets_cover_required_target(
                            supporting_contract_targets,
                            target,
                        )
                        for target in primary_targets
                    )
                    else 0.0
                )
            )
        )
        contract_axis_coverage = (
            len(direct_contract_axes.intersection(required_axes)) / len(required_axes)
            if (
                contract_target_focus > 0.0
                and required_axes
                and not supplemental_issue_coverage
                and (
                    not normalized_required_targets
                    or any(
                        _entry_targets_cover_required_target(
                            direct_contract_targets,
                            target,
                        )
                        for target in normalized_required_targets
                    )
                )
            )
            else 0.0
        )
        if contract_target_focus > 0.0:
            independence += 0.08
        independence = _clamp(independence)

        usefulness = 0.0
        if str(scored.get("content") or "").strip():
            usefulness += 0.18
        if descriptions:
            usefulness += 0.14
        if focus_files or focus_entry_tests:
            usefulness += 0.16
        if recognized_sources:
            usefulness += 0.12
        if str(scored.get("strategy") or "") in _KNOWN_TEST_STRATEGIES:
            usefulness += 0.14
        if validation.get("artifact_discovered"):
            usefulness += 0.12
        if validation.get("execution_succeeded"):
            usefulness += 0.14
        if contract_target_focus > 0.0:
            usefulness += 0.04
        if contract_axis_coverage > 0.0:
            usefulness += min(0.08, 0.08 * contract_axis_coverage)
        if issue_target_alignment > 0.0:
            usefulness += min(0.04, 0.04 * issue_target_alignment)
        usefulness = _clamp(usefulness)

        stability_components: list[float] = []
        if "baseline_preserved" in validation:
            stability_components.append(1.0 if validation.get("baseline_preserved") else 0.0)
        if "collection_succeeded" in validation:
            stability_components.append(1.0 if validation.get("collection_succeeded") else 0.0)
        if "execution_succeeded" in validation:
            stability_components.append(1.0 if validation.get("execution_succeeded") else 0.0)
        if "rerun_consistent" in validation:
            stability_components.append(1.0 if validation.get("rerun_consistent") else 0.0)
        if not stability_components and scored.get("flake_signal") is not None:
            stability_components.append(_clamp(scored.get("flake_signal")))
        stability = (
            sum(stability_components) / len(stability_components)
            if stability_components
            else (0.20 if str(scored.get("content") or "").strip() else 0.0)
        )
        stability = _clamp(stability)

        coverage_signal_measured = bool(
            validation.get("coverage_signal_measured", scored.get("coverage_signal_measured"))
        )
        mutation_signal_measured = bool(
            validation.get("mutation_signal_measured", scored.get("mutation_signal_measured"))
        )
        mutation_discrimination_passed = _entry_mutation_discrimination_passed(scored)
        coverage_bonus = (
            _clamp(validation.get("coverage_signal", scored.get("coverage_signal")))
            if coverage_signal_measured
            else 0.0
        )
        mutation_bonus = (
            _clamp(validation.get("mutation_signal", scored.get("mutation_signal")))
            if mutation_signal_measured
            else 0.0
        )

        redundancy_penalty = 0.0
        fingerprint = _entry_focus_fingerprint(scored)
        if fingerprint in seen_fingerprints:
            redundancy_penalty = min(0.28, 0.14 * seen_fingerprints[fingerprint])
        seen_fingerprints[fingerprint] = seen_fingerprints.get(fingerprint, 0) + 1

        patch_overfit_risk = _clamp(scored.get("patch_overfit_risk"))
        if (
            patch_overfit_risk < 0.10
            and changed
            and focus_files
            and focus_files.issubset(changed)
            and not recognized_sources.intersection(
                {"docs", "examples", "types", "existing_tests", "reproduction", "traceback"}
            )
            and not focus_entry_tests.intersection(focus_tests)
        ):
            patch_overfit_risk = 0.35
        patch_overfit_penalty = 0.22 * patch_overfit_risk
        supplemental_issue_penalty = 0.08 if supplemental_issue_coverage else 0.0
        schema_constraint_penalty = 0.12 * schema_constraint_drift_risk
        targeted_execution_failed = bool(validation.get("execution_targeted_supported")) and (
            "execution_succeeded" in validation and not bool(validation.get("execution_succeeded"))
        )
        execution_failure_penalty = 0.22 if targeted_execution_failed else 0.0
        mutation_survivors_present = int(validation.get("plausible_mutant_survived_count") or 0) > 0
        mutation_survivor_penalty = 0.24 if mutation_survivors_present else 0.0

        score = (
            (0.34 * stability)
            + (0.28 * usefulness)
            + (0.28 * independence)
            + (0.05 * coverage_bonus)
            + (0.05 * mutation_bonus)
            + (0.04 * contract_target_focus)
            + (0.04 * contract_axis_coverage)
            - redundancy_penalty
            - patch_overfit_penalty
            - supplemental_issue_penalty
            - schema_constraint_penalty
            - field_path_negative_shape_penalty
            - execution_failure_penalty
            - mutation_survivor_penalty
        )
        score = _clamp(score)

        execution_targeted_supported = bool(validation.get("execution_targeted_supported"))
        baseline_preserved = bool(validation.get("baseline_preserved"))
        collection_succeeded = bool(validation.get("collection_succeeded"))
        execution_succeeded = bool(validation.get("execution_succeeded"))
        rerun_consistent = bool(validation.get("rerun_consistent"))
        reasons: list[str] = []
        if baseline_preserved:
            reasons.append("baseline_preserved")
        if execution_succeeded:
            reasons.append("targeted_execution_passed")
        if rerun_consistent:
            reasons.append("flake_check_passed")
        if recognized_sources:
            reasons.append("independently_justified")
        if contract_target_focus > 0.0:
            reasons.append("direct_contract_target")
        if issue_target_alignment > 0.0:
            reasons.append("issue_declared_target")
        if contract_axis_coverage >= 1.0:
            reasons.append("contract_matrix_complete_for_artifact")
        elif contract_axis_coverage > 0.0:
            reasons.append("contract_matrix_partial_for_artifact")
        if coverage_bonus > 0.0:
            reasons.append("coverage_signal")
        if mutation_bonus > 0.0:
            reasons.append("mutation_signal")
        if mutation_survivors_present:
            reasons.append("mutation_survivors")
        if redundancy_penalty > 0.0:
            reasons.append("redundancy_penalty")
        if patch_overfit_risk >= 0.35:
            reasons.append("patch_overfit_risk")
        if supplemental_issue_coverage:
            reasons.append("supplemental_issue_coverage")
        if schema_constraint_drift_risk > 0.0:
            reasons.append("schema_constraint_drift")
        if field_path_negative_shape_gap_count > 0:
            reasons.append("field_path_negative_shape_gap")
        if targeted_execution_failed:
            reasons.append("targeted_execution_failed")

        promotion_status = "exploratory"
        if (
            score >= promoted_threshold
            and baseline_preserved
            and execution_targeted_supported
            and execution_succeeded
            and rerun_consistent
            and independence >= 0.45
            and usefulness >= 0.45
            and patch_overfit_risk < 0.45
            and schema_constraint_drift_risk < 0.35
            and float(field_path_shape_signal.get("coverage_ratio") or 0.0) >= 0.67
            and mutation_discrimination_passed
            and not mutation_survivors_present
            and (
                issue_target_alignment <= 0.0
                or (
                    not supplemental_issue_coverage
                    and contract_target_focus > 0.0
                    and contract_axis_coverage >= 1.0
                )
            )
        ):
            promotion_status = "promoted"
            promoted_ids.append(str(scored.get("artifact_id") or ""))
        elif (
            score >= promoted_threshold
            and baseline_preserved
            and collection_succeeded
            and issue_target_alignment > 0.0
            and not supplemental_issue_coverage
            and contract_axis_coverage >= 1.0
            and independence >= 0.55
            and usefulness >= 0.55
            and patch_overfit_risk < 0.35
            and schema_constraint_drift_risk < 0.35
            and float(field_path_shape_signal.get("coverage_ratio") or 0.0) >= 1.0
            and coverage_bonus >= 0.60
            and contract_target_focus > 0.0
            and contract_axis_coverage >= 1.0
            and not mutation_survivors_present
        ):
            promotion_status = "promoted"
            promoted_ids.append(str(scored.get("artifact_id") or ""))
            reasons.append("strong_static_evidence")
        elif (
            score >= candidate_threshold
            and baseline_preserved
            and independence >= 0.32
            and patch_overfit_risk < 0.60
            and schema_constraint_drift_risk < 0.70
            and float(field_path_shape_signal.get("coverage_ratio") or 0.0) >= 0.67
            and not targeted_execution_failed
            and not mutation_survivors_present
            and (
                issue_target_alignment <= 0.0
                or (
                    not supplemental_issue_coverage
                    and contract_target_focus > 0.0
                    and contract_axis_coverage >= 1.0
                )
            )
        ):
            promotion_status = "candidate_public"
            candidate_ids.append(str(scored.get("artifact_id") or ""))
        else:
            exploratory_ids.append(str(scored.get("artifact_id") or ""))

        scored.update(
            {
                "promotion_status": promotion_status,
                "promotion_score": round(score, 4),
                "promotion_reasons": reasons,
                "scores": {
                    "stability": round(stability, 4),
                    "usefulness": round(usefulness, 4),
                    "independent_justification": round(independence, 4),
                    "coverage_signal": round(coverage_bonus, 4),
                    "mutation_signal": round(mutation_bonus, 4),
                    "coverage_signal_measured": coverage_signal_measured,
                    "mutation_signal_measured": mutation_signal_measured,
                    "mutation_discrimination_passed": mutation_discrimination_passed,
                    "contract_target_focus": round(contract_target_focus, 4),
                    "contract_axis_coverage": round(contract_axis_coverage, 4),
                    "issue_target_alignment": round(issue_target_alignment, 4),
                    "redundancy_penalty": round(redundancy_penalty, 4),
                    "patch_overfit_risk": round(patch_overfit_risk, 4),
                    "supplemental_issue_penalty": round(supplemental_issue_penalty, 4),
                    "schema_constraint_drift_risk": round(schema_constraint_drift_risk, 4),
                    "mutation_survivor_penalty": round(mutation_survivor_penalty, 4),
                    "field_path_negative_shape_coverage_ratio": round(
                        float(field_path_shape_signal.get("coverage_ratio") or 0.0),
                        4,
                    ),
                    "field_path_negative_shape_penalty": round(
                        field_path_negative_shape_penalty,
                        4,
                    ),
                },
                "issue_target_alignment": round(issue_target_alignment, 4),
                "supplemental_issue_coverage": supplemental_issue_coverage,
                "schema_constraint_drift_tokens": sorted(unjustified_schema_tokens),
                "field_path_contracts": list(field_path_shape_signal.get("field_paths") or []),
                "field_path_negative_shape_covered": list(
                    field_path_shape_signal.get("covered_shapes") or []
                ),
                "field_path_negative_shape_gaps": list(field_path_shape_signal.get("gaps") or []),
            }
        )
        scored_entries.append(scored)

    scored_entries, minimization_details = _minimize_public_test_artifacts(scored_entries)
    normalized["minimization_summary"] = {
        **dict(normalized.get("minimization_summary") or {}),
        **dict(minimization_details or {}),
    }
    promoted_ids = _dedupe_strings(
        [
            str(item.get("artifact_id") or "").strip()
            for item in scored_entries
            if str(item.get("promotion_status") or "").strip().lower() == "promoted"
        ]
    )
    candidate_ids = _dedupe_strings(
        [
            str(item.get("artifact_id") or "").strip()
            for item in scored_entries
            if str(item.get("promotion_status") or "").strip().lower() == "candidate_public"
        ]
    )
    exploratory_ids = _dedupe_strings(
        [
            str(item.get("artifact_id") or "").strip()
            for item in scored_entries
            if str(item.get("promotion_status") or "").strip().lower() == "exploratory"
        ]
    )

    strategy_diversity = len(
        {
            str(item.get("strategy") or "").strip().lower()
            for item in scored_entries
            if str(item.get("strategy") or "").strip()
        }
    )
    promoted_scores = [
        float(item.get("promotion_score"))
        for item in scored_entries
        if str(item.get("promotion_status") or "") == "promoted"
    ]
    candidate_scores = [
        float(item.get("promotion_score"))
        for item in scored_entries
        if str(item.get("promotion_status") or "") == "candidate_public"
    ]
    public_score = 0.0
    public_reason = "exploratory_synthetic_tests_only"
    if promoted_scores:
        public_score = min(
            0.95,
            0.58
            + (0.22 * (sum(promoted_scores) / len(promoted_scores)))
            + (0.05 * min(strategy_diversity, 4) / 4.0)
            + (0.05 if len(vendor_families) >= 2 else 0.0),
        )
        public_reason = "validated_synthetic_test_portfolio"
    elif candidate_scores:
        public_score = min(
            0.78,
            0.30
            + (0.24 * (sum(candidate_scores) / len(candidate_scores)))
            + (0.06 * min(strategy_diversity, 4) / 4.0)
            + (0.04 if len(vendor_families) >= 2 else 0.0),
        )
        public_reason = "candidate_synthetic_test_portfolio"
    elif scored_entries:
        public_score = min(0.25, 0.08 + (0.05 * min(strategy_diversity, 3) / 3.0))

    contract_matrix = _evaluate_contract_matrix(
        normalized,
        required_targets=required_contract_targets,
        required_axes=required_axes,
    )
    contract_gap_count = int(contract_matrix.get("gap_count") or 0)
    fully_covered_target_count = len(list(contract_matrix.get("fully_covered_targets") or []))
    primary_target_count = len(list(contract_matrix.get("primary_targets") or []))
    target_coverage_ratio = (
        fully_covered_target_count / primary_target_count if primary_target_count else 0.0
    )
    if contract_gap_count == 0 and primary_target_count > 0 and public_score > 0.0:
        public_score = min(0.96, public_score + 0.03)
    if contract_gap_count > 0:
        public_score = max(0.0, public_score - min(0.18, 0.04 * contract_gap_count))
        if public_reason in {
            "validated_synthetic_test_portfolio",
            "candidate_synthetic_test_portfolio",
        }:
            public_reason = f"{public_reason}_with_contract_gaps"
    if normalized_required_targets and primary_target_count > 0 and fully_covered_target_count == 0:
        public_score = min(public_score, 0.35)
        if public_reason.startswith("validated_synthetic_test_portfolio"):
            public_reason = "validated_synthetic_test_portfolio_missing_issue_contract_targets"
        elif public_reason.startswith("candidate_synthetic_test_portfolio"):
            public_reason = "candidate_synthetic_test_portfolio_missing_issue_contract_targets"

    promoted_first_descriptions: list[str] = []
    for status in ("promoted", "candidate_public", "exploratory"):
        for item in scored_entries:
            if str(item.get("promotion_status") or "") != status:
                continue
            promoted_first_descriptions.extend(list(item.get("test_descriptions") or []))

    design_rollup = _rollup_test_generation_design(
        normalized,
        scored_entries=scored_entries,
    )
    scored_entries = list(design_rollup.get("test_artifacts") or scored_entries)

    normalized.update(
        {
            "test_artifacts": scored_entries,
            "required_contract_targets": normalized_required_targets,
            "task_contract": _normalize_task_contract(normalized.get("task_contract")),
            "milestones": list(design_rollup.get("milestones") or []),
            "test_objectives": list(design_rollup.get("test_objectives") or []),
            "regression_suite_summary": dict(design_rollup.get("regression_suite_summary") or {}),
            "minimization_summary": dict(design_rollup.get("minimization_summary") or {}),
            "promoted_artifact_ids": _dedupe_strings(promoted_ids),
            "candidate_artifact_ids": _dedupe_strings(
                list(
                    dict(design_rollup.get("regression_suite_summary") or {}).get(
                        "candidate_artifact_ids"
                    )
                    or candidate_ids
                )
            ),
            "exploratory_artifact_ids": _dedupe_strings(exploratory_ids),
            "test_descriptions": _dedupe_strings(
                promoted_first_descriptions + list(normalized.get("test_descriptions") or [])
            ),
            "public_signal": {
                "score": round(public_score, 4),
                "reason": public_reason,
                "promoted_artifact_count": len(promoted_ids),
                "candidate_artifact_count": len(candidate_ids),
                "exploratory_artifact_count": len(exploratory_ids),
                "strategy_diversity": strategy_diversity,
                "vendor_families": sorted(vendor_families),
                "required_contract_axes": list(required_axes),
                "contract_matrix_gap_count": contract_gap_count,
                "contract_matrix_primary_target_count": primary_target_count,
                "contract_matrix_fully_covered_target_count": fully_covered_target_count,
                "contract_matrix_target_coverage_ratio": round(target_coverage_ratio, 4),
                "design_milestone_count": len(list(design_rollup.get("milestones") or [])),
                "design_objective_count": len(list(design_rollup.get("test_objectives") or [])),
                "dual_version_verified_artifact_count": int(
                    design_rollup.get("dual_version_verified_artifact_count") or 0
                ),
                "design_artifact_metadata_gap_count": int(
                    design_rollup.get("design_artifact_metadata_gap_count") or 0
                ),
            },
            "validation_summary": {
                **existing_validation_summary,
                "artifact_count": len(scored_entries),
                "promoted_artifact_count": len(promoted_ids),
                "candidate_artifact_count": len(candidate_ids),
                "exploratory_artifact_count": len(exploratory_ids),
                "strategy_diversity": strategy_diversity,
                "generator_vendor_count": len(vendor_families),
                "avg_promotion_score": (
                    round(
                        (
                            sum(
                                float(item.get("promotion_score") or 0.0) for item in scored_entries
                            )
                            / len(scored_entries)
                        ),
                        4,
                    )
                    if scored_entries
                    else 0.0
                ),
                "contract_matrix": contract_matrix,
                "issue_contract_targets": normalized_required_targets,
                "required_contract_axes": list(required_axes),
                "contract_matrix_gap_count": contract_gap_count,
                "contract_matrix_primary_target_count": primary_target_count,
                "contract_matrix_fully_covered_target_count": fully_covered_target_count,
                "contract_matrix_target_coverage_ratio": round(target_coverage_ratio, 4),
                "design_milestone_count": len(list(design_rollup.get("milestones") or [])),
                "design_objective_count": len(list(design_rollup.get("test_objectives") or [])),
                "dual_version_verified_artifact_count": int(
                    design_rollup.get("dual_version_verified_artifact_count") or 0
                ),
                "pass_then_invert_artifact_count": int(
                    design_rollup.get("pass_then_invert_artifact_count") or 0
                ),
                "design_artifact_metadata_gap_count": int(
                    design_rollup.get("design_artifact_metadata_gap_count") or 0
                ),
                "design_artifact_metadata_gaps": list(
                    design_rollup.get("design_artifact_metadata_gaps") or []
                ),
            },
        }
    )
    if not str(normalized.get("promotion_summary") or "").strip():
        normalized["promotion_summary"] = (
            f"Promoted {len(promoted_ids)} synthetic tests, kept {len(candidate_ids)} as "
            f"weak public evidence, and left {len(exploratory_ids)} exploratory."
        )
    if not str(normalized.get("portfolio_summary") or "").strip():
        normalized["portfolio_summary"] = (
            f"Generated {len(scored_entries)} test artifacts across {max(strategy_diversity, 1)} strategy buckets."
            if scored_entries
            else ""
        )
    if not str(normalized.get("summary") or "").strip():
        normalized["summary"] = normalized["promotion_summary"]
    if not str(normalized.get("test_code") or "").strip():
        for item in scored_entries:
            if str(item.get("promotion_status") or "") != "promoted":
                continue
            content = str(item.get("content") or "").strip()
            if content:
                normalized["test_code"] = content
                break
    return normalized
