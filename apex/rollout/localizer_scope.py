"""Adaptive localizer scope diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import PurePosixPath
from typing import Any, Iterable

HOST_PATH_PREFIXES = (
    "/usr/",
    "/opt/",
    "/System/",
    "/Library/",
    "/Applications/",
    "/private/",
    "/var/",
)

HOST_PATH_MARKERS = (
    "site-packages/",
    "dist-packages/",
    ".venv/",
    "venv/",
    ".tox/",
    "__pycache__/",
)

PATH_PATTERN_METACHARS = frozenset("*?[")

TEST_DIR_PARTS = frozenset({"tests", "test", "__tests__"})
TEST_FILENAME_GLOBS = (
    "test_*.py",
    "*_test.py",
    "test_*.pyx",
    "*.test.ts",
    "*.test.tsx",
    "*.test.js",
    "*.spec.ts",
    "*.spec.js",
    "*Test.java",
    "*_test.go",
    "*_spec.rb",
    "tests.py",
)
TEST_FIXTURE_NAMES = frozenset({"conftest.py", "fixtures.py", "fixture.py"})
APEX_HARNESS_BASENAMES = frozenset(
    {
        "_apex_run_expected_ids.py",
        "_apex_expected_ids_filter.py",
        ".apex_expected_test_ids.txt",
        "_apex_expected_test_ids.txt",
    }
)


@dataclass
class LocalizerFocus:
    editable_focus_files: list[str] = field(default_factory=list)
    noneditable_context_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "editable_focus_files": list(self.editable_focus_files),
            "noneditable_context_files": list(self.noneditable_context_files),
        }


def _normalize(path: Any) -> str:
    text = str(path or "").strip().replace("\\", "/")
    if "::" in text:
        text = text.split("::", 1)[0]
    while text.startswith("./"):
        text = text[2:]
    return text


def _normalized_incomplete_test_files(values: Iterable[Any]) -> set[str]:
    return {_normalize(value) for value in values if _normalize(value)}


def is_test_path(path: Any, *, repo_relative: bool = True) -> bool:
    normalized = _normalize(path)
    if not normalized:
        return False
    if repo_relative and (normalized.startswith("/") or normalized.startswith("~")):
        return False
    parts = PurePosixPath(normalized).parts
    if not parts:
        return False
    name = parts[-1]
    if any(part in TEST_DIR_PARTS for part in parts):
        return True
    if any(fnmatch(name, pattern) for pattern in TEST_FILENAME_GLOBS):
        return True
    if name == "conftest.py":
        return True
    return name in TEST_FIXTURE_NAMES and any(part in TEST_DIR_PARTS for part in parts)


def is_apex_harness_path(path: Any, *, repo_relative: bool = True) -> bool:
    normalized = _normalize(path)
    if not normalized:
        return False
    if repo_relative and (normalized.startswith("/") or normalized.startswith("~")):
        return False
    return PurePosixPath(normalized).name in APEX_HARNESS_BASENAMES


def has_unresolved_path_syntax(path: Any) -> bool:
    normalized = _normalize(path)
    if not normalized:
        return False
    if any(char in normalized for char in PATH_PATTERN_METACHARS):
        return True
    return "$" in normalized


def is_repo_relative_editable_path(
    path: Any,
    *,
    evidence_mode: str = "",
    incomplete_test_files: Iterable[Any] = (),
) -> bool:
    normalized = _normalize(path)
    if not normalized:
        return False
    if has_unresolved_path_syntax(normalized):
        return False
    if normalized.startswith(HOST_PATH_PREFIXES):
        return False
    if normalized.startswith("/") or normalized.startswith("~"):
        return False
    parts = PurePosixPath(normalized).parts
    if any(part == ".." for part in parts):
        return False
    if any(marker in normalized for marker in HOST_PATH_MARKERS):
        return False
    if is_apex_harness_path(normalized):
        return False
    if str(evidence_mode or "").strip() == "gold_suite_visible" and is_test_path(normalized):
        if normalized not in _normalized_incomplete_test_files(incomplete_test_files):
            return False
    return True


def split_localizer_focus(
    paths: Iterable[Any],
    *,
    evidence_mode: str = "",
    incomplete_test_files: Iterable[Any] = (),
) -> LocalizerFocus:
    editable: list[str] = []
    noneditable: list[str] = []
    for raw in paths:
        normalized = _normalize(raw)
        if not normalized:
            continue
        if is_repo_relative_editable_path(
            normalized,
            evidence_mode=evidence_mode,
            incomplete_test_files=incomplete_test_files,
        ):
            editable.append(normalized)
        else:
            noneditable.append(normalized)
    return LocalizerFocus(
        editable_focus_files=sorted(dict.fromkeys(editable)),
        noneditable_context_files=sorted(dict.fromkeys(noneditable)),
    )


def module_group_write_scope(
    search_policy: Any,
) -> tuple[list[str], bool]:
    """Return ``(allowed_owned_plus_bridge_files, enforce)`` for a module group.

    TIER 2 (T2.3): a decomposition-scale module-group brief carries its owned +
    bridge files and an ``enforce_module_group_write_scope`` flag in the
    rollout brief's ``search_policy``. This helper extracts the enforced
    write-scope allow-list (owned files, with bridge files as read-context) so
    the engine can install an ENFORCED ACI write scope
    (``set_write_scope(..., enforce=True)``) that reverts off-group edits via
    ``_enforce_write_scope``. Returns ``([], False)`` for any non-module-group
    brief so today's behavior is unchanged off the giants.
    """
    policy = search_policy if isinstance(search_policy, dict) else {}
    if not bool(policy.get("decomposition_module_group")):
        return [], False
    if not bool(policy.get("enforce_module_group_write_scope")):
        return [], False
    owned = [_normalize(path) for path in list(policy.get("module_group_owned_files") or [])]
    owned = [path for path in owned if path]
    if not owned:
        return [], False
    return sorted(dict.fromkeys(owned)), True


def infer_scope_class(
    *,
    editable_focus_files: Iterable[Any],
    solution_changed_files: Iterable[Any],
    generated_test_task: bool = False,
) -> str:
    if generated_test_task:
        return "generated_test_task"
    changed = [path for path in solution_changed_files if _normalize(path)]
    focus = [path for path in editable_focus_files if _normalize(path)]
    if len(changed) >= 50:
        return "library_reconstruction"
    if len(changed) >= 10:
        return "module_completion"
    if any(
        _normalize(path).startswith(("pyproject.", "setup.", "setup/", "config/"))
        for path in changed
    ):
        return "infrastructure_fix"
    if focus or changed:
        return "targeted_fix"
    return "unknown"


def localizer_severity(
    *,
    scope_class: str,
    out_of_scope_solution_files: Iterable[Any],
) -> str:
    count = len([path for path in out_of_scope_solution_files if _normalize(path)])
    if count <= 0:
        return "none"
    if scope_class == "library_reconstruction":
        return "low"
    if scope_class in {"module_completion", "infrastructure_fix"}:
        return "medium"
    return "high"
