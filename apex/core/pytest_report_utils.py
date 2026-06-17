"""
Helpers for parsing pytest JSON reports and matching canonical test expectations.
"""

from __future__ import annotations

import ast
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..controller_policy import canonical_expected_test_ids

_PARAMETRIZED_NODE_ID_RE = re.compile(r"^(?P<base>.+)\[(?P<params>.*)\]$")
_DIGIT_RUN_RE = re.compile(r"\d+")
_GENERATED_ORDINAL_SUFFIX_RE = re.compile(r"^(?P<prefix>.+-[A-Za-z_][A-Za-z_]*)(?:\d+)$")
_PARAM_SUFFIX_MIN_MATCH = 4


@dataclass(frozen=True)
class VisibleTestEditDisposition:
    """How to treat a protected visible-test file during final selection/eval."""

    action: str
    reason: str
    sanitized_text: Optional[str] = None


def load_pytest_json_report(report_path: Path) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def extract_pytest_report_tests(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    tests = payload.get("tests")
    if not isinstance(tests, list):
        return []
    return [test for test in tests if isinstance(test, dict)]


def extract_pytest_report_outcomes(tests: list[dict[str, Any]]) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for test in tests:
        node_id = test.get("nodeid")
        if not isinstance(node_id, str) or not node_id:
            continue
        outcome = pytest_report_outcome(test)
        if outcome:
            outcomes[node_id] = outcome
    return outcomes


def count_pytest_report_outcomes(outcomes: dict[str, str]) -> dict[str, int]:
    """Count JSON-report outcomes in the buckets APEX scoring uses."""

    return {
        "passed": sum(
            1 for outcome in outcomes.values() if outcome in {"passed", "xfailed", "xpassed"}
        ),
        "failed": sum(1 for outcome in outcomes.values() if outcome == "failed"),
        "errors": sum(1 for outcome in outcomes.values() if outcome == "error"),
    }


def extract_pytest_report_summary_counts(payload: Any) -> dict[str, int]:
    """Extract pytest-json-report summary counts in APEX scoring buckets."""

    summary = payload.get("summary") if isinstance(payload, dict) else {}
    if not isinstance(summary, dict):
        return {"passed": 0, "failed": 0, "errors": 0}

    def as_int(*keys: str) -> int:
        total = 0
        for key in keys:
            value = summary.get(key)
            if isinstance(value, int):
                total += max(0, value)
        return total

    return {
        "passed": as_int("passed", "xfailed", "xpassed"),
        "failed": as_int("failed", "failures", "failure"),
        "errors": as_int("error", "errors"),
    }


def pytest_report_outcome(test: dict[str, Any]) -> str:
    top_level = normalize_pytest_outcome(test.get("outcome"))
    call = test.get("call") or {}
    if isinstance(call, dict):
        call_outcome = normalize_pytest_outcome(call.get("outcome"))
        if _pytest_report_has_xfail_marker(test, call):
            if call_outcome == "skipped" or top_level == "skipped":
                return "xfailed"
            if (call_outcome == "passed" or top_level == "passed") and top_level != "failed":
                return "xpassed"
        if top_level:
            return top_level
        if call_outcome == "skipped":
            keywords = test.get("keywords") or []
            if isinstance(keywords, list) and "xfail" in keywords:
                return "xfailed"
        if call_outcome:
            return call_outcome
    if top_level:
        return top_level

    for phase in ("setup", "teardown"):
        phase_result = test.get(phase) or {}
        if not isinstance(phase_result, dict):
            continue
        outcome = normalize_pytest_outcome(phase_result.get("outcome"))
        if outcome:
            return outcome

    return ""


def _pytest_report_has_xfail_marker(test: dict[str, Any], call: dict[str, Any]) -> bool:
    keywords = test.get("keywords") or []
    if isinstance(keywords, list) and "xfail" in keywords:
        return True
    for payload in (test, call):
        if payload.get("wasxfail"):
            return True
    return False


def normalize_pytest_outcome(value: Any) -> str:
    if value is None:
        return ""
    normalized = str(value).strip().lower()
    aliases = {
        "pass": "passed",
        "passed": "passed",
        "fail": "failed",
        "failure": "failed",
        "failures": "failed",
        "failed": "failed",
        "error": "error",
        "errors": "error",
        "xfail": "xfailed",
        "xfailed": "xfailed",
        "xpass": "xpassed",
        "xpassed": "xpassed",
        "skip": "skipped",
        "skipped": "skipped",
    }
    return aliases.get(normalized, "")


def parse_pytest_terminal_summary_counts(output: Any) -> dict[str, int]:
    """Parse pytest's final textual summary into normalized outcome buckets."""

    text = str(output or "")
    # Pytest terminal summaries report xfailed/xpassed/skipped as collected outcomes.
    result = {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "xfailed": 0,
        "xpassed": 0,
    }
    aliases = {
        "passed": ("passed", "pass"),
        "failed": ("failed", "failures", "failure"),
        "errors": ("errors", "error"),
        "skipped": ("skipped", "skip"),
        "xfailed": ("xfailed", "xfail"),
        "xpassed": ("xpassed", "xpass"),
    }
    candidate_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not re.search(
            r"\b\d+\s+(?:passed|pass|failed|failures|failure|errors?|"
            r"skipped|skip|xfailed|xfail|xpassed|xpass)\b",
            line,
            flags=re.I,
        ):
            continue
        normalized = line.strip("= ").strip()
        # Pytest tracebacks can contain "N errors" prose; only final summary lines are counts.
        summary_like = re.search(
            r"\bin\s+\d+(?:\.\d+)?\s*(?:s|sec|secs|second|seconds)\b", line, re.I
        ) is not None or (line.startswith("=") and line.endswith("="))
        if summary_like:
            candidate_lines.append(normalized)
    if candidate_lines:
        text = candidate_lines[-1]
    for key, words in aliases.items():
        for word in words:
            for match in re.finditer(rf"\b(\d+)\s+{re.escape(word)}\b", text, flags=re.I):
                result[key] = max(result[key], int(match.group(1)))
    return result


def summarize_expected_pytest_coverage(
    expected_test_ids: list[str],
    outcomes: dict[str, str],
) -> dict[str, Any]:
    """Compute expected-vs-discovered coverage stats including missing IDs.

    The summary now includes ``missing_test_ids`` — the explicit list of
    expected test IDs the actual pytest run did not collect or report on.
    Surfacing the IDs (not just the count) lets the residual-feedback
    path tell the agent *which specific tests are absent*, so the agent
    can investigate marker filters / parametrize / conftest collection
    skips rather than silently undercounting.
    """

    expected = [test_id for test_id in expected_test_ids if test_id]
    if not expected:
        return {
            "matched_expected_test_count": 0,
            "missing_expected_test_count": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "missing_test_ids": [],
        }

    passed = 0
    errors = 0
    skipped = 0
    matched_expected: set[str] = set()
    used_actual: set[str] = set()

    def record(outcome: str) -> None:
        nonlocal passed, errors, skipped
        if outcome in {"passed", "xfailed", "xpassed"}:
            passed += 1
        elif outcome == "error":
            errors += 1
        elif outcome == "skipped":
            skipped += 1

    def dynamic_param_shape(node_id: str) -> tuple[str, str] | None:
        match = _PARAMETRIZED_NODE_ID_RE.match(node_id)
        if not match:
            return None
        params = match.group("params")
        if not _DIGIT_RUN_RE.search(params):
            return None
        shape = _DIGIT_RUN_RE.sub(
            lambda digit_match: f"<digits:{len(digit_match.group(0))}>",
            params,
        )
        return match.group("base"), shape

    def dynamic_shape_is_generic(shape: str) -> bool:
        remainder = re.sub(r"<digits:\d+>", "", shape)
        return not re.search(r"[A-Za-z_]", remainder)

    def generated_ordinal_param_shape(node_id: str) -> tuple[str, str] | None:
        match = _PARAMETRIZED_NODE_ID_RE.match(node_id)
        if not match:
            return None
        suffix = _GENERATED_ORDINAL_SUFFIX_RE.match(match.group("params"))
        if not suffix:
            return None
        return match.group("base"), f"{suffix.group('prefix')}<ordinal>"

    def split_parametrized(node_id: str) -> tuple[str, str | None]:
        match = _PARAMETRIZED_NODE_ID_RE.match(node_id)
        if not match:
            return node_id, None
        return match.group("base"), match.group("params")

    def longest_common_suffix_len(a: str, b: str) -> int:
        limit = min(len(a), len(b))
        matched = 0
        for offset in range(1, limit + 1):
            if a[-offset] == b[-offset]:
                matched = offset
            else:
                break
        return matched

    for test_id in expected:
        outcome = outcomes.get(test_id)
        if outcome is None:
            continue
        matched_expected.add(test_id)
        used_actual.add(test_id)
        record(outcome)

    remaining_expected_by_base: dict[str, list[str]] = {}
    for test_id in expected:
        if test_id in matched_expected:
            continue
        base = parameterized_node_id_base(test_id)
        if base is None:
            continue
        remaining_expected_by_base.setdefault(base, []).append(test_id)

    remaining_actual_by_base: dict[str, list[tuple[str, str]]] = {}
    for node_id, outcome in outcomes.items():
        if node_id in used_actual:
            continue
        base = parameterized_node_id_base(node_id)
        if base is None:
            continue
        remaining_actual_by_base.setdefault(base, []).append((node_id, outcome))

    for test_id in expected:
        if test_id in matched_expected:
            continue
        base, params = split_parametrized(test_id)
        if params is None:
            continue
        candidates = [
            (node_id, actual_params, outcome)
            for node_id, outcome in remaining_actual_by_base.get(base, [])
            if node_id not in used_actual
            for _actual_base, actual_params in [split_parametrized(node_id)]
            if actual_params is not None
        ]
        best_node_id: str | None = None
        best_outcome = ""
        best_overlap = 0
        for node_id, actual_params, outcome in candidates:
            overlap = longest_common_suffix_len(params, actual_params)
            if overlap > best_overlap:
                best_overlap = overlap
                best_node_id = node_id
                best_outcome = outcome
        if best_node_id is not None and best_overlap >= _PARAM_SUFFIX_MIN_MATCH:
            matched_expected.add(test_id)
            used_actual.add(best_node_id)
            record(best_outcome)

    remaining_expected_by_ordinal_shape: dict[tuple[str, str], list[str]] = {}
    for test_id in expected:
        if test_id in matched_expected:
            continue
        key = generated_ordinal_param_shape(test_id)
        if key is None:
            continue
        remaining_expected_by_ordinal_shape.setdefault(key, []).append(test_id)

    remaining_actual_by_ordinal_shape: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for node_id, outcome in outcomes.items():
        if node_id in used_actual:
            continue
        key = generated_ordinal_param_shape(node_id)
        if key is None:
            continue
        remaining_actual_by_ordinal_shape.setdefault(key, []).append((node_id, outcome))

    for key, expected_group in remaining_expected_by_ordinal_shape.items():
        actual_group = remaining_actual_by_ordinal_shape.get(key)
        if not actual_group or len(actual_group) != len(expected_group):
            continue
        for expected_id, (node_id, outcome) in zip(
            sorted(expected_group),
            sorted(actual_group),
        ):
            matched_expected.add(expected_id)
            used_actual.add(node_id)
            record(outcome)

    remaining_expected_by_shape: dict[tuple[str, str], list[str]] = {}
    for test_id in expected:
        if test_id in matched_expected:
            continue
        key = dynamic_param_shape(test_id)
        if key is None:
            continue
        remaining_expected_by_shape.setdefault(key, []).append(test_id)

    remaining_actual_by_shape: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for node_id, outcome in outcomes.items():
        if node_id in used_actual:
            continue
        key = dynamic_param_shape(node_id)
        if key is None:
            continue
        remaining_actual_by_shape.setdefault(key, []).append((node_id, outcome))

    for key, expected_group in remaining_expected_by_shape.items():
        # Keep scorer matching aligned with _apex_expected_ids_filter: a
        # one-to-one dynamic-shape group should count even when the same
        # parametrized base collected unrelated extra variants.
        actual_group = remaining_actual_by_shape.get(key)
        if actual_group:
            actual_group = [
                (node_id, outcome)
                for node_id, outcome in actual_group
                if node_id not in used_actual
            ]
        if not actual_group or len(actual_group) != len(expected_group):
            continue
        base, shape = key
        if dynamic_shape_is_generic(shape):
            # Commit0/Python expected-id fact: numeric-only param shapes such as
            # dates/times are ambiguous unless every same-base numeric shape is balanced.
            expected_shapes = {
                candidate_shape
                for candidate_base, candidate_shape in remaining_expected_by_shape
                if candidate_base == base and dynamic_shape_is_generic(candidate_shape)
            }
            actual_shapes = {
                candidate_shape
                for candidate_base, candidate_shape in remaining_actual_by_shape
                if candidate_base == base and dynamic_shape_is_generic(candidate_shape)
            }
            if expected_shapes != actual_shapes:
                continue
            if any(
                len(remaining_expected_by_shape.get((base, candidate_shape), []))
                != len(remaining_actual_by_shape.get((base, candidate_shape), []))
                for candidate_shape in expected_shapes
            ):
                continue
        for expected_id, (node_id, outcome) in zip(
            sorted(expected_group),
            sorted(actual_group),
        ):
            matched_expected.add(expected_id)
            used_actual.add(node_id)
            record(outcome)

    for base, expected_group in remaining_expected_by_base.items():
        expected_group = [test_id for test_id in expected_group if test_id not in matched_expected]
        if not expected_group:
            continue
        actual_group = remaining_actual_by_base.get(base)
        if actual_group:
            actual_group = [
                (node_id, outcome)
                for node_id, outcome in actual_group
                if node_id not in used_actual
            ]
        if not actual_group or len(actual_group) != len(expected_group):
            continue
        matched_expected.update(expected_group)
        for node_id, outcome in sorted(actual_group):
            used_actual.add(node_id)
            record(outcome)

    # Package-root prefix reconciliation. Some pytest invocations (notably a
    # broad repo-root run that bypasses the expected-ID wrapper) emit json-report
    # nodeids relative to the package directory (``distributions/tests/x.py::t``)
    # while the Commit0 expected-ID inventory is repo-relative
    # (``statsmodels/distributions/tests/x.py::t``). When an observed nodeid is
    # exactly the expected nodeid with one or more *leading* path segments
    # dropped (the package root), match it so a fully-passing run is not
    # misreported as a 100%-missing coverage collapse.
    #
    # This is strictly directional: the observed path must be a tail (suffix) of
    # the expected path. The reverse — an observed nodeid with EXTRA leading
    # segments (e.g. a leaked ``.../workspaces/<repo>/rollout_0/tests/...`` run
    # under the wrong rootdir) — is deliberately NOT matched, because such a
    # nodeid was collected from outside the candidate's own root and must not be
    # credited. Exact matches were already consumed above, so this fallback only
    # ever sees genuinely prefix-stripped nodeids.
    still_unused_actual = any(node_id not in used_actual for node_id in outcomes)
    if still_unused_actual and len(matched_expected) < len(expected):

        def _path_and_test(node_id: str) -> tuple[str, str]:
            path, _sep, test = node_id.partition("::")
            return path, test

        # Index unused observed nodeids by their FULL (path, test) so an expected
        # id can look up an observed whose full path equals a tail of its own.
        actual_by_full_key: dict[tuple[str, str], str] = {}
        for node_id in outcomes:
            if node_id in used_actual:
                continue
            actual_path, actual_test = _path_and_test(node_id)
            actual_by_full_key.setdefault((actual_path, actual_test), node_id)

        def _expected_path_tails(path: str) -> list[str]:
            # Tails with >=1 leading segment dropped, longest first. The full
            # path itself is excluded (an equal path is an exact match, already
            # handled), so we only accept package-root-stripped observed nodeids.
            parts = [part for part in path.split("/") if part != ""]
            return ["/".join(parts[i:]) for i in range(1, len(parts))]

        for test_id in expected:
            if test_id in matched_expected:
                continue
            expected_path, expected_test = _path_and_test(test_id)
            for tail in _expected_path_tails(expected_path):
                node_id = actual_by_full_key.get((tail, expected_test))
                if node_id is None or node_id in used_actual:
                    continue
                matched_expected.add(test_id)
                used_actual.add(node_id)
                record(outcomes[node_id])
                break

    matched_expected_count = len(matched_expected)
    # ``failed`` covers expected IDs that were neither passed/errored/skipped:
    # genuine pytest failures plus expected IDs pytest never reported on
    # (collection breakage). Skipped tests are tracked separately so callers
    # can keep them out of the pass-rate denominator.
    failed = len(expected) - passed - errors - skipped
    # Preserve original expected-list order so downstream feedback
    # references match the canonical commit0/benchmark order.
    missing_test_ids = [test_id for test_id in expected if test_id not in matched_expected]
    return {
        "matched_expected_test_count": matched_expected_count,
        "missing_expected_test_count": len(expected) - matched_expected_count,
        "passed": passed,
        "failed": max(failed, 0),
        "errors": errors,
        "skipped": skipped,
        "missing_test_ids": missing_test_ids,
    }


def parameterized_node_id_base(node_id: str) -> Optional[str]:
    match = _PARAMETRIZED_NODE_ID_RE.match(node_id)
    if not match:
        return None
    return match.group("base")


def pytest_node_id_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.split("::", 1)[0].strip()


def looks_like_test_path(path: Any) -> bool:
    normalized = pytest_node_id_path(path)
    if not normalized:
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
        or name == "tests.py"
        or "/tests/" in lowered_path
    )


def _test_context_source(test_context: Any) -> Any:
    if test_context is None:
        return None
    context_source = test_context
    if isinstance(test_context, dict):
        nested = test_context.get("test_context")
    else:
        nested = getattr(test_context, "test_context", None)
    if nested is not None:
        context_source = nested
    return context_source


def _test_context_list(test_context: Any, name: str) -> list[str]:
    context_source = _test_context_source(test_context)
    if context_source is None:
        return []
    raw: Any
    if isinstance(context_source, dict):
        raw = context_source.get(name)
    else:
        raw = getattr(context_source, name, None)
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def incomplete_test_files_from_context(test_context: Any) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for path in _test_context_list(test_context, "incomplete_test_files"):
        normalized = pytest_node_id_path(path)
        if not normalized or normalized in seen or not looks_like_test_path(normalized):
            continue
        ordered.append(normalized)
        seen.add(normalized)
    return ordered


def _is_docstring_stmt(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(getattr(node, "value", None), ast.Constant)
        and isinstance(node.value.value, str)
    )


def _split_function_body(body: list[ast.stmt]) -> tuple[list[ast.stmt], list[ast.stmt]]:
    if body and _is_docstring_stmt(body[0]):
        return [body[0]], body[1:]
    return [], body


def _is_placeholder_stmt(node: ast.stmt) -> bool:
    if isinstance(node, ast.Pass):
        return True
    if isinstance(node, ast.Expr):
        value = node.value
        if isinstance(value, ast.Constant):
            if value.value is Ellipsis:
                return True
            if isinstance(value.value, str):
                lowered = value.value.strip().lower()
                return "todo" in lowered or "implement" in lowered
        return False
    if isinstance(node, ast.Raise):
        exc = node.exc
        if isinstance(exc, ast.Call):
            exc = exc.func
        name = ""
        if isinstance(exc, ast.Name):
            name = exc.id
        elif isinstance(exc, ast.Attribute):
            name = exc.attr
        return name.lower() in {"notimplementederror", "notimplemented"}
    return False


def _has_placeholder_body(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    _, tail = _split_function_body(list(node.body))
    return bool(tail) and all(_is_placeholder_stmt(stmt) for stmt in tail)


def _function_path_maps(
    tree: ast.AST,
) -> tuple[dict[tuple[str, ...], ast.AST], dict[tuple[str, ...], ast.AST]]:
    functions: dict[tuple[str, ...], ast.AST] = {}
    classes: dict[tuple[str, ...], ast.AST] = {}

    def visit_body(body: list[ast.stmt], prefix: tuple[str, ...]) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                path = prefix + (f"class:{node.name}",)
                classes[path] = node
                visit_body(list(node.body), path)
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                path = prefix + (
                    ("async:" if isinstance(node, ast.AsyncFunctionDef) else "def:") + node.name,
                )
                functions[path] = node

    module_body = getattr(tree, "body", None)
    if isinstance(module_body, list):
        visit_body(module_body, ())
    return functions, classes


def _class_header_signature(
    node: ast.ClassDef,
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        node.name,
        tuple(ast.dump(base, include_attributes=False) for base in node.bases),
        tuple(ast.dump(keyword, include_attributes=False) for keyword in node.keywords),
        tuple(ast.dump(decorator, include_attributes=False) for decorator in node.decorator_list),
    )


def _copy_allowed_placeholder_completions(
    baseline_tree: ast.AST,
    candidate_tree: ast.AST,
) -> ast.AST:
    sanitized_tree = copy.deepcopy(baseline_tree)
    baseline_functions, baseline_classes = _function_path_maps(baseline_tree)
    candidate_functions, candidate_classes = _function_path_maps(candidate_tree)
    sanitized_functions, sanitized_classes = _function_path_maps(sanitized_tree)

    for path, baseline_node in baseline_functions.items():
        if not _has_placeholder_body(baseline_node):
            continue
        candidate_node = candidate_functions.get(path)
        sanitized_node = sanitized_functions.get(path)
        if not isinstance(candidate_node, type(baseline_node)) or not isinstance(
            sanitized_node,
            type(baseline_node),
        ):
            continue

        class_path = path[:-1]
        if class_path:
            baseline_class = baseline_classes.get(class_path)
            candidate_class = candidate_classes.get(class_path)
            if not isinstance(baseline_class, ast.ClassDef) or not isinstance(
                candidate_class,
                ast.ClassDef,
            ):
                continue
            if _class_header_signature(baseline_class) != _class_header_signature(candidate_class):
                continue

        baseline_prefix, _ = _split_function_body(list(baseline_node.body))
        _, candidate_tail = _split_function_body(list(candidate_node.body))
        if not candidate_tail or all(_is_placeholder_stmt(stmt) for stmt in candidate_tail):
            continue

        sanitized_node.args = copy.deepcopy(candidate_node.args)
        sanitized_node.decorator_list = list(copy.deepcopy(candidate_node.decorator_list))
        sanitized_node.returns = copy.deepcopy(candidate_node.returns)
        if hasattr(sanitized_node, "type_comment"):
            sanitized_node.type_comment = getattr(candidate_node, "type_comment", None)
        sanitized_node.body = list(copy.deepcopy(candidate_node.body))
        if baseline_prefix and not _is_docstring_stmt(sanitized_node.body[0]):
            sanitized_node.body = list(copy.deepcopy(baseline_prefix)) + list(
                copy.deepcopy(candidate_tail)
            )

    return sanitized_tree


def _normalize_visible_test_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized if normalized.endswith("\n") else normalized + "\n"


def analyze_visible_test_edit(
    *,
    rel_path: str,
    baseline_text: str,
    candidate_text: str,
    allow_placeholder_completion: bool,
) -> VisibleTestEditDisposition:
    baseline_normalized = _normalize_visible_test_text(baseline_text)
    candidate_normalized = _normalize_visible_test_text(candidate_text)
    if baseline_normalized == candidate_normalized:
        return VisibleTestEditDisposition(
            action="restore",
            reason=f"{rel_path} has no material visible-test change after normalization.",
        )
    if not allow_placeholder_completion:
        return VisibleTestEditDisposition(
            action="restore",
            reason=f"{rel_path} is a protected visible test file.",
        )

    try:
        baseline_tree = ast.parse(baseline_normalized)
        candidate_tree = ast.parse(candidate_normalized)
    except SyntaxError as exc:
        return VisibleTestEditDisposition(
            action="restore",
            reason=f"{rel_path} could not be parsed for protected visible-test analysis: {exc.msg}.",
        )

    sanitized_tree = _copy_allowed_placeholder_completions(
        baseline_tree,
        candidate_tree,
    )
    baseline_dump = ast.dump(baseline_tree, include_attributes=False)
    candidate_dump = ast.dump(candidate_tree, include_attributes=False)
    sanitized_dump = ast.dump(sanitized_tree, include_attributes=False)

    if sanitized_dump == baseline_dump:
        return VisibleTestEditDisposition(
            action="restore",
            reason=(
                f"{rel_path} edited protected visible tests outside explicit placeholder scaffolds."
            ),
        )
    if candidate_dump == sanitized_dump:
        return VisibleTestEditDisposition(
            action="allow",
            reason=(
                f"{rel_path} only completes explicit placeholder bodies in an incomplete visible test."
            ),
        )
    return VisibleTestEditDisposition(
        action="sanitize",
        reason=(
            f"{rel_path} mixed placeholder completion with unrelated protected visible-test edits."
        ),
        sanitized_text=_normalize_visible_test_text(ast.unparse(sanitized_tree)),
    )


def protected_test_files_from_context(
    test_context: Any,
    *,
    exclude_incomplete_test_files: bool = False,
) -> list[str]:
    if test_context is None:
        return []

    protected: list[str] = []
    protected.extend(
        path
        for path in _test_context_list(test_context, "focus_test_files")
        if looks_like_test_path(path)
    )
    protected.extend(
        path
        for path in (
            pytest_node_id_path(test_id)
            for test_id in _test_context_list(test_context, "failing_test_ids")
        )
        if looks_like_test_path(path)
    )
    protected.extend(
        path
        for path in (
            pytest_node_id_path(test_id) for test_id in canonical_expected_test_ids(test_context)
        )
        if looks_like_test_path(path)
    )

    incomplete = set(incomplete_test_files_from_context(test_context))
    ordered: list[str] = []
    seen: set[str] = set()
    for path in protected:
        normalized = pytest_node_id_path(path)
        if not normalized or normalized in seen:
            continue
        # Some rollout-time write boundaries intentionally relax explicit
        # scaffold tests so the agent can inspect or complete placeholder
        # bodies while searching. Final-candidate selection/evaluation
        # must use the default strict mode so protected visible tests are
        # still restored or rejected before scoring.
        if exclude_incomplete_test_files and normalized in incomplete:
            continue
        ordered.append(normalized)
        seen.add(normalized)
    return ordered
