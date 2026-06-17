"""Static quality checks for generated test artifacts.

Execution signals (pass@1, F2P, coverage, mutation) are the primary
test-generation gates. This module catches cheap static anti-patterns
that often pass execution while contributing little oracle value:
assertion-free tests, tautological assertions, and broad exception
oracles.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from apex.core.generated_tests import normalize_generated_test_content


@dataclass
class TestQualityIssue:
    path: str
    code: str
    message: str
    line: int = 0
    severity: str = "warning"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "code": self.code,
            "message": self.message,
            "line": self.line,
            "severity": self.severity,
        }


@dataclass
class TestArtifactQuality:
    path: str
    language: str
    parse_ok: bool = True
    test_function_count: int = 0
    assertion_count: int = 0
    skipped_test_count: int = 0
    anchor_test_count: int = 0
    focal_reference_count: int = 0
    meaningful_test_count: int = 0
    issues: list[TestQualityIssue] = field(default_factory=list)

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    @property
    def assertion_effect_score(self) -> float:
        if not self.parse_ok or self.test_function_count <= 0:
            return 0.0
        if self.meaningful_test_count <= 0:
            return 0.0
        if self.assertion_count <= 0:
            return 0.0
        blocking = sum(1 for issue in self.issues if issue.severity == "error")
        warnings = sum(1 for issue in self.issues if issue.severity == "warning")
        return max(0.0, 1.0 - 0.5 * blocking - 0.2 * warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "language": self.language,
            "parse_ok": self.parse_ok,
            "test_function_count": self.test_function_count,
            "assertion_count": self.assertion_count,
            "skipped_test_count": self.skipped_test_count,
            "anchor_test_count": self.anchor_test_count,
            "focal_reference_count": self.focal_reference_count,
            "meaningful_test_count": self.meaningful_test_count,
            "issue_count": self.issue_count,
            "assertion_effect_score": round(self.assertion_effect_score, 4),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass
class TestQualityReport:
    artifacts: list[TestArtifactQuality] = field(default_factory=list)

    @property
    def artifact_count(self) -> int:
        return len(self.artifacts)

    @property
    def issue_count(self) -> int:
        return sum(artifact.issue_count for artifact in self.artifacts)

    @property
    def weak_artifact_count(self) -> int:
        return sum(1 for artifact in self.artifacts if artifact.issue_count > 0)

    @property
    def mean_assertion_effect_score(self) -> float:
        if not self.artifacts:
            return 0.0
        return sum(a.assertion_effect_score for a in self.artifacts) / len(self.artifacts)

    @property
    def meaningful_test_count(self) -> int:
        return sum(artifact.meaningful_test_count for artifact in self.artifacts)

    @property
    def skipped_test_count(self) -> int:
        return sum(artifact.skipped_test_count for artifact in self.artifacts)

    @property
    def anchor_test_count(self) -> int:
        return sum(artifact.anchor_test_count for artifact in self.artifacts)

    @property
    def focal_reference_count(self) -> int:
        return sum(artifact.focal_reference_count for artifact in self.artifacts)

    def to_dict(self) -> dict[str, Any]:
        issue_counts = Counter(
            issue.code for artifact in self.artifacts for issue in artifact.issues
        )
        return {
            "artifact_count": self.artifact_count,
            "weak_artifact_count": self.weak_artifact_count,
            "issue_count": self.issue_count,
            "meaningful_test_count": self.meaningful_test_count,
            "skipped_test_count": self.skipped_test_count,
            "anchor_test_count": self.anchor_test_count,
            "focal_reference_count": self.focal_reference_count,
            "issue_counts": dict(sorted(issue_counts.items())),
            "mean_assertion_effect_score": round(
                self.mean_assertion_effect_score,
                4,
            ),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


def analyze_test_artifacts_quality(
    artifacts: list[dict[str, Any]],
    *,
    language: str = "python",
    focal_module: str = "",
    focal_symbols: list[str] | set[str] | tuple[str, ...] | None = None,
) -> TestQualityReport:
    """Analyze a generated test artifact list for static oracle weaknesses."""
    report = TestQualityReport()
    for artifact in artifacts or []:
        if not isinstance(artifact, dict):
            continue
        path = str(artifact.get("path") or "").strip()
        content = str(artifact.get("content") or "")
        report.artifacts.append(
            analyze_test_artifact_quality(
                path=path,
                content=content,
                language=language,
                focal_module=focal_module,
                focal_symbols=focal_symbols,
            )
        )
    return report


def analyze_test_artifact_quality(
    *,
    path: str,
    content: str,
    language: str = "python",
    focal_module: str = "",
    focal_symbols: list[str] | set[str] | tuple[str, ...] | None = None,
) -> TestArtifactQuality:
    """Analyze one generated test artifact for static oracle weaknesses."""
    normalized_language = (language or "").lower()
    content = normalize_generated_test_content(content or "")
    if normalized_language not in {"python", "py", "python3"}:
        if normalized_language in {"javascript", "js", "jsx", "typescript", "ts", "tsx"}:
            return _analyze_js_ts_test_quality(
                path=path,
                content=content,
                language=normalized_language,
                focal_module=focal_module,
                focal_symbols=focal_symbols,
            )
        if normalized_language in {"go", "golang"}:
            return _analyze_go_test_quality(path=path, content=content)
        return TestArtifactQuality(path=path, language=normalized_language)
    quality = TestArtifactQuality(path=path, language="python")
    try:
        tree = ast.parse(content or "")
    except SyntaxError as exc:
        quality.parse_ok = False
        quality.issues.append(
            TestQualityIssue(
                path=path,
                code="syntax_error",
                message=str(exc),
                line=int(getattr(exc, "lineno", 0) or 0),
                severity="error",
            )
        )
        return quality

    test_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]
    quality.test_function_count = len(test_functions)
    if not test_functions:
        quality.issues.append(
            TestQualityIssue(
                path=path,
                code="no_test_functions",
                message="No Python test functions were found.",
                severity="error",
            )
        )
        return quality

    focal_terms = _focal_terms(focal_module=focal_module, focal_symbols=focal_symbols)
    for test_func in test_functions:
        segment = ast.get_source_segment(content, test_func) or ""
        func_assertions = 0
        skipped = _is_skipped_python_test(test_func, segment)
        anchor = "Apex baseline selector anchor" in segment
        if skipped:
            quality.skipped_test_count += 1
        if anchor:
            quality.anchor_test_count += 1
        if _segment_references_any(segment, focal_terms):
            quality.focal_reference_count += 1
        for node in ast.walk(test_func):
            if isinstance(node, ast.Assert):
                quality.assertion_count += 1
                func_assertions += 1
                _inspect_assert_node(quality, node)
            elif isinstance(node, ast.Call):
                if _is_assertion_call(node):
                    quality.assertion_count += 1
                    func_assertions += 1
                    _inspect_assertion_call(quality, node)
                elif _is_pytest_raises_call(node):
                    quality.assertion_count += 1
                    func_assertions += 1
                    _inspect_pytest_raises_call(quality, node)
        if (
            func_assertions > 0
            and not skipped
            and not anchor
            and (not focal_terms or _segment_references_any(segment, focal_terms))
        ):
            quality.meaningful_test_count += 1

    if quality.assertion_count == 0:
        quality.issues.append(
            TestQualityIssue(
                path=path,
                code="no_assertions",
                message="Test functions contain no recognized assertions.",
                severity="error",
            )
        )
    if quality.test_function_count > 0 and quality.meaningful_test_count == 0:
        quality.issues.append(
            TestQualityIssue(
                path=path,
                code="no_meaningful_generated_tests",
                message="No non-skipped, non-anchor test with a recognized assertion was found.",
                severity="error",
            )
        )
    if focal_terms and quality.focal_reference_count == 0:
        quality.issues.append(
            TestQualityIssue(
                path=path,
                code="no_focal_references",
                message="Generated tests do not reference the focal module or public focal symbols.",
                severity="error",
            )
        )
    _inspect_mock_only_assertions(quality, content, language="python")
    return quality


def _analyze_js_ts_test_quality(
    *,
    path: str,
    content: str,
    language: str,
    focal_module: str = "",
    focal_symbols: list[str] | set[str] | tuple[str, ...] | None = None,
) -> TestArtifactQuality:
    quality = TestArtifactQuality(path=path, language=language)
    text = content or ""
    test_matches = list(
        re.finditer(
            r"\b(?:it|test)\s*(?:\.(?:only|skip))?\s*\("
            r"|\bo\s*\.\s*spec\s*\("
            r"|\bo\s*\(\s*['\"]",
            text,
        )
    )
    quality.test_function_count = len(test_matches)
    if quality.test_function_count == 0:
        quality.issues.append(
            TestQualityIssue(
                path=path,
                code="no_test_functions",
                message="No JavaScript/TypeScript test() or it() blocks were found.",
                severity="error",
            )
        )
        return quality

    assertion_patterns = [
        r"\bexpect\s*\(",
        r"\bassert(?:\.\w+)?\s*\(",
        r"\b(?:should|expect)\.",
        r"\b(?:toEqual|toStrictEqual|toBe|toContain|toThrow|toHaveProperty)\s*\(",
        r"\bo\s*\(.+?\)\s*\.\s*(?:equals|deepEquals|notEquals|throws|notThrows)\s*\(",
        r"\bverify\s*\(",
    ]
    quality.assertion_count = sum(len(re.findall(pattern, text)) for pattern in assertion_patterns)
    if quality.assertion_count == 0:
        quality.issues.append(
            TestQualityIssue(
                path=path,
                code="no_assertions",
                message="Test blocks contain no recognized JS/TS assertions.",
                severity="error",
            )
        )
    focal_terms = _focal_terms(focal_module=focal_module, focal_symbols=focal_symbols)
    if quality.assertion_count > 0:
        quality.meaningful_test_count = quality.test_function_count
    if focal_terms:
        quality.focal_reference_count = sum(
            1 for term in focal_terms if re.search(rf"\b{re.escape(term)}\b", text)
        )
        if quality.focal_reference_count == 0:
            quality.meaningful_test_count = 0
            quality.issues.append(
                TestQualityIssue(
                    path=path,
                    code="no_focal_references",
                    message="Generated tests do not reference the focal module or public focal symbols.",
                    severity="error",
                )
            )
    _inspect_mock_only_assertions(quality, text, language=language)

    weak_patterns = [
        (
            r"expect\s*\(\s*true\s*\)\s*\.\s*(?:toBe|toEqual)\s*\(\s*true\s*\)",
            "tautological_assertion",
        ),
        (r"assert(?:\.ok)?\s*\(\s*true\s*\)", "tautological_assertion"),
        (r"expect\s*\(.+?\)\s*\.\s*(?:toBeDefined|toBeTruthy)\s*\(", "weak_presence_assertion"),
        (
            r"expect\s*\(.+?\)\s*\.\s*not\s*\.\s*(?:toBeNull|toBeUndefined)\s*\(",
            "weak_non_null_assertion",
        ),
        (r"catch\s*\([^)]*\)\s*\{\s*\}", "broad_exception_oracle"),
    ]
    for pattern, code in weak_patterns:
        for match in re.finditer(pattern, text, flags=re.MULTILINE | re.DOTALL):
            quality.issues.append(
                TestQualityIssue(
                    path=path,
                    code=code,
                    message="Generated JS/TS test uses a weak oracle pattern.",
                    line=_line_number_for_offset(text, match.start()),
                    severity="error" if code == "tautological_assertion" else "warning",
                )
            )
    return quality


def _analyze_go_test_quality(*, path: str, content: str) -> TestArtifactQuality:
    quality = TestArtifactQuality(path=path, language="go")
    text = content or ""
    test_matches = list(
        re.finditer(r"\bfunc\s+Test[A-Za-z0-9_]*\s*\(\s*t\s+\*testing\.T\s*\)", text)
    )
    quality.test_function_count = len(test_matches)
    if quality.test_function_count == 0:
        quality.issues.append(
            TestQualityIssue(
                path=path,
                code="no_test_functions",
                message="No Go testing.T test functions were found.",
                severity="error",
            )
        )
        return quality

    assertion_patterns = [
        r"\bt\.(?:Error|Errorf|Fatal|Fatalf|Fail|FailNow)\s*\(",
        r"\b(?:assert|require)\.\w+\s*\(",
    ]
    quality.assertion_count = sum(len(re.findall(pattern, text)) for pattern in assertion_patterns)
    if quality.assertion_count == 0:
        quality.issues.append(
            TestQualityIssue(
                path=path,
                code="no_assertions",
                message="Go test functions contain no recognized failure assertions.",
                severity="error",
            )
        )
    elif quality.test_function_count > 0:
        quality.meaningful_test_count = quality.test_function_count
    _inspect_mock_only_assertions(quality, text, language="go")

    for match in re.finditer(r"if\s+true\s*\{[^}]*t\.(?:Fatal|Error)", text):
        quality.issues.append(
            TestQualityIssue(
                path=path,
                code="tautological_assertion",
                message="Go test failure branch is guarded by a constant true condition.",
                line=_line_number_for_offset(text, match.start()),
                severity="error",
            )
        )
    return quality


def _focal_terms(
    *,
    focal_module: str = "",
    focal_symbols: list[str] | set[str] | tuple[str, ...] | None = None,
) -> set[str]:
    terms = {str(item).strip() for item in (focal_symbols or []) if str(item).strip()}
    module = str(focal_module or "").strip()
    if module:
        terms.add(module)
        terms.add(module.rsplit(".", 1)[-1])
    return {term for term in terms if term}


def _segment_references_any(segment: str, terms: set[str]) -> bool:
    if not terms:
        return False
    return any(re.search(rf"\b{re.escape(term)}\b", segment or "") for term in terms)


def _is_skipped_python_test(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    segment: str,
) -> bool:
    if "pytest.skip(" in segment:
        return True
    for decorator in node.decorator_list:
        text = ast.unparse(decorator)
        if text.startswith("pytest.mark.skip") or text.startswith("unittest.skip"):
            return True
    return False


def _inspect_mock_only_assertions(
    quality: TestArtifactQuality,
    content: str,
    *,
    language: str,
) -> None:
    if quality.assertion_count <= 0:
        return
    text = content or ""
    if not _has_mock_interaction_assertion(text, language=language):
        return
    if _has_concrete_outcome_assertion(text, language=language):
        return
    quality.issues.append(
        TestQualityIssue(
            path=quality.path,
            code="mock_only_assertions",
            message=(
                "Test only asserts mock/interactions; pair it with a concrete "
                "observable outcome assertion."
            ),
            severity="error",
        )
    )


def _has_mock_interaction_assertion(text: str, *, language: str) -> bool:
    normalized_language = (language or "").lower()
    if normalized_language in {"python", "py", "python3"}:
        return bool(
            re.search(
                r"\bassert_(?:called|called_once|called_with|called_once_with|has_calls|not_called)\b",
                text,
            )
        )
    if normalized_language in {"javascript", "js", "jsx", "typescript", "ts", "tsx"}:
        return bool(
            re.search(
                r"\b(?:toHaveBeenCalled(?:With)?|toBeCalled(?:With)?|verify)\s*\(",
                text,
            )
        )
    if normalized_language in {"go", "golang"}:
        return bool(re.search(r"\bmock\.[A-Za-z0-9_]+\.Assert", text))
    return False


def _has_concrete_outcome_assertion(text: str, *, language: str) -> bool:
    normalized_language = (language or "").lower()
    if normalized_language in {"python", "py", "python3"}:
        return bool(
            re.search(r"\bassert\s+.+?(?:==|!=|<=|>=|<|>|\bin\b|\bis\b)", text)
            or re.search(
                r"\bassert(?:Equal|NotEqual|In|NotIn|Regex|Raises|Greater|Less|AlmostEqual|Is|IsNone)\s*\(",
                text,
            )
            or re.search(
                r"\bself\.assert(?:Equal|NotEqual|In|NotIn|Regex|Raises|Greater|Less|AlmostEqual|Is|IsNone)\s*\(",
                text,
            )
        )
    if normalized_language in {"javascript", "js", "jsx", "typescript", "ts", "tsx"}:
        return bool(
            re.search(
                r"\bexpect\s*\(.+?\)\s*\.\s*(?:not\s*\.\s*)?"
                r"(?:toBe|toEqual|toStrictEqual|toContain|toThrow|toHaveProperty|toMatch)\s*\(",
                text,
                flags=re.DOTALL,
            )
            or re.search(
                r"\bassert\.(?:equal|strictEqual|deepEqual|throws|match)\s*\(",
                text,
            )
            or re.search(
                r"\bo\s*\(.+?\)\s*\.\s*(?:equals|deepEquals|notEquals|throws|notThrows)\s*\(",
                text,
                flags=re.DOTALL,
            )
        )
    if normalized_language in {"go", "golang"}:
        return bool(
            re.search(
                r"\bif\s+.+?(?:==|!=|<=|>=|<|>)\s+.+?\{[^}]*\bt\.(?:Fatal|Error)", text, re.DOTALL
            )
            or re.search(r"\b(?:assert|require)\.\w+\s*\(", text)
        )
    return quality_has_non_mock_assertion_fallback(text)


def quality_has_non_mock_assertion_fallback(text: str) -> bool:
    return bool(re.search(r"\bassert|expect|require|should|throws|equals", text))


def _line_number_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, offset)) + 1


def _inspect_assert_node(quality: TestArtifactQuality, node: ast.Assert) -> None:
    expression = node.test
    if isinstance(expression, ast.Constant) and bool(expression.value) is True:
        quality.issues.append(
            TestQualityIssue(
                path=quality.path,
                code="tautological_assertion",
                message="Assertion is always true.",
                line=int(getattr(node, "lineno", 0) or 0),
                severity="error",
            )
        )
        return
    if isinstance(expression, ast.Compare):
        if _compare_is_self_comparison(expression):
            quality.issues.append(
                TestQualityIssue(
                    path=quality.path,
                    code="tautological_assertion",
                    message="Assertion compares an expression with itself.",
                    line=int(getattr(node, "lineno", 0) or 0),
                    severity="error",
                )
            )
        if _compare_is_non_null(expression):
            quality.issues.append(
                TestQualityIssue(
                    path=quality.path,
                    code="weak_non_null_assertion",
                    message="Non-null checks are weak unless paired with exact value or shape assertions.",
                    line=int(getattr(node, "lineno", 0) or 0),
                )
            )


def _inspect_assertion_call(quality: TestArtifactQuality, node: ast.Call) -> None:
    name = _call_name(node)
    args = list(node.args or [])
    if name.endswith(".assertEqual") and len(args) >= 2:
        if _ast_equivalent(args[0], args[1]):
            quality.issues.append(
                TestQualityIssue(
                    path=quality.path,
                    code="tautological_assertion",
                    message="assertEqual compares an expression with itself.",
                    line=int(getattr(node, "lineno", 0) or 0),
                    severity="error",
                )
            )
    if name.endswith(".assertTrue") and args:
        first = args[0]
        if isinstance(first, ast.Constant) and bool(first.value) is True:
            quality.issues.append(
                TestQualityIssue(
                    path=quality.path,
                    code="tautological_assertion",
                    message="assertTrue(True) is always true.",
                    line=int(getattr(node, "lineno", 0) or 0),
                    severity="error",
                )
            )
    if name.endswith(".assertIsNotNone"):
        quality.issues.append(
            TestQualityIssue(
                path=quality.path,
                code="weak_non_null_assertion",
                message="assertIsNotNone is weak unless paired with exact value or shape assertions.",
                line=int(getattr(node, "lineno", 0) or 0),
            )
        )


def _inspect_pytest_raises_call(
    quality: TestArtifactQuality,
    node: ast.Call,
) -> None:
    if not node.args:
        return
    exception_name = _expr_name(node.args[0])
    if exception_name in {"Exception", "BaseException"}:
        quality.issues.append(
            TestQualityIssue(
                path=quality.path,
                code="broad_exception_oracle",
                message="pytest.raises should assert a specific exception type.",
                line=int(getattr(node, "lineno", 0) or 0),
            )
        )


def _is_assertion_call(node: ast.Call) -> bool:
    name = _call_name(node)
    leaf = name.rsplit(".", 1)[-1]
    return leaf.startswith("assert") or leaf.startswith("assert_")


def _is_pytest_raises_call(node: ast.Call) -> bool:
    return _call_name(node).endswith("pytest.raises")


def _call_name(node: ast.Call) -> str:
    return _expr_name(node.func)


def _expr_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expr_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _compare_is_self_comparison(node: ast.Compare) -> bool:
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    if not isinstance(node.ops[0], (ast.Eq, ast.Is)):
        return False
    return _ast_equivalent(node.left, node.comparators[0])


def _compare_is_non_null(node: ast.Compare) -> bool:
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    right = node.comparators[0]
    return (
        isinstance(node.ops[0], ast.IsNot)
        and isinstance(right, ast.Constant)
        and right.value is None
    )


def _ast_equivalent(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(
        right,
        include_attributes=False,
    )
