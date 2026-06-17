"""Extract documented examples for test-generation prompts and seed tests."""

from __future__ import annotations

import ast
import doctest
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExtractedExample:
    expression: str
    expected_output: str
    source: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "expression": self.expression,
            "expected_output": self.expected_output,
            "source": self.source,
        }


def extract_examples_from_source(
    *,
    source: str,
    language: str = "python",
    path: str = "",
) -> list[ExtractedExample]:
    normalized = (language or "").lower()
    if normalized in {"python", "py", "python3"}:
        return _extract_python_doctest_examples(source=source, path=path)
    if normalized in {"javascript", "typescript", "js", "ts", "jsx", "tsx"}:
        return _extract_tagged_examples(source=source, marker="@example", path=path)
    if normalized in {"java"}:
        return _extract_tagged_examples(source=source, marker="@code", path=path)
    if normalized in {"rust", "rs"}:
        return _extract_rust_doc_examples(source=source, path=path)
    return []


def render_examples_prompt_block(examples: list[ExtractedExample]) -> str:
    if not examples:
        return ""
    lines = [
        "## Verified examples",
        "Use these documented examples as ground-truth oracles when writing tests.",
    ]
    for example in examples[:12]:
        expected = " ".join(example.expected_output.strip().split())
        expression = " ".join(example.expression.strip().split())
        if expression and expected:
            lines.append(f"- `{expression}` -> `{expected}`")
    return "\n".join(lines) if len(lines) > 2 else ""


def synthesize_python_doctest_seed_artifact(
    *,
    examples: list[ExtractedExample],
    focal_module: str,
    default_path: str,
) -> dict[str, Any] | None:
    """Create a conservative pytest seed artifact from simple doctest examples."""

    module = str(focal_module or "").strip()
    if not module or not examples:
        return None
    tests: list[str] = [f"from {module} import *  # noqa: F401,F403", ""]
    count = 0
    for example in examples:
        expression = str(example.expression or "").strip()
        expected_literal = _literal_expected_value(example.expected_output)
        if not expression or expected_literal is None:
            continue
        if not _looks_like_expression(expression):
            continue
        count += 1
        tests.extend(
            [
                f"def test_apex_doctest_example_{count}():",
                f"    assert {expression} == {expected_literal}",
                "",
            ]
        )
        if count >= 4:
            break
    if count == 0:
        return None
    return {"path": default_path, "content": "\n".join(tests).rstrip() + "\n"}


def _extract_python_doctest_examples(
    *,
    source: str,
    path: str,
) -> list[ExtractedExample]:
    parser = doctest.DocTestParser()
    examples: list[ExtractedExample] = []
    try:
        tree = ast.parse(source or "")
    except SyntaxError:
        return []

    docstrings: list[tuple[str, str]] = []
    module_doc = ast.get_docstring(tree)
    if module_doc:
        docstrings.append((path or "<module>", module_doc))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):
                continue
            doc = ast.get_docstring(node)
            if doc:
                docstrings.append((node.name, doc))

    seen: set[tuple[str, str]] = set()
    for owner, doc in docstrings:
        try:
            parsed = parser.get_doctest(doc, {}, owner, path or "<source>", 0)
        except ValueError:
            continue
        for example in parsed.examples:
            expression = example.source.strip()
            expected = example.want.strip()
            if not expression or not expected:
                continue
            key = (expression, expected)
            if key in seen:
                continue
            examples.append(
                ExtractedExample(
                    expression=expression,
                    expected_output=expected,
                    source=owner,
                )
            )
            seen.add(key)
            if len(examples) >= 24:
                return examples
    return examples


def _extract_tagged_examples(
    *,
    source: str,
    marker: str,
    path: str,
) -> list[ExtractedExample]:
    examples: list[ExtractedExample] = []
    pattern = re.compile(rf"{re.escape(marker)}(?P<body>.*?)(?:\*/|\n\s*\*)", re.S)
    for match in pattern.finditer(source or ""):
        body = "\n".join(
            line.strip(" *")
            for line in match.group("body").strip().splitlines()
            if line.strip(" *")
        )
        if body:
            examples.append(ExtractedExample(expression=body, expected_output="", source=path))
    return examples[:24]


def _extract_rust_doc_examples(*, source: str, path: str) -> list[ExtractedExample]:
    blocks = re.findall(r"(?m)^\s*///\s*(?P<line>.+)$", source or "")
    text = "\n".join(blocks)
    return _extract_tagged_examples(source=text, marker="```", path=path)


def _literal_expected_value(output: str) -> str | None:
    text = str(output or "").strip()
    if not text or "\n" in text:
        return None
    try:
        ast.literal_eval(text)
        return text
    except Exception:
        pass
    if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return text
    if text in {"True", "False", "None"}:
        return text
    return None


def _looks_like_expression(source: str) -> bool:
    if "\n" in source or ";" in source:
        return False
    try:
        parsed = ast.parse(source, mode="eval")
    except SyntaxError:
        return False
    return not isinstance(parsed.body, (ast.Lambda, ast.Yield, ast.YieldFrom))
