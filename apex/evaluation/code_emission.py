"""Language-aware code-emission normalization for generated tests."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Protocol


class LanguageEmitter(Protocol):
    def roundtrip(self, source: str) -> str: ...


@dataclass(frozen=True)
class PythonEmitter:
    """Parse and re-emit Python code through the stdlib AST."""

    def roundtrip(self, source: str) -> str:
        return roundtrip_python(source)


_EMITTERS: dict[str, LanguageEmitter] = {
    "python": PythonEmitter(),
    "py": PythonEmitter(),
    "python3": PythonEmitter(),
}


def roundtrip_python(source: str) -> str:
    """Return Python source normalized through ``ast.parse``/``ast.unparse``.

    The function intentionally raises ``SyntaxError`` instead of returning a
    best-effort string. Callers use that hard failure as the parse gate.
    """

    text = str(source or "").replace("\r\n", "\n").strip()
    tree = ast.parse(text)
    rendered = ast.unparse(tree).strip()
    ast.parse(rendered)
    return rendered + ("\n" if rendered else "")


def roundtrip(source: str, *, language: str = "python") -> str:
    emitter = _EMITTERS.get((language or "").lower())
    if emitter is None:
        return str(source or "")
    return emitter.roundtrip(source)


def build_parse_repair_prompt(
    source: str,
    error: SyntaxError,
    *,
    language: str = "python",
) -> str:
    """Render a fixed-layout repair prompt for syntax-only repairs."""

    line = getattr(error, "lineno", None)
    offset = getattr(error, "offset", None)
    message = getattr(error, "msg", str(error))
    location = f"line {line}" if line else "unknown line"
    if offset:
        location += f", column {offset}"
    return "\n".join(
        [
            "Repair this generated test file so it parses.",
            "Only correct the syntax at the indicated location; do not add new tests.",
            f"Language: {language or 'unknown'}",
            f"SyntaxError: {message} ({location})",
            "",
            "Current file:",
            "```" + (language or ""),
            str(source or ""),
            "```",
        ]
    )
