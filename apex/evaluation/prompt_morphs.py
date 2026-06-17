"""V5 prompt morphs × masks (heterogeneous candidate generation).

When N agents in the ensemble all see the same prompt, candidate
diversity collapses to "just the agent's own decoding noise" — the
ceiling on cross-candidate voting drops. The fix used by every SOTA
testgen system is to vary the prompt deterministically across slots.

This module defines two orthogonal axes:

  * **Morph** — what goal the test should pursue:
      - ``verbatim``       : exact spec from the focal source
      - ``simplified``     : minimal happy-path coverage
      - ``exception``      : focus on documented error paths
      - ``boundary``       : focus on edge cases / error paths
  * **Mask** — what context the agent sees:
      - ``full_focal``     : entire focal module
      - ``localized``      : only the touched function + nearby helpers
      - ``signature``      : signature + docstring only
      - ``focal_plus_tests`` : focal source + project's existing test
                               files (highest-context mode)

The (morph, mask) pairs are assigned to the ensemble in a stable order
so that re-runs with the same N produce the same matrix, but every cell
is genuinely different: a ``simplified``×``signature`` agent generates
very different artifacts from a ``verbatim``×``full_focal`` agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MORPHS = ("verbatim", "simplified", "boundary", "exception")
MASKS = ("full_focal", "localized", "signature", "focal_plus_tests")


@dataclass(frozen=True)
class PromptVariant:
    """One (morph, mask) cell, plus the agent slot it's assigned to."""

    slot: int
    agent: str
    morph: str
    mask: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "agent": self.agent,
            "morph": self.morph,
            "mask": self.mask,
        }


def assign_variants(
    *,
    agents: list[str],
    morph_set: tuple[str, ...] = MORPHS,
    mask_set: tuple[str, ...] = MASKS,
) -> list[PromptVariant]:
    """Deterministically assign one (morph, mask) cell to each agent.

    With 4 agents and 4 morphs × 4 masks = 16 cells, we walk the
    diagonal first so the smallest ensembles still get maximum
    diversity, then fill remaining cells in a Latin-square-ish order.
    """

    if not agents:
        return []
    cells: list[PromptVariant] = []
    n_morphs = max(1, len(morph_set))
    n_masks = max(1, len(mask_set))
    for i, agent in enumerate(agents):
        morph = morph_set[i % n_morphs]
        mask = mask_set[(i // n_morphs + i) % n_masks]
        cells.append(PromptVariant(slot=i, agent=agent, morph=morph, mask=mask))
    return cells


def render_prompt(
    *,
    variant: PromptVariant,
    focal_path: str,
    focal_source: str,
    existing_test_source: str = "",
    signature_summary: str = "",
    artifact_path: str = "tests/test_apex.py",
    bug_description: str = "",
    typed_constraints_block: str = "",
    style: Any | None = None,
    test_runner: str = "",
    language: str = "python",
) -> str:
    """Render the prompt for ``variant`` using the appropriate mask + morph.

    ``typed_constraints_block`` is the rendered output of
    :func:`apex.evaluation.typed_assertion_constraints.build_typed_constraints`.
    When non-empty it is injected after the focal context so the agent
    sees the return-type advice before it writes assertions. Empty
    string ⇒ no insertion (the legacy contract).
    """

    prompt_language = str(language or getattr(style, "language", "") or "python")
    context_block = _render_mask_block(
        mask=variant.mask,
        focal_path=focal_path,
        focal_source=focal_source,
        existing_test_source=existing_test_source,
        signature_summary=signature_summary,
        language=prompt_language,
    )
    morph_instruction = _morph_instruction(variant.morph)
    runner = str(test_runner or getattr(style, "runner", "") or "").strip()
    framework_instruction = _framework_instruction(
        runner=runner,
        language=prompt_language,
        assertion_style=str(getattr(style, "assertion_style", "") or ""),
    )
    parts = [
        morph_instruction,
        f"Output a single {prompt_language} test file at path: {artifact_path}",
        framework_instruction,
        f"Do not include any prose — output ONLY the {prompt_language} source.",
        "",
        f"Focal file: {focal_path}",
        context_block,
    ]
    if typed_constraints_block:
        parts.extend(["", typed_constraints_block])
    if bug_description:
        parts.extend(["Bug description:", bug_description, ""])
    parts.append("Test file source:")
    return "\n".join(parts)


def _framework_instruction(
    *,
    runner: str,
    language: str,
    assertion_style: str,
) -> str:
    normalized_runner = (runner or "").lower()
    normalized_language = (language or "python").lower()
    if normalized_language in {"python", "py", "python3"}:
        if normalized_runner in {"unittest", "django", "django-runtests"}:
            return "Use unittest/Django TestCase style with self.assert* assertions. Do not import pytest."
        if normalized_runner == "sympy-bin-test":
            return "Use SymPy's existing test style and sympy.testing.pytest helpers when already imported. Do not import bare pytest."
        if normalized_runner == "pytest" or not normalized_runner:
            return "Use pytest style."
        if "self.assert" in assertion_style.lower():
            return "Use the existing unittest-style assertions. Do not import pytest unless existing tests already do."
        return f"Use the repository's {runner} test style."
    if normalized_language in {"javascript", "typescript"}:
        if normalized_runner in {"jest", "vitest"}:
            return f"Use {normalized_runner} test/expect style."
        if normalized_runner == "mocha":
            return "Use Mocha describe/it style and the repository's existing assertion helper."
        return "Use the repository's JavaScript/TypeScript test style."
    if normalized_language == "go":
        return "Use Go testing package style with TestXxx functions."
    if normalized_language == "java":
        return "Use the repository's JUnit test style."
    return "Use the repository's existing test runner style."


def _morph_instruction(morph: str) -> str:
    if morph == "verbatim":
        return (
            "Write thorough repository-native tests that exercise the focal module's "
            "documented behavior verbatim from its source. Tight assertions "
            "on exact values, not loose `is not None` checks."
        )
    if morph == "simplified":
        return (
            "Write the smallest set of repository-native tests that meaningfully exercises "
            "the happy path of the focal module. Prefer one assertion per test."
        )
    if morph == "inverted":
        return (
            "Write repository-native tests that PASS on the current focal source as-is, "
            "but would FAIL on a *correctly* patched version that fixes the bug "
            "described. Use the bug description to invert the oracle: assert the "
            "BUGGY output, then the dual-version verifier will detect mismatch "
            "against the correct output. Document the inversion intent in a "
            "module-level comment."
        )
    if morph == "boundary":
        return (
            "Write repository-native tests focused on edge cases of the focal module: "
            "empty inputs, boundary values, error paths, type-edge inputs. "
            "Skip the obvious happy path; another agent has it."
        )
    if morph == "exception":
        return (
            "Write repository-native tests focused on documented error paths and invalid "
            "inputs for the focal module. Prefer specific exception types and "
            "messages over broad `Exception` checks."
        )
    return "Write repository-native tests that thoroughly cover the focal module."


def _render_mask_block(
    *,
    mask: str,
    focal_path: str,
    focal_source: str,
    existing_test_source: str,
    signature_summary: str,
    language: str = "python",
) -> str:
    fence = language or "text"
    if mask == "signature":
        body = signature_summary or _strip_to_signature(focal_source)
        header = "Focal signatures (only):"
    elif mask == "localized":
        body = _localize_focal(focal_source)
        header = "Localized focal source (touched function + nearby helpers):"
    elif mask == "focal_plus_tests":
        body = (
            f"Focal source:\n```{fence}\n{focal_source}\n```\n\n"
            "Existing project test source (do not overwrite, learn the style):\n"
            f"```{fence}\n{existing_test_source}\n```"
        )
        return body
    else:
        body = focal_source
        header = "Focal source:"
    return f"{header}\n```{fence}\n{body}\n```"


def _strip_to_signature(source: str) -> str:
    """Reduce focal source to a list of `def`/`class` headers + first docline."""

    if not source:
        return ""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    src_lines = source.splitlines()
    headers: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        header_line = _read_header_line(src_lines, node)
        if not header_line:
            continue
        seg = header_line.rstrip()
        doc = ast.get_docstring(node) or ""
        if doc:
            indent = " " * (len(header_line) - len(header_line.lstrip()) + 4)
            seg = f'{seg}\n{indent}"""{doc.splitlines()[0][:120]}"""'
        else:
            seg = seg.rstrip(":") + ": ..."
        headers.append(seg)
    return "\n\n".join(headers) if headers else source


def _read_header_line(lines: list[str], node: Any) -> str:
    """Pull the def/class header (possibly multi-line) verbatim from source."""

    start = max(0, getattr(node, "lineno", 1) - 1)
    if start >= len(lines):
        return ""
    text = lines[start]
    while not text.rstrip().endswith(":") and start + 1 < len(lines):
        start += 1
        text += "\n" + lines[start]
    return text


def _localize_focal(source: str) -> str:
    """Best-effort: keep ``def``/``class`` blocks but trim long bodies.

    Without a known target function we just cap each function body at
    ~40 lines so the prompt stays compact. This is a heuristic; the
    repo_context module owns true localization when called by the
    pipeline.
    """

    if not source:
        return ""
    out_lines: list[str] = []
    in_block = False
    block_lines = 0
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("def ", "async def ", "class ")):
            in_block = True
            block_lines = 0
        if in_block:
            if block_lines < 40:
                out_lines.append(line)
            elif block_lines == 40:
                out_lines.append(" " * (len(line) - len(stripped)) + "# ... (truncated)")
            block_lines += 1
            if stripped == "" and block_lines > 1:
                in_block = False
        else:
            out_lines.append(line)
    return "\n".join(out_lines)
