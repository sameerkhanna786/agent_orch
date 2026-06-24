"""libcst transformer — neutralize docstrings + strip comments (de-memo).

Runs in the BUILD venv (needs ``libcst``).  A lossless CST transform that blanks
module/function/class docstrings and removes comments, so the perturbed surface
leaks no natural-language hint that could let the model recall the original API.

GUARD: callers MUST skip neutralization for repos whose ``test_cmd``/conftest
enables ``--doctest-modules``/``--doctest-glob`` (it would delete the doctests
the gold suite scores).  The CLI passes ``--neutralize-docs`` only when that
guard is clear.
"""

from __future__ import annotations

from pathlib import Path

import libcst as cst


_PLACEHOLDER = '""'  # blanked docstring; keeps a string-expr statement valid


class _Neutralizer(cst.CSTTransformer):
    def __init__(self) -> None:
        super().__init__()

    def _blank_docstring(self, body):
        """Replace a leading docstring SimpleStatementLine with an empty string."""
        if not body or not isinstance(body[0], cst.SimpleStatementLine):
            return body
        stmt = body[0]
        if (
            len(stmt.body) == 1
            and isinstance(stmt.body[0], cst.Expr)
            and isinstance(stmt.body[0].value, (cst.SimpleString, cst.ConcatenatedString))
        ):
            new_expr = stmt.body[0].with_changes(
                value=cst.SimpleString(value=_PLACEHOLDER)
            )
            new_stmt = stmt.with_changes(body=[new_expr])
            return [new_stmt, *body[1:]]
        return body

    def leave_Module(self, original, updated):
        new_body = self._blank_docstring(list(updated.body))
        return updated.with_changes(body=new_body)

    def leave_FunctionDef(self, original, updated):
        if isinstance(updated.body, cst.IndentedBlock):
            new_inner = self._blank_docstring(list(updated.body.body))
            return updated.with_changes(body=updated.body.with_changes(body=new_inner))
        return updated

    def leave_ClassDef(self, original, updated):
        if isinstance(updated.body, cst.IndentedBlock):
            new_inner = self._blank_docstring(list(updated.body.body))
            return updated.with_changes(body=updated.body.with_changes(body=new_inner))
        return updated

    def leave_Comment(self, original, updated):
        # Strip comment text but keep an empty comment token only if removal would
        # break formatting; libcst lets us drop it by emptying the value safely.
        return updated.with_changes(value="#")


def neutralize_file(path: Path) -> bool:
    """Neutralize docstrings/comments in *path* in place.  Returns True if changed."""
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    try:
        module = cst.parse_module(src)
    except Exception:
        return False
    new_module = module.visit(_Neutralizer())
    if new_module.code != src:
        path.write_text(new_module.code, encoding="utf-8")
        return True
    return False


def neutralize_tree(roots: list[Path]) -> int:
    """Neutralize every .py under *roots*.  Returns count of files changed."""
    changed = 0
    seen: set[Path] = set()
    for root in roots:
        files = [root] if root.is_file() else root.rglob("*.py")
        for f in files:
            f = f.resolve()
            if f in seen or "__pycache__" in f.parts or ".ropeproject" in f.parts:
                continue
            seen.add(f)
            if neutralize_file(f):
                changed += 1
    return changed
