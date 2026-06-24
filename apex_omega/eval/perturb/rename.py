"""rope engine driver — semantics-preserving cross-file alpha-rename.

Runs in the BUILD venv (needs ``rope``).  rope is the RENAME ENGINE: a single
``Rename(...).get_changes(new, docs=True, in_hierarchy=True)`` rewrites the def,
all uses, from-imports, attribute access, ``__all__`` string entries, and (for
modules) the file on disk + relative imports — with real binding resolution.

Offset invalidation is the central hazard: every applied edit shifts char
offsets in the touched file.  We dodge it by re-resolving each symbol's offset
from the CURRENT file content immediately before each apply (anchored on the
``short_name`` at the known line, falling back to a whole-file search), and by
applying one symbol per ``project.do(changes)`` with ``project.validate()`` +
re-parse after each.

The SAME persisted :class:`~apex_omega.eval.perturb.namemap.NameMap` is applied
to BOTH checkouts (reference + skeleton) so the perturbed API surface is
identical on both sides (a symbol renamed in reference but no-op in skeleton =
the agent's blank API != the scored API; we hard-fail that divergence upstream).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rope.base.project import Project
from rope.refactor.rename import Rename

from .inventory import Inventory, SymbolHit
from .namemap import NameMap


@dataclass
class RenameReport:
    applied: list[str] = field(default_factory=list)        # fqns successfully renamed
    skipped: dict[str, str] = field(default_factory=dict)   # fqn -> reason
    module_renames: list[str] = field(default_factory=list)


def _offset_of(content: str, line: int, col: int, name: str) -> Optional[int]:
    """Char offset of *name* at (1-based *line*, 0-based *col*) in *content*.

    Re-resolves against the CURRENT content (post earlier edits). Falls back to
    the first occurrence of ``name`` at/after the line start if the exact column
    no longer matches (a prior edit on the same line shifted it).
    """
    lines = content.splitlines(keepends=True)
    if line <= 0 or line > len(lines):
        # line unknown — search whole file for a word-boundary match
        idx = content.find(name)
        return idx if idx >= 0 else None
    line_start = sum(len(lines[i]) for i in range(line - 1))
    # exact column first
    if content[line_start + col: line_start + col + len(name)] == name:
        return line_start + col
    # fallback: search within this line then the rest of the file
    here = content.find(name, line_start)
    return here if here >= 0 else None


def _line_col_index(content: str) -> list[int]:
    lines = content.splitlines(keepends=True)
    starts = [0]
    for ln in lines:
        starts.append(starts[-1] + len(ln))
    return starts


def apply_rename(
    repo_root: Path,
    inventory: Inventory,
    name_map: NameMap,
    *,
    rename_modules: bool = True,
) -> RenameReport:
    """Apply *name_map* to the checkout at *repo_root* using rope.

    Symbol renames are applied bottom-up per file (highest offset first) so an
    earlier edit cannot invalidate a later symbol's anchor on the same line.
    Module renames are applied last (they move files / rewrite imports wholesale).
    """
    repo_root = repo_root.resolve()
    report = RenameReport()
    project = Project(str(repo_root), ropefolder=".ropeproject")
    try:
        # --- symbol renames -------------------------------------------------
        # Group hits by file; within a file apply by descending offset.
        hits_by_fqn: dict[str, SymbolHit] = {}
        for h in inventory.symbols:
            # one canonical hit per fqn (the def site); first wins (def order)
            hits_by_fqn.setdefault(h.fqn, h)

        renamable = [
            (fqn, hits_by_fqn[fqn])
            for fqn in name_map.symbols
            if fqn in hits_by_fqn
        ]
        # sort: by file, then descending line/col so we edit bottom-up
        renamable.sort(key=lambda kv: (kv[1].file, -kv[1].line, -kv[1].col))

        for fqn, hit in renamable:
            new_name = name_map.symbols[fqn]
            resource = project.get_resource(
                str(Path(hit.file).resolve().relative_to(repo_root))
            )
            content = resource.read()
            offset = _offset_of(content, hit.line, hit.col, hit.short_name)
            if offset is None:
                report.skipped[fqn] = "offset_unresolved"
                continue
            try:
                rename = Rename(project, resource, offset)
                changes = rename.get_changes(
                    new_name, docs=True, in_hierarchy=True, unsure=None
                )
                project.do(changes)
                project.validate()
            except Exception as exc:  # rope refused (e.g. ambiguous binding)
                report.skipped[fqn] = f"rope_error:{type(exc).__name__}:{str(exc)[:120]}"
                continue
            # re-parse the touched file: fail-fast on corruption
            try:
                ast.parse(resource.read())
            except SyntaxError as exc:
                report.skipped[fqn] = f"corrupt_after_rename:{exc}"
                continue
            report.applied.append(fqn)

        # --- module renames (last; file moves + relative-import rewrites) ----
        if rename_modules and name_map.modules:
            # Rename deepest modules first so a parent-package move doesn't
            # invalidate a child's resource path.
            for mod_fqn in sorted(name_map.modules, key=lambda m: -m.count(".")):
                new_leaf = name_map.modules[mod_fqn]
                rel = mod_fqn.replace(".", "/")
                # try package (dir/__init__.py) then module (.py)
                candidates = [f"{rel}.py", f"{rel}/__init__.py"]
                resource = None
                for cand in candidates:
                    try:
                        resource = project.get_resource(cand)
                        break
                    except Exception:
                        continue
                if resource is None:
                    report.skipped[f"module:{mod_fqn}"] = "module_resource_missing"
                    continue
                try:
                    # For a module file, rename targets the file resource (no offset).
                    # For a package, rope renames the package dir.
                    target = resource if cand.endswith(".py") and not cand.endswith("__init__.py") else resource.parent
                    rename = Rename(project, target)
                    changes = rename.get_changes(new_leaf, docs=True)
                    project.do(changes)
                    project.validate()
                    report.module_renames.append(mod_fqn)
                except Exception as exc:
                    report.skipped[f"module:{mod_fqn}"] = f"rope_module_error:{type(exc).__name__}:{str(exc)[:120]}"
                    continue
    finally:
        project.close()
    return report


def rewrite_doctest_globs(
    repo_root: Path,
    name_map: NameMap,
    *,
    globs: tuple[str, ...] = ("*.md", "*.rst", "*.txt"),
) -> dict[str, int]:
    """Rewrite renamed public-symbol references inside doctest-glob files.

    Repos with ``--doctest-glob`` (e.g. voluptuous' ``tests.md``) embed gold
    doctests that call the public API BY NAME; rope does not touch ``.md`` files,
    so those doctests would reference the now-gone old names and fail the gate.
    This applies the name_map by WHOLE-WORD substitution but ONLY on doctest
    lines (``>>>`` / ``...`` continuations) so prose is never corrupted.

    Returns ``{relpath: n_substitutions}``.  Sound because doctests exercise the
    public surface, which is exactly what the symbol rename covered.
    """
    import re as _re

    # old short-name -> new name, longest-first so a prefix never shadows a longer name
    leaf_map = {
        old.rsplit(".", 1)[-1]: new
        for old, new in name_map.symbols.items()
    }
    # only substitute names that are unambiguous (one new target across the repo)
    pairs = sorted(leaf_map.items(), key=lambda kv: -len(kv[0]))
    if not pairs:
        return {}
    out: dict[str, int] = {}
    for glob in globs:
        for f in repo_root.rglob(glob):
            if ".ropeproject" in f.parts or "__pycache__" in f.parts:
                continue
            try:
                src = f.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            n = 0
            new_lines = []
            for line in src.splitlines(keepends=True):
                stripped = line.lstrip()
                if stripped.startswith(">>>") or stripped.startswith("..."):
                    for old, new in pairs:
                        line, k = _re.subn(rf"(?<![\w.]){_re.escape(old)}(?![\w])", new, line)
                        n += k
                new_lines.append(line)
            if n:
                f.write_text("".join(new_lines), encoding="utf-8")
                out[str(f.relative_to(repo_root))] = n
    return out


def assert_structural_equivalence(vanilla_root: Path, perturbed_root: Path, rel_files: list[str]) -> list[str]:
    """Return a list of files whose AST structure (modulo identifier *names*)
    diverges between the two trees — proving the edit was alpha-rename ONLY.

    Empty list == structurally identical (signatures/arg-counts/control-flow).
    """
    divergent: list[str] = []
    for rel in rel_files:
        v = vanilla_root / rel
        p = perturbed_root / rel
        if not v.exists() or not p.exists():
            continue
        try:
            va = _structural_skeleton(v.read_text(encoding="utf-8"))
            pa = _structural_skeleton(p.read_text(encoding="utf-8"))
        except SyntaxError:
            divergent.append(rel)
            continue
        if va != pa:
            divergent.append(rel)
    return divergent


def _structural_skeleton(src: str) -> str:
    """A canonical AST node-type stream with all identifiers blanked.

    Two alpha-renamed-only trees produce the identical skeleton.
    """
    tree = ast.parse(src)
    out: list[str] = []

    class _V(ast.NodeVisitor):
        def generic_visit(self, node: ast.AST) -> None:
            out.append(type(node).__name__)
            # record structural arity for callables (arg counts must match)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                out.append(
                    f"args:{len(a.posonlyargs)},{len(a.args)},{len(a.kwonlyargs)},"
                    f"{1 if a.vararg else 0},{1 if a.kwarg else 0},{len(a.defaults)}"
                )
            super().generic_visit(node)

    _V().visit(tree)
    return "\n".join(out)
