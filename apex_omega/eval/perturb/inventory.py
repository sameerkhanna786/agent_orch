"""libcst FQN classifier -> rename worklist (repo-defined symbols ONLY).

Runs in the BUILD venv (needs ``libcst``).  Uses
``libcst.metadata.FullyQualifiedNameProvider`` to partition every ``Name`` into
repo-defined vs builtin vs imported/external, and keeps ONLY def-sites whose FQN
is prefixed by the repo's top-level package(s).  This is the hard
"repo-defined-only" guarantee that auto-excludes stdlib/third-party/dunders and
prevents the gate-invisible over-rename blind spot.

We classify DEF-SITES (top-level functions, classes, methods, intra-repo
modules) on the REFERENCE (gold) tree only — the skeleton has ``pass`` bodies so
its bindings are absent/ambiguous.  Bare attributes/instance-vars are EXCLUDED
(renaming them risks silently rebinding same-spelled attrs on duck-typed/external
objects, which the gold-test gate cannot see).
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import libcst as cst
from libcst.metadata import (
    FullyQualifiedNameProvider,
    MetadataWrapper,
    PositionProvider,
    ScopeProvider,
)
from libcst.metadata.scope_provider import (
    BuiltinAssignment,
    ClassScope,
    FunctionScope,
    GlobalScope,
    ImportAssignment,
)

from . import namemap


@dataclass
class SymbolHit:
    """One repo-defined definition discovered in the reference tree."""

    fqn: str                 # fully-qualified name, e.g. "voluptuous.validators.Coerce"
    short_name: str          # the leaf name (what rope renames at the def offset)
    kind: str                # namemap.SymbolKind.*
    module: str              # the module FQN it is defined in
    file: str                # absolute path of the defining file
    line: int                # 1-based line of the def/class name
    col: int                 # 0-based column of the NAME (rope offset anchor)


@dataclass
class Inventory:
    """The full reference-tree symbol inventory."""

    top_packages: tuple[str, ...]
    symbols: list[SymbolHit] = field(default_factory=list)
    # every repo-defined FQN seen (for the namemap reserved set / collision guard)
    all_fqns: set[str] = field(default_factory=set)
    modules: set[str] = field(default_factory=set)   # intra-repo module FQNs
    # short names that appear inside string literals (unsound to rename)
    string_literal_names: set[str] = field(default_factory=set)

    def def_worklist(
        self,
        *,
        kinds: tuple[str, ...] = ("function", "class"),
        exclude_module_prefixes: tuple[str, ...] = (),
        exclude_string_literal_names: bool = True,
        scope_prefix: Optional[str] = None,
    ) -> list[tuple[str, str]]:
        """``[(fqn, kind), ...]`` for the namemap (de-duplicated, canonical).

        Pilot soundness policy defaults:

        * ``kinds=("function","class")`` — methods/properties are EXCLUDED
          (accessed via runtime-typed values in tests that rope cannot statically
          rewrite — the gate-invisible blind spot the spec flags).
        * ``exclude_string_literal_names`` — drop any symbol whose short name
          appears in a string literal (repr/__name__ assertions, leaked names).
        * ``exclude_module_prefixes`` — drop symbols under these module prefixes
          (e.g. the tests package, to keep node ids comparable to vanilla).
        * ``scope_prefix`` — when set, rename ONLY symbols whose FQN is under this
          module prefix (module-scoped perturbation, e.g. ``networkx.algorithms.dag``);
          everything else (incl. ``_dispatchable``/backends/``__init__``) is left
          untouched.  The inventory still SCANS the whole tree so rope rewrites all
          cross-file uses of the scoped symbols.
        """
        seen: dict[str, str] = {}
        for s in self.symbols:
            if s.kind not in kinds:
                continue
            if scope_prefix and not (s.fqn == scope_prefix or s.fqn.startswith(scope_prefix + ".")):
                continue
            if any(s.module == p or s.module.startswith(p + ".") for p in exclude_module_prefixes):
                continue
            if any(s.fqn == p or s.fqn.startswith(p + ".") for p in exclude_module_prefixes):
                continue
            if exclude_string_literal_names and s.short_name in self.string_literal_names:
                continue
            seen.setdefault(s.fqn, s.kind)
        return list(seen.items())

    def module_worklist(
        self, *, exclude_prefixes: tuple[str, ...] = (), scope_prefix: Optional[str] = None
    ) -> list[str]:
        """Intra-repo module FQNs eligible for rename (excludes top package roots
        and any under *exclude_prefixes*).  ``scope_prefix`` restricts to modules
        under one subtree (module-scoped perturbation)."""
        out = []
        for m in sorted(self.modules):
            if m in self.top_packages:
                continue  # never rename the top-level importable package
            if scope_prefix and not (m == scope_prefix or m.startswith(scope_prefix + ".")):
                continue
            if any(m == p or m.startswith(p + ".") for p in exclude_prefixes):
                continue
            if m in self.string_literal_names or m.rsplit(".", 1)[-1] in self.string_literal_names:
                continue  # a module referenced by string (import_module) is unsound
            out.append(m)
        return out


# Coupled string-derived symbol families to EXCLUDE as a unit (renaming one
# breaks an implicit dispatch contract the gold tests may not exercise per-key).
_DYNAMIC_DISPATCH_PREFIXES = (
    "visit_",      # libcst/ast visitors, jinja node visitors
    "compile_",    # babel plural compilers
    "pytest_",     # pytest hooks
    "test_",       # test functions — keep node ids comparable to vanilla
)


def _discover_py_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            files.append(root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # skip caches / venvs that might live under a checkout
            dirnames[:] = [
                d for d in dirnames
                if d not in {"__pycache__", ".git", ".tox", ".venv", "build", "dist", ".eggs"}
                and not d.endswith(".egg-info")
            ]
            for fn in filenames:
                if fn.endswith(".py"):
                    files.append(Path(dirpath) / fn)
    return sorted(set(files))


def _module_fqn_for(path: Path, repo_root: Path) -> Optional[str]:
    """Derive the dotted module FQN for *path* relative to *repo_root*.

    Walks up only while an ``__init__.py`` is present so we get the importable
    package path (mirrors how FullyQualifiedNameProvider computes module names).
    """
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return None
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def _is_excluded_name(name: str) -> bool:
    if name.startswith("__") and name.endswith("__"):
        return True  # dunders
    if any(name.startswith(p) for p in _DYNAMIC_DISPATCH_PREFIXES):
        return True
    if name in namemap._PY_KEYWORDS or name in namemap._BUILTIN_NAMES:
        return True
    return False


class _DefCollector(cst.CSTVisitor):
    """Collect top-level function/class defs and methods with their FQN + offset."""

    METADATA_DEPENDENCIES = (
        FullyQualifiedNameProvider,
        ScopeProvider,
        PositionProvider,
    )

    def __init__(self, top_packages: tuple[str, ...], file_path: Path, module_fqn: Optional[str]):
        super().__init__()
        self.top_packages = top_packages
        self.file_path = file_path
        self.module_fqn = module_fqn or ""
        self.hits: list[SymbolHit] = []
        self.all_fqns: set[str] = set()

    def _fqn_of(self, node: cst.CSTNode) -> Optional[str]:
        try:
            quals = self.get_metadata(FullyQualifiedNameProvider, node)
        except KeyError:
            return None
        for q in quals or ():
            return q.name
        return None

    def _repo_prefixed(self, fqn: Optional[str]) -> bool:
        if not fqn:
            return False
        return any(fqn == p or fqn.startswith(p + ".") for p in self.top_packages)

    def _record(self, name_node: cst.Name, kind: str, owner_node: cst.CSTNode) -> None:
        name = name_node.value
        fqn = self._fqn_of(name_node) or self._fqn_of(owner_node)
        if not self._repo_prefixed(fqn):
            return
        self.all_fqns.add(fqn)  # type: ignore[arg-type]
        if _is_excluded_name(name):
            return
        # Scope guard: drop builtin/imported bindings (defensive — defs are
        # Assignments, but this rejects an edge case where a name re-binds an import).
        try:
            scope = self.get_metadata(ScopeProvider, name_node)
            for assignment in (scope[name] if scope and name in scope else ()):
                if isinstance(assignment, (BuiltinAssignment, ImportAssignment)):
                    return
        except (KeyError, Exception):
            pass
        try:
            pos = self.get_metadata(PositionProvider, name_node)
            line, col = pos.start.line, pos.start.column
        except KeyError:
            line, col = 0, 0
        self.hits.append(
            SymbolHit(
                fqn=fqn,  # type: ignore[arg-type]
                short_name=name,
                kind=kind,
                module=self.module_fqn,
                file=str(self.file_path),
                line=line,
                col=col,
            )
        )

    def _enclosing_is_class(self, node: cst.FunctionDef) -> bool:
        try:
            scope = self.get_metadata(ScopeProvider, node.name)
            return isinstance(scope, ClassScope)
        except (KeyError, Exception):
            return False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        kind = namemap.SymbolKind.METHOD if self._enclosing_is_class(node) else namemap.SymbolKind.FUNCTION
        self._record(node.name, kind, node)

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self._record(node.name, namemap.SymbolKind.CLASS, node)


def build_inventory(
    repo_root: Path,
    target_roots: list[Path],
    top_packages: tuple[str, ...],
) -> Inventory:
    """Classify all repo-defined def-sites under *target_roots*.

    Args:
        repo_root: the variant checkout root (rope ``Project`` root).
        target_roots: dirs/files to scan ({src_dir} ∪ {test_dir}); absolute.
        top_packages: the repo's top-level importable package name(s), e.g.
            ``("voluptuous",)`` or ``("networkx",)``.
    """
    repo_root = repo_root.resolve()
    files = _discover_py_files([p.resolve() for p in target_roots])
    inv = Inventory(top_packages=tuple(top_packages))
    # FullyQualifiedNameProvider needs the per-file module-name cache so module
    # FQNs resolve against the package layout (libcst 1.8 API: gen_cache returns
    # {path_str: ModuleNameAndPackage}, passed back as the provider's cache).
    fqn_cache = FullyQualifiedNameProvider.gen_cache(
        repo_root, [str(f) for f in files]
    )
    for f in files:
        module_fqn = _module_fqn_for(f, repo_root)
        if module_fqn:
            inv.modules.add(module_fqn)
        try:
            src = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        per_file = fqn_cache.get(str(f))
        if per_file is None:
            continue
        try:
            module = cst.parse_module(src)
            wrapper = MetadataWrapper(
                module,
                cache={FullyQualifiedNameProvider: per_file},
            )
            collector = _DefCollector(tuple(top_packages), f, module_fqn)
            wrapper.visit(collector)
        except Exception:
            # A file that fails to parse/classify is skipped, not fatal — the
            # closing structural check + gate catch any resulting unsoundness.
            continue
        inv.symbols.extend(collector.hits)
        inv.all_fqns |= collector.all_fqns
    # only intra-repo modules under target roots are rename candidates
    inv.modules = {m for m in inv.modules if any(
        m == p or m.startswith(p + ".") for p in top_packages
    )}
    # String-literal leak guard: a symbol whose SHORT NAME appears (as a whole
    # word) inside a NON-DOCSTRING code string literal is UNSOUND to rename — the
    # literal would still carry the old name (e.g. a runtime ``repr``/``__name__``
    # assertion in a test, or a dispatch key), and rope never rewrites string
    # CONTENTS, so the gate would fail.  Record these so the worklist excludes them.
    #
    # DOCSTRINGS and COMMENTS are DELIBERATELY EXCLUDED from this scan: rope's
    # ``docs=True`` DOES rewrite symbol references inside docstrings/comments/doctests
    # (verified), so docstring prose is retained AND references stay consistent.
    # Scanning docstrings here would over-exclude common class names that appear as
    # ordinary English words in prose (e.g. "All", "Any", "Schema"), needlessly
    # weakening the rename.
    inv.string_literal_names = _collect_code_string_literal_names(files)
    return inv


def _collect_code_string_literal_names(files: list[Path]) -> set[str]:
    """Identifier-like whole-word tokens inside NON-DOCSTRING string literals.

    Uses the AST to skip the module/class/function leading docstring expressions
    (those are handled by rope's ``docs=True``); only genuine code string literals
    (assignment values, call args, f-strings, ``repr`` targets) are scanned.
    """
    import re as _re

    word_re = _re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    tokens: set[str] = set()
    for f in files:
        try:
            src = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        # collect the id() of every docstring-string-node to skip it
        docstring_nodes: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = getattr(node, "body", None) or []
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(getattr(body[0], "value", None), ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    docstring_nodes.add(id(body[0].value))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) in docstring_nodes:
                    continue  # leading docstring — rope handles its references
                tokens.update(word_re.findall(node.value))
    return tokens
