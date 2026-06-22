"""Zero-token AST import-graph pre-pass (diagnose STAGE 1; O2/O3/O4 redesign).

Before any agent runs, statically answer "what stops this repo's gold suite from even COLLECTING?".
On a commit0 base the implementation is stripped, so a test/conftest top-level import of a name that
does not yet exist (the pydantic ``GenerateSchema`` collection-collapse class) makes pytest error at
COLLECTION — nothing runs until a large fraction is implemented, and the failing-test north star reads
0/0 the whole time. A pure-Python AST walk catches this tokenlessly:

  * resolve the test-bootstrap imports (conftest.py + the test files + their __init__ chain) against
    the repo's OWN source tree, classifying each as internal-resolved / internal-UNRESOLVED (missing
    module OR missing top-level symbol = the real "implement me first" closure) / external,
  * compute ``import_depth`` (deepest dotted internal module reached before the first unresolved hop —
    the SPFG++ rising signal on a collection-collapse repo),
  * parse pytest ``addopts`` and flag plugin options whose plugin is not importable in this env
    (the ``--memray`` class),
  * emit ``collects_cleanly`` (best-effort: no unresolved internal import found in the bootstrap set).

This NEVER imports repo or third-party code (no side effects, no env mutation) and NEVER raises — a
parse failure on one file degrades that file to "unknown", not the whole pre-pass. Output is a plain
dict merged into the repo map under ``diagnosis`` (gated by APEX_OMEGA_DIAG); the STAGE-2 scouts
fact-check their classification against ``unresolved_internal`` / ``import_depth`` from here.
"""
from __future__ import annotations

import ast
import configparser
import importlib.util
from pathlib import Path
from typing import Optional

_SKIP_SEGS = (".git/", ".venv/", "venv/", "site-packages/", "__pycache__/", ".tox/", "build/", "dist/")
# pytest addopts that REQUIRE a third-party plugin (option -> distribution/import module). Mirrors the
# scoring-side strip map; here we only SURFACE the risk so the orchestrator can plan around it.
_PLUGIN_OPTION_MODULES = {
    "--memray": "pytest_memray",
    "--benchmark": "pytest_benchmark",
    "--cov": "pytest_cov",
    "--mypy": "pytest_mypy",
    "--flake8": "pytest_flake8",
    "--black": "pytest_black",
    "--snapshot-update": "syrupy",
    "--asyncio-mode": "pytest_asyncio",
}


# Top-level dirs that are SOURCE ROOTS (not packages) under which the real package lives — stripped
# from the dotted module key so `src/pkg/m.py` indexes as `pkg.m` (src-layout, edge f).
_SRC_ROOTS = ("src",)
_IMPORT_ERROR_NAMES = ("ImportError", "ModuleNotFoundError")


def _rel_ok(rel: str) -> bool:
    return not any(seg in rel for seg in _SKIP_SEGS)


def _module_top_level_names(path: Path) -> Optional[set]:
    """Top-level def/class/assign/import names a module exports (None if it can't be parsed)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    names: set = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".", 1)[0])
            # a `from x import *` re-exports an unknowable set -> treat as wildcard
            if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
                names.add("*")
    return names


class _RepoIndex:
    """Map a dotted module path to its file within the repo (package dirs + plain modules)."""

    def __init__(self, root: Path):
        self.root = root
        self.by_dotted: dict[str, Path] = {}
        # src-layout (edge f): a package under src/ is imported as `pkg`, NOT `src.pkg`. Treat a
        # top-level source-root dir with no __init__.py as a layout root to strip from dotted keys
        # (a legitimate `src` PACKAGE would have src/__init__.py and is left alone).
        self.strip_roots: set = set()
        for r in _SRC_ROOTS:
            d = root / r
            if d.is_dir() and not (d / "__init__.py").exists():
                self.strip_roots.add(r)
        for p in root.rglob("*.py"):
            rel = p.relative_to(root).as_posix()
            if not _rel_ok(rel):
                continue
            parts = rel[:-3].split("/")          # strip .py
            if len(parts) > 1 and parts[0] in self.strip_roots:
                parts = parts[1:]                # src/pkg/mod.py -> pkg.mod
            if parts[-1] == "__init__":
                parts = parts[:-1]
                if not parts:
                    continue
            self.by_dotted[".".join(parts)] = p

    def resolve(self, dotted: str) -> Optional[Path]:
        """The file for a dotted module, or its package __init__, if it lives in the repo.
        Recognizes PEP-420 namespace packages (a dir with indexed submodules but no __init__.py)
        as resolvable so `from nspkg import sub` is not a spurious missing-module wall (edge g2)."""
        if dotted in self.by_dotted:
            return self.by_dotted[dotted]
        parts = dotted.split(".")
        pkg_init = self.root / Path(*parts) / "__init__.py"
        if pkg_init.exists() and _rel_ok(pkg_init.relative_to(self.root).as_posix()):
            return pkg_init
        prefix = dotted + "."
        if any(k == dotted or k.startswith(prefix) for k in self.by_dotted):
            # namespace package: referenced by indexed submodules though it has no __init__.py.
            # Return its dir path (truthy = resolvable); _module_top_level_names() yields None on a
            # dir, so the symbol check is conservatively skipped (never a false missing_symbol).
            return self.root / Path(*parts)
        return None

    def top_module(self, dotted: str) -> str:
        return dotted.split(".", 1)[0]

    @property
    def top_packages(self) -> set:
        return {d.split(".", 1)[0] for d in self.by_dotted}


def _try_guards_import(node: ast.Try) -> bool:
    """True iff a ``try`` block catches ImportError/ModuleNotFoundError — a guarded OPTIONAL import
    whose whole purpose is to tolerate absence, so its body must NOT count toward a collection wall."""
    for h in node.handlers:
        t = h.type
        if isinstance(t, ast.Name) and t.id in _IMPORT_ERROR_NAMES:
            return True
        if isinstance(t, ast.Tuple) and any(
                isinstance(e, ast.Name) and e.id in _IMPORT_ERROR_NAMES for e in t.elts):
            return True
    return False


def _collection_imports(tree: ast.AST, pkg_parts: list):
    """Yield (abs_module_dotted, name_or_None, depth) for imports that execute at MODULE-COLLECTION
    time only. Covers two correctness edges beyond a flat ast.walk:

      * (e) RELATIVE imports (``from . import x`` / ``from ..y import z``) are resolved against the
        importer's package ``pkg_parts`` to an ABSOLUTE dotted module and classified like any internal
        import — instead of being skipped (which silently missed a relative conftest wall).
      * (i) Only MODULE-TOP-LEVEL imports count: we descend into top-level if/with/try bodies but NOT
        into function/class bodies (deferred — they do not run at collection), and we SKIP a ``try``
        that guards ImportError/ModuleNotFoundError (an optional import that cannot wall collection)."""
    def _emit(node):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, None, a.name.count(".") + 1
        elif isinstance(node, ast.ImportFrom):
            level = node.level or 0
            mod = node.module or ""
            if level > 0:
                keep = len(pkg_parts) - (level - 1)
                base = pkg_parts[:keep] if keep >= 0 else []
                abs_mod = ".".join([p for p in (list(base) + ([mod] if mod else [])) if p])
            else:
                abs_mod = mod
            if not abs_mod:
                return
            depth = abs_mod.count(".") + 1
            for a in node.names:
                yield abs_mod, (None if a.name == "*" else a.name), depth

    def _walk(body):
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                yield from _emit(node)
            elif isinstance(node, ast.If):
                yield from _walk(node.body)
                yield from _walk(node.orelse)
            elif isinstance(node, ast.With):
                yield from _walk(node.body)
            elif isinstance(node, ast.Try):
                if not _try_guards_import(node):
                    yield from _walk(node.body)
                yield from _walk(node.orelse)
                yield from _walk(node.finalbody)
            # FunctionDef / AsyncFunctionDef / ClassDef bodies are DEFERRED -> not collection-time

    yield from _walk(getattr(tree, "body", []))


def _external_importable(top: str) -> bool:
    """Is a top-level external package importable in THIS env (no actual import — spec only)?"""
    try:
        return importlib.util.find_spec(top) is not None
    except Exception:
        return False


def _parse_addopts(root: Path) -> str:
    """Best-effort pytest addopts from pyproject.toml / pytest.ini / tox.ini / setup.cfg."""
    # pyproject [tool.pytest.ini_options] addopts
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib  # py311+
            data = tomllib.loads(pyproject.read_text(encoding="utf-8", errors="replace"))
            ao = data.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("addopts")
            if ao:
                return ao if isinstance(ao, str) else " ".join(ao)
        except Exception:
            pass
    for fname, section in (("pytest.ini", "pytest"), ("tox.ini", "pytest"), ("setup.cfg", "tool:pytest")):
        f = root / fname
        if not f.exists():
            continue
        try:
            cp = configparser.ConfigParser()
            cp.read_string(f.read_text(encoding="utf-8", errors="replace"))
            if cp.has_option(section, "addopts"):
                return cp.get(section, "addopts")
        except Exception:
            continue
    return ""


def _suspect_plugin_addopts(addopts: str) -> list:
    out = []
    for opt, mod in _PLUGIN_OPTION_MODULES.items():
        if opt in (addopts or "") and not _external_importable(mod):
            out.append({"option": opt, "plugin_module": mod})
    return out


def _bootstrap_files(root: Path, idx: _RepoIndex, expected_test_ids=None) -> list:
    """The collection-bootstrap set: every conftest.py + the gold test files (or a sample of all
    test files). These are what pytest imports at collection time — the first failure point."""
    files: list[Path] = []
    for p in root.rglob("conftest.py"):
        rel = p.relative_to(root).as_posix()
        if _rel_ok(rel):
            files.append(p)
    test_paths = set()
    for tid in (expected_test_ids or []):
        fp = str(tid).split("::", 1)[0]
        if fp:
            test_paths.add(fp)
    if test_paths:
        for tp in sorted(test_paths)[:60]:
            cand = root / tp
            if cand.exists() and _rel_ok(tp):
                files.append(cand)
    else:
        n = 0
        for p in root.rglob("*.py"):
            rel = p.relative_to(root).as_posix()
            if _rel_ok(rel) and "test" in Path(rel).name:
                files.append(p)
                n += 1
                if n >= 40:
                    break
    # dedupe, preserve order
    seen, uniq = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


def analyze_collection(source_repo: str, *, expected_test_ids=None, max_files: int = 120) -> dict:
    """Static collection diagnosis (see module docstring). Pure read; never imports repo code;
    never raises. Returns a JSON-safe dict."""
    root = Path(source_repo)
    result = {
        "collects_cleanly": True,
        "unresolved_internal": [],     # [{module, symbol, importer}] -> the must-implement closure
        "unresolved_external": [],     # external top-level packages not importable in this env
        "first_failing_import": None,  # the shallowest unresolved internal import (likely first error)
        "import_depth": 0,             # deepest internal dotted module reached (SPFG++ rising signal)
        "addopts": "",
        "suspect_plugin_addopts": [],
        "bootstrap_files": [],
        "evidence": [],
    }
    try:
        idx = _RepoIndex(root)
    except Exception:
        return result
    top_pkgs = idx.top_packages
    boot = _bootstrap_files(root, idx, expected_test_ids)[:max_files]
    result["bootstrap_files"] = [b.relative_to(root).as_posix() for b in boot]
    seen_internal: set = set()
    seen_external: set = set()
    max_depth = 0
    for bf in boot:
        try:
            tree = ast.parse(bf.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        importer = bf.relative_to(root).as_posix()
        # the importer's own package (for resolving relative imports, edge e); src-stripped (edge f)
        pkg_parts = importer.split("/")[:-1]
        if pkg_parts and pkg_parts[0] in idx.strip_roots:
            pkg_parts = pkg_parts[1:]
        for mod, name, depth in _collection_imports(tree, pkg_parts):
            top = idx.top_module(mod)
            is_internal = top in top_pkgs or idx.resolve(mod) is not None
            if not is_internal:
                if not _external_importable(top) and top not in seen_external:
                    seen_external.add(top)
                    result["unresolved_external"].append(top)
                continue
            target = idx.resolve(mod)
            if target is None:
                # internal dotted module referenced but no file/package for it -> implement it
                key = (mod, name or "")
                if key not in seen_internal:
                    seen_internal.add(key)
                    result["unresolved_internal"].append(
                        {"module": mod, "symbol": name, "importer": importer, "reason": "missing_module"})
                continue
            # the internal module exists; if a SPECIFIC symbol was imported, check it is exported
            max_depth = max(max_depth, depth)
            if name is not None:
                # SUBMODULE-AWARE (critical): `from pkg import name` is VALID when `name` is a SUBMODULE
                # (pkg/name.py or pkg/name/__init__.py exists) — it imports the submodule, regardless of
                # whether `name` appears in pkg/__init__.py. The commit0 base strips __init__ re-exports
                # but keeps the submodule files, so without this check every `from pkg import submodule`
                # is a FALSE "missing symbol" -> a wrong collection-wall verdict (the babel 922 regression:
                # all 18 "missing" symbols were existing submodule files). Only a name that is NEITHER an
                # __init__ export NOR a submodule (e.g. pydantic GenerateSchema) is genuinely missing.
                if idx.resolve(f"{mod}.{name}") is not None:
                    max_depth = max(max_depth, depth + 1)   # a deeper internal module resolves
                    continue
                names = _module_top_level_names(target)
                if names is not None and "*" not in names and name not in names:
                    key = (mod, name)
                    if key not in seen_internal:
                        seen_internal.add(key)
                        result["unresolved_internal"].append(
                            {"module": mod, "symbol": name, "importer": importer,
                             "reason": "missing_symbol"})
    result["import_depth"] = max_depth
    result["unresolved_external"].sort()
    # CONFTEST-GATED collection wall. A conftest.py is imported by pytest BEFORE any test, so an
    # unresolved import THERE blocks the WHOLE suite from collecting (the pydantic GenerateSchema
    # case) — that is the genuine total wall a synthetic Phase 0 targets. A test-FILE-level unresolved
    # import only stops THAT file from collecting; the rest of the suite still collects + runs, and the
    # normal decompose->solve flow implements the module incrementally. Triggering Phase 0 on test-file
    # imports mis-steered babel (922/5663 vs the baseline's 5663 solve), so collects_cleanly + the
    # must_implement closure are driven by CONFTEST blockers only. All unresolved are kept in
    # unresolved_internal for diagnose()'s blocker_class signal.
    conftest_unres = [u for u in result["unresolved_internal"]
                      if Path(u.get("importer", "")).name == "conftest.py"]
    result["conftest_unresolved"] = conftest_unres
    result["collects_cleanly"] = not conftest_unres
    _mi, _seen = [], set()
    for u in conftest_unres:
        m = u.get("module")
        if m and m not in _seen:
            _seen.add(m)
            _mi.append(m)
    result["must_implement_modules"] = _mi
    if conftest_unres:
        first = min(conftest_unres, key=lambda d: d["module"].count("."))
        result["first_failing_import"] = first
        result["evidence"].append(
            f"{len(conftest_unres)} CONFTEST-level unresolved import(s) block whole-suite collection; "
            f"first: {first['importer']} -> {first['module']}"
            + (f".{first['symbol']}" if first['symbol'] else ""))
    elif result["unresolved_internal"]:
        # informational only: per-file imports that resolve incrementally (NOT a collection wall)
        result["evidence"].append(
            f"{len(result['unresolved_internal'])} test-file-level unresolved import(s) "
            "(incremental — collects per-file as modules land; not a collection wall)")
    result["addopts"] = _parse_addopts(root)
    result["suspect_plugin_addopts"] = _suspect_plugin_addopts(result["addopts"])
    if result["suspect_plugin_addopts"]:
        result["evidence"].append(
            "pytest addopts reference uninstalled plugin(s): "
            + ", ".join(s["option"] for s in result["suspect_plugin_addopts"]))
    if result["unresolved_external"]:
        result["evidence"].append(
            "external import(s) not importable in this env: " + ", ".join(result["unresolved_external"][:8]))
    return result


def must_implement_modules(diagnosis: dict) -> list:
    """The dedup'd repo-internal module closure whose CONFTEST-level imports block whole-suite
    collection (consumed by Phase 0's must_implement reconcile). Driven by the conftest blockers only
    — a test-file-level unresolved import is incremental work, not a collection wall, so it is NOT in
    this closure (prevents the wrong-Phase-0 mis-steer). Prefers the precomputed field; falls back to
    deriving from ``conftest_unresolved`` for older diagnosis dicts."""
    diagnosis = diagnosis or {}
    if diagnosis.get("must_implement_modules") is not None:
        return list(diagnosis["must_implement_modules"])
    mods, seen = [], set()
    for u in diagnosis.get("conftest_unresolved", []) or []:
        m = u.get("module")
        if m and m not in seen:
            seen.add(m)
            mods.append(m)
    return mods
