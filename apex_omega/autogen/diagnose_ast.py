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
        for p in root.rglob("*.py"):
            rel = p.relative_to(root).as_posix()
            if not _rel_ok(rel):
                continue
            parts = rel[:-3].split("/")          # strip .py
            if parts[-1] == "__init__":
                parts = parts[:-1]
                if not parts:
                    continue
            self.by_dotted[".".join(parts)] = p

    def resolve(self, dotted: str) -> Optional[Path]:
        """The file for a dotted module, or its package __init__, if it lives in the repo."""
        if dotted in self.by_dotted:
            return self.by_dotted[dotted]
        pkg_init = self.root / Path(*dotted.split(".")) / "__init__.py"
        if pkg_init.exists() and _rel_ok(pkg_init.relative_to(self.root).as_posix()):
            return pkg_init
        return None

    def top_module(self, dotted: str) -> str:
        return dotted.split(".", 1)[0]

    @property
    def top_packages(self) -> set:
        return {d.split(".", 1)[0] for d in self.by_dotted}


def _iter_imports(tree: ast.AST):
    """Yield (module_dotted, imported_name_or_None, depth) for each import in a parsed module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, None, a.name.count(".") + 1
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative import: resolved within its own package, skip cross-repo classify
            mod = node.module or ""
            if not mod:
                continue
            depth = mod.count(".") + 1
            for a in node.names:
                yield mod, (None if a.name == "*" else a.name), depth


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
        for mod, name, depth in _iter_imports(tree):
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
    if result["unresolved_internal"]:
        result["collects_cleanly"] = False
        # the shallowest (fewest dots) unresolved internal import is the likely first collection error
        first = min(result["unresolved_internal"], key=lambda d: d["module"].count("."))
        result["first_failing_import"] = first
        result["evidence"].append(
            f"{len(result['unresolved_internal'])} unresolved internal import(s) in the collection "
            f"bootstrap; first likely error: {first['importer']} -> "
            f"{first['module']}" + (f".{first['symbol']}" if first['symbol'] else ""))
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
    """The dedup'd repo-internal module closure that must be implemented before collection succeeds
    (consumed by Phase 0's must_implement reconcile)."""
    mods, seen = [], set()
    for u in (diagnosis or {}).get("unresolved_internal", []) or []:
        m = u.get("module")
        if m and m not in seen:
            seen.add(m)
            mods.append(m)
    return mods
