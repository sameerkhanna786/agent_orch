"""Build the perturbed SKELETON from the commit0 BASE-COMMIT skeleton (text-level).

The commit0 base_commit skeleton is the canonical *incomplete* repo: signatures
+ docstrings kept, bodies replaced by ``pass`` (so it imports cleanly even with
module-level helper calls), and the agent must fill the bodies in.  It does NOT
parse as complete Python and it carries the VANILLA symbol surface.

We cannot re-parse it (incomplete), and deriving the skeleton by body-stripping
the perturbed REFERENCE breaks module-level calls (a stubbed helper raising at
import zeroes test collection).  So we apply the persisted name_map to the base
skeleton at the TEXT level, which is sound because:

* module/symbol short-names are unambiguous identifiers (the name_map guarantees
  collision-freeness), so whole-word identifier substitution is exact;
* renaming module FILES on disk + rewriting ``import``/``from`` statements is the
  same set of edits rope performs on the reference, derived from the same map;
* the base skeleton's import-cleanliness (commit0's design) is preserved verbatim
  — we only rename identifiers, never touch control flow or bodies.

The result: a perturbed, import-clean, incomplete skeleton whose renamed surface
matches the scored reference EXACTLY (closes spec §3.5).  Test files are renamed
too (they ARE part of the base skeleton and define the task against the renamed
surface).
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from .namemap import NameMap


def _apply_text_rename(text: str, leaf_map: dict[str, str], module_map: dict[str, str]) -> str:
    """Whole-word identifier substitution + dotted-module path rewrite.

    Longest names first so a shorter name never shadows a longer one.
    """
    # 1) dotted module paths in import statements (e.g. ``voluptuous.schema_builder``)
    #    rewrite the LAST dotted component (the renamed leaf) wherever the full
    #    dotted path appears.  module_map keys are full FQNs.
    #
    #    CRITICAL: a module's leaf name is often an ordinary English word (e.g.
    #    ``error``); substituting the BARE word everywhere would corrupt docstring
    #    PROSE ("An error was encountered" -> "An mod_xxx was encountered").  So we
    #    rewrite the leaf ONLY in unambiguous module CONTEXTS: dotted paths and
    #    explicit ``import``/``from ... import`` statements — never bare prose words.
    for old_fqn, new_leaf in sorted(module_map.items(), key=lambda kv: -len(kv[0])):
        old_leaf = old_fqn.rsplit(".", 1)[-1]
        prefix = old_fqn.rsplit(".", 1)[0]
        # (a) dotted path: ``voluptuous.schema_builder`` -> ``voluptuous.mod_xxx``
        text = re.sub(
            rf"(?<![\w.]){re.escape(prefix)}\.{re.escape(old_leaf)}(?![\w])",
            f"{prefix}.{new_leaf}",
            text,
        )
        # (b) ``from <prefix> import ... error ...`` / ``from . import error``
        #     rewrite the leaf only when it appears in an import list after
        #     ``import`` (a comma/whitespace-delimited name in an import statement).
        def _rewrite_import_list(m: "re.Match") -> str:
            head, names = m.group(1), m.group(2)
            names = re.sub(rf"(?<![\w.]){re.escape(old_leaf)}(?![\w])", new_leaf, names)
            return head + names
        # `from X import a, b, error as e`  (single-line import lists)
        text = re.sub(
            r"(^[ \t]*from[ \t]+[\w.]+[ \t]+import[ \t]+)([^\n#]*)",
            _rewrite_import_list, text, flags=re.MULTILINE,
        )
        # `import error` / `import error as e`  (bare module import)
        text = re.sub(
            rf"(^[ \t]*import[ \t]+){re.escape(old_leaf)}(?![\w])",
            rf"\1{new_leaf}", text, flags=re.MULTILINE,
        )
    # 2) symbol leaf names (functions/classes) — whole-word, longest-first.
    #    Skip occurrences inside a leading docstring's PROSE? No: rope on the
    #    REFERENCE is binding-aware and rewrites real symbol references in
    #    docstrings (consistency), so renaming the symbol's leaf word in the
    #    skeleton's matching docstring keeps the two surfaces aligned.  A bare
    #    class-name word in prose (e.g. "Required field") is a genuine symbol
    #    reference and SHOULD track the rename.
    for old, new in sorted(leaf_map.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"(?<![\w.]){re.escape(old)}(?![\w])", new, text)
    return text


def build_skeleton_from_base(
    base_tree: Path,
    skeleton_out: Path,
    name_map: NameMap,
    *,
    top_package: str,
) -> dict:
    """Copy the base-commit skeleton to *skeleton_out* and text-rename it.

    Returns ``{"files_renamed": N, "modules_moved": M}``.
    """
    base_tree = base_tree.resolve()
    if skeleton_out.exists():
        shutil.rmtree(skeleton_out)
    shutil.copytree(
        base_tree, skeleton_out,
        ignore=shutil.ignore_patterns(".git", ".ropeproject", "__pycache__", "*.egg-info"),
    )

    leaf_map = {old.rsplit(".", 1)[-1]: new for old, new in name_map.symbols.items()}
    module_map = dict(name_map.modules)

    # 1) rewrite identifiers + import paths inside every text file
    files_renamed = 0
    for f in skeleton_out.rglob("*"):
        if not f.is_file():
            continue
        if "__pycache__" in f.parts or ".ropeproject" in f.parts:
            continue
        if f.suffix not in (".py", ".md", ".rst", ".txt", ".cfg", ".toml", ".ini"):
            continue
        try:
            src = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        # doctest/markdown: only rewrite doctest lines to avoid corrupting prose
        if f.suffix in (".md", ".rst", ".txt"):
            new_lines = []
            changed = False
            for line in src.splitlines(keepends=True):
                s = line.lstrip()
                if s.startswith(">>>") or s.startswith("..."):
                    nl = _apply_text_rename(line, leaf_map, module_map)
                    changed = changed or (nl != line)
                    new_lines.append(nl)
                else:
                    new_lines.append(line)
            new = "".join(new_lines)
        else:
            new = _apply_text_rename(src, leaf_map, module_map)
            changed = (new != src)
        if changed:
            f.write_text(new, encoding="utf-8")
            files_renamed += 1

    # 2) rename module FILES / package dirs on disk (deepest-first)
    modules_moved = 0
    for old_fqn, new_leaf in sorted(module_map.items(), key=lambda kv: -kv[0].count(".")):
        rel = old_fqn.replace(".", "/")
        old_py = skeleton_out / f"{rel}.py"
        old_pkg = skeleton_out / rel
        if old_py.exists():
            new_py = old_py.with_name(f"{new_leaf}.py")
            old_py.rename(new_py)
            modules_moved += 1
        elif old_pkg.is_dir() and (old_pkg / "__init__.py").exists():
            new_pkg = old_pkg.with_name(new_leaf)
            old_pkg.rename(new_pkg)
            modules_moved += 1

    return {"files_renamed": files_renamed, "modules_moved": modules_moved}
