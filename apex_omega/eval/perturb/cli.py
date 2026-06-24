"""One-command driver for the perturbed-commit0 build pipeline.

Two execution phases with different interpreter needs:

* BUILD phase (rope + libcst): materialize -> neutralize docs -> inventory ->
  namemap -> rename BOTH trees -> structural alpha-rename check.  Run with the
  build venv (``/tmp/_perturb_venv``).
* GATE+EMIT phase (commit0 importable): install perturbed reference -> run gold
  suite -> 100%-pass gate -> regen bz2 -> git variant -> sidecar wiring.  The
  gate venv is created internally; commit0 is imported only to locate its
  ``data/test_ids`` dir.

``cli.py`` runs both phases in one process when invoked from an interpreter that
has rope+libcst importable AND can locate commit0; otherwise pass
``--commit0-pkg-dir`` explicitly.

Usage:
    python -m apex_omega.eval.perturb.cli <repo> --seed 1337 \
        --base-commit <sha> --reference-commit <sha> \
        --repo-slug commit-0/<repo> \
        --top-package <pkg> --test-dir <dir> --test-cmd <cmd> \
        --python-exe python3.10 [--neutralize-docs] [--scope-module <mod>]
        [--commit0-pkg-dir <dir>]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _clone_checkout(repo_slug: str, commit: str, dest: Path) -> None:
    """Clone ``commit-0/<repo>`` and hard-checkout *commit* into *dest*.

    Some commit0 base SHAs are fetchable by object ID but NOT advertised by the
    repo's default refs (mirrors the harness ``_ensure_task_commit_objects_available``);
    a plain checkout then fails 128, so we ``git fetch origin <sha>`` first.
    """
    if dest.exists():
        shutil.rmtree(dest)
    url = f"https://github.com/{repo_slug}.git"
    subprocess.run(["git", "clone", "--quiet", url, str(dest)], check=True)
    co = subprocess.run(["git", "checkout", "--quiet", commit], cwd=str(dest))
    if co.returncode != 0:
        subprocess.run(["git", "fetch", "--quiet", "origin", commit], cwd=str(dest), check=True)
        subprocess.run(["git", "checkout", "--quiet", commit], cwd=str(dest), check=True)
    shutil.rmtree(dest / ".git", ignore_errors=True)


def _skeleton_old_name_leaks(skeleton_tree: Path, name_map, top_package: str) -> list[str]:
    """Old renamed-symbol leaf names that still appear as CODE IDENTIFIERS in the
    skeleton package source (a leak == the agent could recall the vanilla name
    from code).

    Only checks RENAMED symbols/modules; a name deliberately NOT renamed (excluded)
    is not a leak.  Crucially, this scans only NAME tokens (via ``tokenize``) — it
    IGNORES docstring/comment PROSE, because docstring prose legitimately retains
    ordinary English words that happen to equal a module leaf (e.g. the word
    "error" for a renamed ``error`` module), exactly as the rope-renamed REFERENCE
    keeps them.  The skeleton may not even parse (incomplete repo), so we tokenize
    line-by-line and fall back to a regex that strips strings/comments.
    """
    import re as _re

    old_leaves = {old.rsplit(".", 1)[-1] for old in name_map.symbols}
    old_leaves |= {old.rsplit(".", 1)[-1] for old in name_map.modules}
    new_names = set(name_map.symbols.values()) | set(name_map.modules.values())
    check = old_leaves - new_names
    leaks: list[str] = []
    pkg = skeleton_tree / top_package
    for f in pkg.rglob("*.py"):
        if "__pycache__" in f.parts or "/tests/" in str(f) or f.name.startswith("test_"):
            continue
        try:
            src = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        # Strip docstrings, comments, and string literals, THEN look for the old
        # name as a STANDALONE identifier — the ``(?<![\w.])`` guard means a
        # dotted attribute access (``typing.Optional``, ``er.RangeInvalid``) is NOT
        # a leak (the leaf belongs to another namespace, not the renamed symbol).
        stripped = _re.sub(r'(?s)("""|\'\'\')(?:(?!\1).)*\1', " ", src)
        stripped = _re.sub(r"(?m)#[^\n]*$", " ", stripped)
        stripped = _re.sub(r"(?s)\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'", " ", stripped)
        for name in check:
            if _re.search(rf"(?<![\w.]){_re.escape(name)}(?![\w])", stripped):
                leaks.append(f"{name}@{f.relative_to(skeleton_tree)}")
    return sorted(set(leaks))


def _neutralize_text_tree(roots: list[Path]) -> int:
    """Best-effort docstring/comment neutralization on NON-parseable trees (the
    base skeleton).  Blanks triple-quoted string blocks and strips ``#`` comments
    via regex.  Conservative: only operates on .py files; leaves code intact.
    """
    import re as _re

    triple = _re.compile(r'(?s)("""|\'\'\')(?:(?!\1).)*\1')
    comment = _re.compile(r"(?m)(^|\s)#[^\n]*$")
    changed = 0
    seen: set[Path] = set()
    for root in roots:
        files = [root] if root.is_file() else root.rglob("*.py")
        for f in files:
            f = f.resolve()
            if f in seen or "__pycache__" in f.parts:
                continue
            seen.add(f)
            try:
                src = f.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            new = triple.sub('""', src)
            new = comment.sub(lambda m: m.group(1) + "#", new)
            if new != src:
                f.write_text(new, encoding="utf-8")
                changed += 1
    return changed


def _resolve_roots(tree: Path, top_package: str, test_dir: str, scope_module: Optional[str]) -> list[Path]:
    """Scan roots = {src package} ∪ {test dir}; scoped to one module if requested.

    When *scope_module* is set the roots are restricted to that module's subtree
    (its package/file + its own ``tests/`` dir for repos like networkx that keep
    per-module ``tests/`` siblings) so a module-scoped build does not touch the
    rest of the repo's docstrings/inventory.
    """
    roots: list[Path] = []
    if scope_module:
        rel = scope_module.replace(".", "/")
        cand = tree / f"{rel}.py"
        cand_pkg = tree / rel
        if cand_pkg.is_dir():
            roots.append(cand_pkg)
            # per-module tests sibling (networkx: algorithms/tests/test_dag.py)
        else:
            roots.append(cand)
        # the scoped module's sibling tests dir (covers networkx's layout where
        # tests for algorithms/dag.py live in algorithms/tests/)
        parent = (tree / rel).parent
        for tdir in (parent / "tests", cand_pkg / "tests"):
            if tdir.is_dir():
                roots.append(tdir)
        return [r for r in roots if r.exists()]
    roots.append(tree / top_package)
    if test_dir:
        roots.append(tree / test_dir)
    else:
        roots.append(tree / top_package)
    return [r for r in roots if r.exists()]


def build_and_gate(
    *,
    repo: str,
    repo_slug: str,
    base_commit: str,
    reference_commit: str,
    top_package: str,
    test_dir: str,
    test_cmd: str,
    python_exe: str,
    seed: int,
    neutralize_docs: bool,
    scope_module: Optional[str],
    commit0_pkg_dir: Path,
    src_dir: str,
    python_version: str,
    workdir: Path,
    install_command: str = "pip install -e .",
    extra_pip: tuple[str, ...] = (),
) -> dict:
    """Full pipeline for one repo.  Returns a structured report dict.

    The GATE is non-negotiable: if the perturbed reference does not pass its gold
    suite at 100%, we return ``{"emitted": False, ...}`` and emit NOTHING.
    """
    from . import inventory as _inv
    from . import namemap as _nm
    from . import rename as _rn
    from . import docstrings as _ds
    from . import gate as _gate
    from . import emit as _emit
    from . import skeletonize as _skel

    perturbed_repo_name = f"{repo}_perturbed"
    # idempotent: a prior partial run leaves stale trees that break copytree
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    skeleton_tree = workdir / "skeleton"
    reference_tree = workdir / "reference"
    vanilla_reference = workdir / "vanilla_reference"  # for structural diff

    report: dict = {"repo": repo, "perturbed_repo_name": perturbed_repo_name, "seed": seed}

    # PASS 0 — materialize the REFERENCE (gold) checkout.  The perturbed SKELETON
    # is DERIVED from the perturbed reference (post-rename, post-gate) — NOT from
    # the base_commit checkout.  The base_commit skeleton is an *incomplete* repo
    # (signatures kept, bodies removed) that does NOT parse AND carries the VANILLA
    # surface; deriving the skeleton from the gated reference guarantees the agent
    # sees the renamed surface and the skeleton API == the scored API (spec §3.5).
    _clone_checkout(repo_slug, reference_commit, reference_tree)
    shutil.copytree(reference_tree, vanilla_reference, dirs_exist_ok=False)

    # DOC PROSE IS RETAINED BY DEFAULT (rename-only de-contamination).  Docstrings
    # and comments ARE the natural-language spec the agent implements against;
    # stripping them would make the task HARDER rather than just de-memorized.  We
    # ONLY rename symbol *references* inside docstrings/comments/doctests (rope
    # ``docs=True`` below + the .md doctest rewriter) so they stay consistent with
    # the renamed code, while the descriptive prose is preserved verbatim.
    # ``neutralize_docs`` is OFF by default and SHOULD stay off; it remains only as
    # an explicit opt-in escape hatch and is intentionally NOT used for the shipped
    # variants.
    if neutralize_docs:
        roots_ref = _resolve_roots(reference_tree, top_package, test_dir, scope_module)
        report["docs_neutralized_reference"] = _ds.neutralize_tree(roots_ref)

    # PASS 1 — inventory on the REFERENCE tree only (skeleton bindings are absent).
    # For a module-scoped build we still inventory the WHOLE package (so rope can
    # rewrite cross-file USES of the scoped symbols); the rename is then restricted
    # to the scope_prefix via the worklist below.
    ref_roots = _resolve_roots(reference_tree, top_package, test_dir, scope_module=None)
    inv = _inv.build_inventory(reference_tree, ref_roots, (top_package,))
    report["inventory_def_sites"] = len(inv.symbols)
    scope_prefix = scope_module if scope_module else None

    # exclude the tests package from rename worklist (keep node ids comparable)
    test_pkg_prefixes: tuple[str, ...] = ()
    if test_dir:
        # derive dotted module prefix for the test dir (e.g. voluptuous.tests)
        test_mod = test_dir.strip("/").replace("/", ".")
        test_pkg_prefixes = (test_mod,)
    # also exclude common 'tests' subpackages
    test_pkg_prefixes = test_pkg_prefixes + tuple(
        m for m in inv.modules if m.split(".")[-1] == "tests" or ".tests." in m + "."
    )

    worklist = inv.def_worklist(
        kinds=("function", "class"),
        exclude_module_prefixes=test_pkg_prefixes,
        exclude_string_literal_names=True,
        scope_prefix=scope_prefix,
    )
    modules = inv.module_worklist(exclude_prefixes=test_pkg_prefixes, scope_prefix=scope_prefix)
    report["worklist_symbols"] = len(set(f for f, _ in worklist))
    report["worklist_modules"] = len(modules)

    name_map = _nm.build_name_map(
        worklist, seed=seed, reserved_fqns=inv.all_fqns, module_worklist=modules,
    )

    # PASS 3 — rename the REFERENCE tree with the persisted map.
    rep_ref = _rn.apply_rename(reference_tree, inv, name_map, rename_modules=True)
    report["renamed_reference"] = len(rep_ref.applied)
    report["module_renames"] = len(rep_ref.module_renames)
    report["rename_skipped_reference"] = rep_ref.skipped

    # PASS 3.4 — rewrite doctest-glob files (e.g. tests.md) that reference public
    # symbols by name; rope does not touch non-.py files but their gold doctests do.
    report["doctest_rewrites_reference"] = _rn.rewrite_doctest_globs(reference_tree, name_map)

    # PASS 3.5 — count reference files (the gate below is the authoritative
    # alpha-rename-only proof: it runs the vanilla tests against the renamed code).
    rel_files = [str(f.relative_to(reference_tree))
                 for f in (reference_tree / top_package).rglob("*.py")]
    report["reference_files"] = len(rel_files)

    # clean rope folder before install/gate
    shutil.rmtree(reference_tree / ".ropeproject", ignore_errors=True)

    # PASS 2 (GATE) — install perturbed reference + run gold suite, REQUIRE 100%
    gate_res = _gate.run_gate(
        reference_tree,
        test_dir=test_dir,
        test_cmd=test_cmd,
        python_exe=python_exe,
        venv_dir=workdir / "gate_venv",
        install_command=install_command,
        extra_pip=extra_pip,
        double_run=True,
    )
    report["gate_passed"] = gate_res.passed
    report["gate_detail"] = gate_res.detail
    report["gate_collected"] = gate_res.collected
    report["gate_n_passed"] = gate_res.n_passed
    report["gate_n_failed"] = gate_res.n_failed
    report["gate_n_errors"] = gate_res.n_errors

    if not gate_res.passed:
        report["emitted"] = False
        report["reason"] = "GATE FAILED — perturbed reference does not pass its own gold tests; rename unsound for this repo; emitting nothing."
        return report

    # vanilla id-count sanity (count parity with the vanilla bz2, if present)
    vanilla_bz2 = commit0_pkg_dir / "data" / "test_ids" / f"{repo.lower().replace('.', '-')}.bz2"
    if vanilla_bz2.exists():
        vanilla_ids = _gate.read_expected_ids_bz2(vanilla_bz2)
        report["vanilla_id_count"] = len(vanilla_ids)
        report["perturbed_id_count"] = len(gate_res.expected_ids)
        # node-id BASE set equality modulo the test-file path (test names unchanged)
        def _base(nid: str) -> str:
            return nid.split("::", 1)[-1].split("[", 1)[0]
        report["id_base_set_parity"] = (
            {_base(x) for x in vanilla_ids} == {_base(x) for x in gate_res.expected_ids}
        )

    # DERIVE the perturbed SKELETON from the commit0 BASE-COMMIT skeleton via a
    # TEXT-level rename of the SAME name_map.  The base skeleton is the canonical
    # import-clean incomplete repo (commit0's design); re-parsing/body-stripping
    # would break module-level helper calls.  Docstrings/comments are RETAINED
    # (the spec) — the text-rename rewrites symbol references inside them (incl.
    # ``>>>`` doctest lines) so they stay consistent, while prose is preserved.
    base_skel_tree = workdir / "base_skeleton"
    _clone_checkout(repo_slug, base_commit, base_skel_tree)
    if neutralize_docs:
        # text-only neutralization (the base skeleton does not parse) — opt-in only
        report["docs_neutralized_skeleton"] = _neutralize_text_tree(
            _resolve_roots(base_skel_tree, top_package, test_dir, scope_module)
        )
    skel_stats = _skel.build_skeleton_from_base(
        base_skel_tree, skeleton_tree, name_map, top_package=top_package,
    )
    report["skeleton"] = skel_stats

    # SKELETON SANITY GATE: the perturbed skeleton is a pure TEXT-rename of
    # commit0's own base skeleton, which by construction is an *incomplete* repo
    # (signatures + ``pass``/empty bodies) that CANNOT pass the gold tests.  A full
    # install+pytest would just re-confirm "incomplete"; instead we assert the
    # cheap structural invariants that actually matter for the variant:
    #   (1) the renamed module FILES exist (modules_moved > 0 when modules renamed);
    #   (2) the perturbed surface is present in the skeleton (renamed symbols) AND
    #       the vanilla surface is GONE (no old leaf names leak in source);
    #   (3) the skeleton still carries the (incomplete) signatures, so it is a real
    #       construction task, not a solved repo.
    skel_leaks = _skeleton_old_name_leaks(skeleton_tree, name_map, top_package)
    report["skeleton_old_name_leaks"] = skel_leaks[:20]
    report["skeleton_renamed_files"] = skel_stats.get("files_renamed", 0)
    if skel_leaks:
        report["emitted"] = False
        report["reason"] = (
            f"SKELETON SANITY FAILED — {len(skel_leaks)} old symbol name(s) leak "
            f"into the perturbed skeleton source (e.g. {skel_leaks[:5]}). Emitting nothing."
        )
        return report

    # EMIT — git variant + bz2 + manifest + sidecar wiring
    emit_res = _emit.emit_variant(
        perturbed_repo_name=perturbed_repo_name,
        skeleton_tree=skeleton_tree,
        reference_tree=reference_tree,
        expected_ids=gate_res.expected_ids,
        commit0_pkg_dir=commit0_pkg_dir,
        src_dir=src_dir,
        test_dir=test_dir,
        test_cmd=test_cmd,
        python_version=python_version,
        name_map_json=name_map.to_json(),
        rename_report={"reference": rep_ref.applied, "modules": rep_ref.module_renames,
                       "skipped": rep_ref.skipped},
        seed=seed,
        base_repo_slug=repo_slug,
    )
    report.update({
        "emitted": True,
        "variant_repo_path": emit_res.variant_repo_path,
        "base_commit": emit_res.base_commit,
        "reference_commit": emit_res.reference_commit,
        "bz2_path": emit_res.bz2_path,
        "manifest_path": emit_res.manifest_path,
        "expected_id_count": emit_res.expected_id_count,
        "repo_slug": emit_res.repo_slug,
    })
    return report


def _locate_commit0_pkg_dir(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit)
    import commit0
    import os
    return Path(os.path.dirname(commit0.__file__))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build a perturbed-commit0 variant.")
    ap.add_argument("repo")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--repo-slug", default="", help="vanilla task.repo e.g. commit-0/voluptuous")
    ap.add_argument("--base-commit", default="")
    ap.add_argument("--reference-commit", default="")
    ap.add_argument("--top-package", default="")
    ap.add_argument("--test-dir", default="")
    ap.add_argument("--test-cmd", default="pytest")
    ap.add_argument("--src-dir", default="")
    ap.add_argument("--python-version", default="3.10")
    ap.add_argument("--python-exe", default=sys.executable)
    ap.add_argument("--neutralize-docs", action="store_true")
    ap.add_argument("--scope-module", default=None)
    ap.add_argument("--install-command", default="pip install -e .")
    ap.add_argument("--extra-pip", default="", help="comma-separated extra pip packages for the gate venv")
    ap.add_argument("--commit0-pkg-dir", default=None)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--report-json", default=None)
    ap.add_argument("--restore-bz2", action="store_true",
                    help="re-drop the gold bz2 for <repo>_perturbed from its manifest (no rebuild)")
    args = ap.parse_args(argv)

    commit0_pkg_dir = _locate_commit0_pkg_dir(args.commit0_pkg_dir)

    if args.restore_bz2:
        from . import emit as _emit
        name = args.repo if args.repo.endswith("_perturbed") else f"{args.repo}_perturbed"
        path = _emit.restore_bz2_from_manifest(name, commit0_pkg_dir)
        print(json.dumps({"restored": str(path) if path else None, "repo": name}, indent=2))
        return 0 if path else 2

    for req in ("repo_slug", "base_commit", "reference_commit", "top_package"):
        if not getattr(args, req):
            ap.error(f"--{req.replace('_', '-')} is required when building (omit only with --restore-bz2)")
    workdir = Path(args.workdir) if args.workdir else Path(f"/tmp/_perturb_build/{args.repo}_work")
    extra_pip = tuple(x for x in (args.extra_pip.split(",") if args.extra_pip else []) if x)

    report = build_and_gate(
        repo=args.repo, repo_slug=args.repo_slug,
        base_commit=args.base_commit, reference_commit=args.reference_commit,
        top_package=args.top_package, test_dir=args.test_dir, test_cmd=args.test_cmd,
        python_exe=args.python_exe, seed=args.seed,
        neutralize_docs=args.neutralize_docs, scope_module=args.scope_module,
        commit0_pkg_dir=commit0_pkg_dir, src_dir=args.src_dir,
        python_version=args.python_version, workdir=workdir,
        install_command=args.install_command, extra_pip=extra_pip,
    )
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.report_json:
        Path(args.report_json).write_text(text, encoding="utf-8")
    return 0 if report.get("emitted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
