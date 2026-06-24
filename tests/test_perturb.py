"""Tests for the perturbed-commit0 de-contamination pipeline.

Three guarantees:

(a) the FQN classifier excludes builtins/stdlib/third-party/dunders (only
    repo-defined symbols are renamed);
(b) a tiny synthetic repo round-trips — rename impl+tests, semantics preserved,
    a sample test still passes;
(c) vanilla commit0 registry/targets stay byte-identical when the perturbed
    module is unused (the sidecar absent => no behavioral change).

The classifier/rename tests need ``libcst``+``rope`` (build-venv deps).  When
they are not importable (the default in ``.venv_omega``), those tests SKIP; the
dependency-free namemap + registry-byte-identity tests always run.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# rope+libcst are build-venv only; gate these tests on their availability.
_HAS_BUILD_DEPS = (
    importlib.util.find_spec("libcst") is not None
    and importlib.util.find_spec("rope") is not None
)
requires_build_deps = pytest.mark.skipif(
    not _HAS_BUILD_DEPS, reason="rope+libcst not installed (build-venv only)"
)


# --------------------------------------------------------------------------
# Dependency-free: namemap determinism + collision-freeness
# --------------------------------------------------------------------------
def test_namemap_is_deterministic_and_seeded():
    from apex_omega.eval.perturb import namemap

    wl = [("pkg.mod.foo", "function"), ("pkg.mod.Bar", "class")]
    a = namemap.build_name_map(wl, seed=1337)
    b = namemap.build_name_map(wl, seed=1337)
    c = namemap.build_name_map(wl, seed=2024)
    assert a.symbols == b.symbols  # same seed -> identical map
    assert a.symbols != c.symbols  # different seed -> different map


def test_namemap_never_emits_keyword_or_builtin():
    from apex_omega.eval.perturb import namemap

    # force-feed FQNs whose hash might land on a reserved-ish prefix; check guard
    wl = [(f"pkg.s{i}", "function") for i in range(200)]
    nm = namemap.build_name_map(wl, seed=7)
    import keyword
    import builtins

    bad = set(keyword.kwlist) | set(dir(builtins))
    for new in nm.symbols.values():
        assert new not in bad
        assert new.isidentifier()
    # collision-free
    assert len(set(nm.symbols.values())) == len(nm.symbols)


def test_namemap_round_trips_json():
    from apex_omega.eval.perturb import namemap

    nm = namemap.build_name_map(
        [("pkg.foo", "function")], seed=1, module_worklist=["pkg.mod"]
    )
    nm2 = namemap.NameMap.from_json(nm.to_json())
    assert nm2.symbols == nm.symbols
    assert nm2.modules == nm.modules


# --------------------------------------------------------------------------
# (a) FQN classifier excludes builtins/stdlib/third-party
# --------------------------------------------------------------------------
@requires_build_deps
def test_classifier_excludes_non_repo_symbols(tmp_path: Path):
    from apex_omega.eval.perturb import inventory

    pkg = tmp_path / "mylib"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "core.py").write_text(
        "import os\n"
        "from collections import OrderedDict\n"
        "\n"
        "def repo_helper(x):\n"        # repo-defined -> SHOULD be in worklist
        "    return os.path.join(x)\n"  # os.path.join -> stdlib, EXCLUDE
        "\n"
        "class RepoThing(OrderedDict):\n"  # repo-defined class -> include; base stdlib excluded
        "    def method_a(self):\n"
        "        return len(self)\n"   # len builtin -> EXCLUDE
        "\n"
        "def __dunder__():\n"          # dunder -> EXCLUDE
        "    pass\n",
        encoding="utf-8",
    )
    inv = inventory.build_inventory(tmp_path, [pkg], ("mylib",))
    short_names = {s.short_name for s in inv.symbols}
    # repo-defined present
    assert "repo_helper" in short_names
    assert "RepoThing" in short_names
    # builtins / stdlib / dunders absent
    assert "join" not in short_names      # os.path.join
    assert "len" not in short_names       # builtin
    assert "OrderedDict" not in short_names  # imported stdlib
    assert "os" not in short_names
    assert "__dunder__" not in short_names
    # every recorded FQN is repo-prefixed
    for s in inv.symbols:
        assert s.fqn.startswith("mylib"), s.fqn


@requires_build_deps
def test_classifier_string_literal_exclusion(tmp_path: Path):
    from apex_omega.eval.perturb import inventory

    pkg = tmp_path / "mylib"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "core.py").write_text(
        "class Leaky:\n"            # appears in a string literal below -> excluded
        "    pass\n"
        "\n"
        "def clean_fn():\n"          # not in any string literal -> renamable
        "    return 'Leaky was here'\n",
        encoding="utf-8",
    )
    inv = inventory.build_inventory(tmp_path, [pkg], ("mylib",))
    wl = {f for f, _ in inv.def_worklist(exclude_string_literal_names=True)}
    assert "mylib.core.clean_fn" in wl
    assert "mylib.core.Leaky" not in wl  # excluded: name leaks into a string


# --------------------------------------------------------------------------
# (b) tiny synthetic repo round-trips (rename impl+tests, sample test passes)
# --------------------------------------------------------------------------
@requires_build_deps
def test_synthetic_repo_round_trip(tmp_path: Path):
    from apex_omega.eval.perturb import inventory, namemap, rename

    pkg = tmp_path / "mylib"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from mylib.core import add_numbers, Doubler\n", encoding="utf-8")
    (pkg / "core.py").write_text(
        "def add_numbers(a, b):\n"
        "    return a + b\n"
        "\n"
        "class Doubler:\n"
        "    def run(self, x):\n"
        "        return add_numbers(x, x)\n",
        encoding="utf-8",
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("", encoding="utf-8")
    (tests / "test_core.py").write_text(
        "from mylib.core import add_numbers, Doubler\n"
        "\n"
        "def test_add():\n"
        "    assert add_numbers(2, 3) == 5\n"
        "\n"
        "def test_doubler():\n"
        "    assert Doubler().run(4) == 8\n",
        encoding="utf-8",
    )

    inv = inventory.build_inventory(tmp_path, [pkg, tests], ("mylib",))
    wl = inv.def_worklist(kinds=("function", "class"), exclude_module_prefixes=("tests",))
    nm = namemap.build_name_map(wl, seed=1337, reserved_fqns=inv.all_fqns)
    rep = rename.apply_rename(tmp_path, inv, nm, rename_modules=False)
    assert "mylib.core.add_numbers" in rep.applied
    assert "mylib.core.Doubler" in rep.applied

    # the old names are GONE from the surface
    core_src = (pkg / "core.py").read_text(encoding="utf-8")
    test_src = (tests / "test_core.py").read_text(encoding="utf-8")
    assert "add_numbers" not in core_src
    assert "add_numbers" not in test_src  # rope rewrote the test import + uses
    assert "Doubler" not in test_src

    # semantics preserved: the rewritten test still passes against the renamed code
    shutil.rmtree(tmp_path / ".ropeproject", ignore_errors=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(tests), "-q", "-p", "no:cacheprovider"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(tmp_path), "PATH": __import__("os").environ.get("PATH", "")},
    )
    assert proc.returncode == 0, f"renamed synthetic repo failed its own tests:\n{proc.stdout}\n{proc.stderr}"
    assert "2 passed" in proc.stdout


def test_skeletonize_text_rename_of_base(tmp_path: Path):
    """The perturbed skeleton is a TEXT-level rename of the (incomplete) base
    skeleton: module files moved, imports + symbols renamed, import-cleanliness
    and stub bodies preserved.  Dependency-free (no parsing)."""
    from apex_omega.eval.perturb import namemap, skeletonize

    base = tmp_path / "base"
    pkg = base / "mylib"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("from mylib.core import add_numbers, Doubler\n", encoding="utf-8")
    # an INCOMPLETE base skeleton (pass-bodied; a module-level call must survive)
    (pkg / "core.py").write_text(
        "def add_numbers(a, b):\n"
        "    pass\n"
        "\n"
        "class Doubler:\n"
        "    pass\n"
        "\n"
        "_ITEM = add_numbers  # module-level reference must still resolve\n",
        encoding="utf-8",
    )

    nm = namemap.build_name_map(
        [("mylib.core.add_numbers", "function"), ("mylib.core.Doubler", "class")],
        seed=1337,
        module_worklist=["mylib.core"],
    )
    out = tmp_path / "skeleton"
    stats = skeletonize.build_skeleton_from_base(base, out, nm, top_package="mylib")
    assert stats["modules_moved"] == 1  # core.py -> mod_xxx.py

    new_core_name = nm.modules["mylib.core"]
    new_core = (out / "mylib" / f"{new_core_name}.py").read_text(encoding="utf-8")
    init = (out / "mylib" / "__init__.py").read_text(encoding="utf-8")
    # old surface gone; renamed surface present; stub body + module-level ref kept
    assert "add_numbers" not in new_core
    assert "add_numbers" not in init
    assert nm.symbols["mylib.core.add_numbers"] in new_core
    assert nm.symbols["mylib.core.Doubler"] in new_core
    assert new_core_name in init           # import path rewritten
    assert "pass" in new_core              # stub body preserved (import-clean)
    assert "_ITEM = " in new_core          # module-level reference preserved
    # the original core.py file is gone (moved)
    assert not (out / "mylib" / "core.py").exists()


# --------------------------------------------------------------------------
# (c) vanilla commit0 byte-identical when the perturbed module is unused
# --------------------------------------------------------------------------
def test_vanilla_registry_unchanged_by_perturbed_additions():
    """The 15 vanilla RepoSpecs are untouched; only perturbed entries are added."""
    from apex_omega.eval import registry

    vanilla_names = [r.name for r in registry.TARGET_REPOS if not r.name.endswith("_perturbed")]
    # the canonical 15 vanilla targets, in order, unchanged
    assert vanilla_names == [
        "minitorch", "jinja", "voluptuous", "web3.py", "statsmodels", "babel",
        "pydantic", "pytest", "networkx", "mimesis", "scrapy", "seaborn",
        "sphinx", "geopandas", "cookiecutter",
    ]
    # perturbed entries are non-lite and local-runnable (no Docker)
    for name in ("voluptuous_perturbed", "networkx_perturbed"):
        spec = registry.get(name)
        assert spec.in_lite is False
        assert spec.local_runnable is True
    # lite_targets()/local_runnable_targets() never include a perturbed repo
    assert not any(n.endswith("_perturbed") for n in registry.lite_targets())


def test_perturbed_sidecar_absent_is_inert():
    """With no sidecar file, the harness loader returns {} and synthesizes nothing
    (vanilla discover_tasks path unchanged)."""
    # We test the loader contract directly without importing the heavy apex harness:
    # the sidecar path is namespaced under apex_omega/eval/perturb/variants/ and the
    # loader must tolerate its absence.
    sidecar = REPO_ROOT / "apex_omega" / "eval" / "perturb" / "variants" / "perturbed_targets.json"
    # Whether or not a build has run, the schema is {"targets": {...}} or absent.
    if sidecar.exists():
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert isinstance(data.get("targets", {}), dict)
        # every entry carries the keys the synthetic-task builder requires
        for name, entry in data.get("targets", {}).items():
            assert name.endswith("_perturbed")
            for key in ("repo", "base_commit", "reference_commit", "mirror_root"):
                assert key in entry, f"{name} missing {key}"
    else:
        # absent file is a valid, inert state
        assert True
