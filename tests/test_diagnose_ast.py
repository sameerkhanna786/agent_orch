"""diagnose STAGE 1: zero-token AST collection pre-pass (O2/O3/O4)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from apex_omega.autogen.diagnose_ast import analyze_collection, must_implement_modules


def _mk(files: dict) -> str:
    d = Path(tempfile.mkdtemp()) / "repo"
    d.mkdir()
    for rel, content in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return str(d)


def test_clean_repo_collects_cleanly():
    repo = _mk({
        "pkg/__init__.py": "from pkg.core import Thing\n",
        "pkg/core.py": "class Thing:\n    pass\n",
        "tests/test_thing.py": "from pkg import Thing\n\ndef test_t():\n    assert Thing\n",
    })
    diag = analyze_collection(repo, expected_test_ids=["tests/test_thing.py::test_t"])
    assert diag["collects_cleanly"] is True
    assert diag["unresolved_internal"] == []


def test_missing_symbol_in_conftest_is_collection_collapse():
    """The pydantic GenerateSchema class: conftest imports a name the (stubbed) module doesn't export
    yet -> collection errors before anything runs. Pure AST must catch it."""
    repo = _mk({
        "pkg/__init__.py": "",
        "pkg/schema.py": "# implementation stripped on the commit0 base\n",  # no GenerateSchema
        "conftest.py": "from pkg.schema import GenerateSchema\n",
        "tests/test_x.py": "def test_x():\n    assert True\n",
    })
    diag = analyze_collection(repo, expected_test_ids=["tests/test_x.py::test_x"])
    assert diag["collects_cleanly"] is False
    assert any(u["symbol"] == "GenerateSchema" and u["reason"] == "missing_symbol"
               for u in diag["unresolved_internal"])
    assert diag["first_failing_import"]["module"] == "pkg.schema"
    assert "pkg.schema" in must_implement_modules(diag)


def test_missing_internal_module():
    repo = _mk({
        "pkg/__init__.py": "",
        "conftest.py": "from pkg.notyet import helper\n",  # pkg/notyet.py does not exist
        "tests/test_y.py": "def test_y():\n    assert True\n",
    })
    diag = analyze_collection(repo, expected_test_ids=["tests/test_y.py::test_y"])
    assert diag["collects_cleanly"] is False
    assert any(u["module"] == "pkg.notyet" and u["reason"] == "missing_module"
               for u in diag["unresolved_internal"])


def test_import_depth_rises_with_resolvable_chain():
    """import_depth = deepest resolvable internal dotted module — the SPFG++ rising signal."""
    repo = _mk({
        "pkg/__init__.py": "",
        "pkg/a/__init__.py": "",
        "pkg/a/b.py": "VALUE = 1\n",
        "conftest.py": "from pkg.a.b import VALUE\n",
        "tests/test_z.py": "def test_z():\n    assert True\n",
    })
    diag = analyze_collection(repo, expected_test_ids=["tests/test_z.py::test_z"])
    assert diag["collects_cleanly"] is True
    assert diag["import_depth"] >= 3  # pkg.a.b


def test_suspect_plugin_addopts_flagged():
    repo = _mk({
        "pkg/__init__.py": "",
        "pyproject.toml": "[tool.pytest.ini_options]\naddopts = \"--memray -q\"\n",
        "tests/test_p.py": "def test_p():\n    assert True\n",
    })
    diag = analyze_collection(repo, expected_test_ids=["tests/test_p.py::test_p"])
    assert "--memray" in diag["addopts"]
    assert any(s["option"] == "--memray" for s in diag["suspect_plugin_addopts"])


def test_external_unresolved_listed_not_fatal():
    repo = _mk({
        "pkg/__init__.py": "",
        "conftest.py": "import totally_not_installed_xyz\n",
        "tests/test_e.py": "def test_e():\n    assert True\n",
    })
    diag = analyze_collection(repo, expected_test_ids=["tests/test_e.py::test_e"])
    assert "totally_not_installed_xyz" in diag["unresolved_external"]
    # an unresolved EXTERNAL import is surfaced but does not by itself flip collects_cleanly
    assert diag["collects_cleanly"] is True


def test_submodule_import_not_flagged_as_missing_symbol():
    """REGRESSION (babel 922): `from pkg import sub` where pkg/sub.py EXISTS is a valid submodule
    import, even if `sub` is not a name in pkg/__init__.py. Must NOT be a missing_symbol / wall."""
    repo = _mk({
        "pkg/__init__.py": "# stripped: no re-exports\n",
        "pkg/support.py": "def helper():\n    return 1\n",   # submodule FILE exists
        "pkg/sub/__init__.py": "VALUE = 1\n",                # subpackage exists
        "tests/test_a.py": "from pkg import support\nfrom pkg import sub\n\ndef test_a():\n    assert True\n",
    })
    diag = analyze_collection(repo, expected_test_ids=["tests/test_a.py::test_a"])
    assert diag["collects_cleanly"] is True
    assert diag["unresolved_internal"] == []
    assert must_implement_modules(diag) == []


def test_testfile_unresolved_is_not_a_collection_wall():
    """A genuinely-missing symbol imported only by a TEST FILE is incremental work (that file collects
    once the module lands) — NOT a whole-suite wall. collects_cleanly stays True; Phase 0 must not fire."""
    repo = _mk({
        "pkg/__init__.py": "",
        "pkg/core.py": "# implementation stripped\n",          # no Widget yet, not a submodule
        "tests/test_core.py": "from pkg.core import Widget\n\ndef test_c():\n    assert True\n",
    })
    diag = analyze_collection(repo, expected_test_ids=["tests/test_core.py::test_c"])
    # the missing symbol IS recorded (informational, for blocker_class)...
    assert any(u["symbol"] == "Widget" for u in diag["unresolved_internal"])
    # ...but it is NOT a collection wall, so no Phase 0
    assert diag["collects_cleanly"] is True
    assert diag.get("conftest_unresolved") == []
    assert must_implement_modules(diag) == []


def test_conftest_unresolved_is_a_collection_wall():
    """CONTRAST: the SAME missing symbol imported by conftest.py blocks the whole suite -> wall."""
    repo = _mk({
        "pkg/__init__.py": "",
        "pkg/core.py": "# implementation stripped\n",
        "conftest.py": "from pkg.core import Widget\n",
        "tests/test_core.py": "def test_c():\n    assert True\n",
    })
    diag = analyze_collection(repo, expected_test_ids=["tests/test_core.py::test_c"])
    assert diag["collects_cleanly"] is False
    assert any(u["symbol"] == "Widget" for u in diag["conftest_unresolved"])
    assert "pkg.core" in must_implement_modules(diag)


def test_relative_import_in_conftest_is_a_wall():
    """edge (e): a RELATIVE conftest import of a stripped symbol is a genuine collection wall
    (previously skipped entirely -> false-negative)."""
    repo = _mk({
        "pkg/__init__.py": "",
        "pkg/core.py": "# stripped\n",
        "pkg/conftest.py": "from .core import GenerateSchema\n",
        "pkg/test_x.py": "def test_x():\n    assert True\n",
    })
    diag = analyze_collection(repo, expected_test_ids=["pkg/test_x.py::test_x"])
    assert diag["collects_cleanly"] is False
    assert "pkg.core" in must_implement_modules(diag)


def test_relative_import_resolves_when_symbol_present():
    """edge (e) negative: a relative import whose symbol exists is not a wall."""
    repo = _mk({
        "pkg/__init__.py": "",
        "pkg/core.py": "class GenerateSchema:\n    pass\n",
        "pkg/conftest.py": "from .core import GenerateSchema\n",
        "pkg/test_x.py": "def test_x():\n    assert True\n",
    })
    diag = analyze_collection(repo, expected_test_ids=["pkg/test_x.py::test_x"])
    assert diag["collects_cleanly"] is True
    assert diag["unresolved_internal"] == []


def test_src_layout_conftest_wall_detected():
    """edge (f): a package under src/ imports as `pkg`; a stripped conftest symbol is still a wall
    and `pkg` must NOT be misclassified as an external dependency."""
    repo = _mk({
        "src/pkg/__init__.py": "",
        "src/pkg/schema.py": "# stripped\n",
        "conftest.py": "from pkg.schema import GenerateSchema\n",
        "tests/test_x.py": "def test_x():\n    assert True\n",
    })
    diag = analyze_collection(repo, expected_test_ids=["tests/test_x.py::test_x"])
    assert diag["collects_cleanly"] is False
    assert "pkg.schema" in must_implement_modules(diag)
    assert "pkg" not in diag["unresolved_external"]


def test_src_layout_clean_collects():
    """edge (f) negative: src-layout that resolves collects cleanly."""
    repo = _mk({
        "src/pkg/__init__.py": "",
        "src/pkg/schema.py": "class GenerateSchema:\n    pass\n",
        "conftest.py": "from pkg.schema import GenerateSchema\n",
        "tests/test_x.py": "def test_x():\n    assert True\n",
    })
    diag = analyze_collection(repo, expected_test_ids=["tests/test_x.py::test_x"])
    assert diag["collects_cleanly"] is True


def test_namespace_from_pkg_import_submodule():
    """edge (g2): PEP-420 namespace package (no __init__.py) — `from pkg import core` resolves."""
    repo = _mk({
        "pkg/core.py": "VALUE = 1\n",   # no pkg/__init__.py -> namespace package
        "conftest.py": "from pkg import core\n",
        "tests/test_x.py": "def test_x():\n    assert True\n",
    })
    diag = analyze_collection(repo, expected_test_ids=["tests/test_x.py::test_x"])
    assert diag["collects_cleanly"] is True
    assert diag["unresolved_internal"] == []


def test_try_except_guarded_import_not_a_wall():
    """edge (i): an ImportError-guarded optional import must not count as a collection wall."""
    repo = _mk({
        "pkg/__init__.py": "",
        "pkg/slow.py": "X = 1\n",
        "conftest.py": "try:\n    from pkg.fast import X\nexcept ImportError:\n    from pkg.slow import X\n",
        "tests/test_x.py": "def test_x():\n    assert True\n",
    })
    diag = analyze_collection(repo, expected_test_ids=["tests/test_x.py::test_x"])
    assert diag["collects_cleanly"] is True


def test_inside_function_import_not_a_wall():
    """edge (i): a deferred import inside a function body does not run at collection -> not a wall."""
    repo = _mk({
        "pkg/__init__.py": "",
        "pkg/core.py": "# stripped\n",
        "conftest.py": "def pytest_configure(config):\n    from pkg.core import Missing  # deferred\n",
        "tests/test_x.py": "def test_x():\n    assert True\n",
    })
    diag = analyze_collection(repo, expected_test_ids=["tests/test_x.py::test_x"])
    assert diag["collects_cleanly"] is True


def test_never_raises_on_unparseable():
    repo = _mk({
        "pkg/__init__.py": "",
        "conftest.py": "def broken(:\n",  # syntax error
        "tests/test_b.py": "def test_b():\n    assert True\n",
    })
    diag = analyze_collection(repo)  # must not raise
    assert isinstance(diag, dict) and "collects_cleanly" in diag


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
