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
