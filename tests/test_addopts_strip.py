"""O4/NEW-I7: scoring-time strip of unloadable plugin addopts options.

Commit0 gold scoring runs ``task.test_cmd`` inside the repo with
``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1``. Pytest applies the repo's config
``addopts`` at runtime, so a plugin-provided option there (e.g. pydantic's
``--memray``) makes pytest exit rc=4 ("unrecognized arguments") BEFORE
collection unless its plugin is importable. The fix strips only the unloadable
plugin options (keeping core pytest options and loadable-plugin options) and
overrides addopts via ``-o addopts=...``.

These tests pin the pure strip/parse helpers so the behavior is deterministic
regardless of which plugins happen to be installed in the test venv.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from apex.evaluation.commit0_benchmark import (
    _addopts_option_plugin_module,
    _coerce_addopts_to_tokens,
    _read_repo_addopts_tokens,
    _strip_unimportable_plugin_addopts,
)


def _never_importable(_module: str) -> bool:
    return False


def _always_importable(_module: str) -> bool:
    return True


def _importable_only(*modules: str):
    allowed = set(modules)
    return lambda module: module in allowed


# ---------------------------------------------------------------------------
# Core behavior: --memray strips iff pytest-memray is not importable.
# ---------------------------------------------------------------------------


def test_memray_stripped_when_pytest_memray_not_importable():
    kept, stripped = _strip_unimportable_plugin_addopts(
        ["--memray"],
        is_module_importable=_never_importable,
    )
    assert kept == []
    assert stripped == ["--memray"]


def test_memray_kept_when_pytest_memray_importable():
    kept, stripped = _strip_unimportable_plugin_addopts(
        ["--memray"],
        is_module_importable=_importable_only("pytest_memray"),
    )
    assert kept == ["--memray"]
    assert stripped == []


# ---------------------------------------------------------------------------
# Loadable plugin option is kept; core options are never stripped.
# ---------------------------------------------------------------------------


def test_loadable_plugin_option_is_kept_while_unloadable_is_stripped():
    # pytest_cov importable, pytest_memray not.
    kept, stripped = _strip_unimportable_plugin_addopts(
        ["--cov=pkg", "--memray"],
        is_module_importable=_importable_only("pytest_cov"),
    )
    assert kept == ["--cov=pkg"]
    assert stripped == ["--memray"]


def test_core_pytest_options_never_stripped_even_when_nothing_importable():
    core = ["-q", "-ra", "--strict-markers", "--strict-config", "--tb=short", "-x"]
    kept, stripped = _strip_unimportable_plugin_addopts(
        list(core),
        is_module_importable=_never_importable,
    )
    assert kept == core
    assert stripped == []


def test_core_options_kept_alongside_stripped_plugin_option():
    kept, stripped = _strip_unimportable_plugin_addopts(
        ["--strict-markers", "--memray", "-q"],
        is_module_importable=_never_importable,
    )
    assert kept == ["--strict-markers", "-q"]
    assert stripped == ["--memray"]


# ---------------------------------------------------------------------------
# Detached values are stripped along with their option.
# ---------------------------------------------------------------------------


def test_strips_detached_value_for_unloadable_option():
    # -n auto from pytest-xdist; xdist not importable -> drop both tokens.
    kept, stripped = _strip_unimportable_plugin_addopts(
        ["-n", "auto", "-q"],
        is_module_importable=_never_importable,
    )
    assert kept == ["-q"]
    assert stripped == ["-n", "auto"]


def test_inline_value_form_is_stripped_as_single_token():
    kept, stripped = _strip_unimportable_plugin_addopts(
        ["--cov=mypkg", "-q"],
        is_module_importable=_never_importable,
    )
    assert kept == ["-q"]
    assert stripped == ["--cov=mypkg"]


def test_does_not_consume_following_option_as_value():
    # --timeout takes a value, but the next token is itself an option, so it is a
    # value-less use and the following option must survive.
    kept, stripped = _strip_unimportable_plugin_addopts(
        ["--timeout", "--strict-markers"],
        is_module_importable=_never_importable,
    )
    assert kept == ["--strict-markers"]
    assert stripped == ["--timeout"]


def test_unknown_options_are_treated_as_core_and_kept():
    # An option not in the plugin map (e.g. a repo-local conftest option) must
    # never be stripped, regardless of importability.
    kept, stripped = _strip_unimportable_plugin_addopts(
        ["--my-repo-flag", "value", "--memray"],
        is_module_importable=_never_importable,
    )
    assert kept == ["--my-repo-flag", "value"]
    assert stripped == ["--memray"]


def test_empty_addopts_is_noop():
    kept, stripped = _strip_unimportable_plugin_addopts(
        [],
        is_module_importable=_never_importable,
    )
    assert kept == []
    assert stripped == []


def test_all_plugins_loadable_strips_nothing():
    addopts = ["--memray", "--cov=pkg", "--timeout", "30", "-q"]
    kept, stripped = _strip_unimportable_plugin_addopts(
        list(addopts),
        is_module_importable=_always_importable,
    )
    assert kept == addopts
    assert stripped == []


def test_importability_probed_once_per_module():
    calls: list[str] = []

    def counting(module: str) -> bool:
        calls.append(module)
        return False

    # Two pytest-memray options -> module probed once (memoized).
    _strip_unimportable_plugin_addopts(
        ["--memray", "--native"],
        is_module_importable=counting,
    )
    assert calls.count("pytest_memray") == 1


# ---------------------------------------------------------------------------
# _addopts_option_plugin_module: maps plugin options, leaves core/unknown alone.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token,expected_module",
    [
        ("--memray", "pytest_memray"),
        ("--cov", "pytest_cov"),
        ("--cov=pkg", "pytest_cov"),
        ("--timeout", "pytest_timeout"),
        ("--reruns", "pytest_rerunfailures"),
        ("-n", "xdist"),
    ],
)
def test_plugin_option_maps_to_module(token, expected_module):
    assert _addopts_option_plugin_module(token) == expected_module


@pytest.mark.parametrize(
    "token",
    ["-q", "--strict-markers", "--tb=short", "-x", "--my-repo-flag", "value", "pkg/tests"],
)
def test_core_and_unknown_options_map_to_none(token):
    assert _addopts_option_plugin_module(token) is None


# ---------------------------------------------------------------------------
# _coerce_addopts_to_tokens: handles str + list forms.
# ---------------------------------------------------------------------------


def test_coerce_string_addopts_to_tokens():
    assert _coerce_addopts_to_tokens("--memray -q --cov=pkg") == [
        "--memray",
        "-q",
        "--cov=pkg",
    ]


def test_coerce_list_addopts_to_tokens():
    assert _coerce_addopts_to_tokens(["--memray", "-q"]) == ["--memray", "-q"]


def test_coerce_non_string_returns_empty():
    assert _coerce_addopts_to_tokens(None) == []
    assert _coerce_addopts_to_tokens(123) == []


# ---------------------------------------------------------------------------
# _read_repo_addopts_tokens: reads from each supported config file.
# ---------------------------------------------------------------------------


def test_read_addopts_from_pyproject(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [tool.pytest.ini_options]
            addopts = "--memray -q"
            """
        ),
        encoding="utf-8",
    )
    assert _read_repo_addopts_tokens(tmp_path) == ["--memray", "-q"]


def test_read_addopts_list_form_from_pyproject(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [tool.pytest.ini_options]
            addopts = ["--memray", "-ra"]
            """
        ),
        encoding="utf-8",
    )
    assert _read_repo_addopts_tokens(tmp_path) == ["--memray", "-ra"]


def test_read_addopts_from_pytest_ini(tmp_path: Path):
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\naddopts = --memray -q\n",
        encoding="utf-8",
    )
    assert _read_repo_addopts_tokens(tmp_path) == ["--memray", "-q"]


def test_read_addopts_from_setup_cfg(tmp_path: Path):
    (tmp_path / "setup.cfg").write_text(
        "[tool:pytest]\naddopts = --memray -q\n",
        encoding="utf-8",
    )
    assert _read_repo_addopts_tokens(tmp_path) == ["--memray", "-q"]


def test_pyproject_addopts_takes_precedence_over_ini(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\naddopts = \"--from-pyproject\"\n",
        encoding="utf-8",
    )
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\naddopts = --from-ini\n",
        encoding="utf-8",
    )
    assert _read_repo_addopts_tokens(tmp_path) == ["--from-pyproject"]


def test_no_config_returns_empty(tmp_path: Path):
    assert _read_repo_addopts_tokens(tmp_path) == []


def test_pyproject_without_addopts_returns_empty(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\nminversion = \"7.0\"\n",
        encoding="utf-8",
    )
    assert _read_repo_addopts_tokens(tmp_path) == []


# ---------------------------------------------------------------------------
# End-to-end on the read->strip path that the scoring command uses.
# ---------------------------------------------------------------------------


def test_pydantic_like_addopts_strips_memray_when_plugin_missing(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [tool.pytest.ini_options]
            addopts = "--benchmark-columns min,max,mean --strict-config --strict-markers --memray"
            """
        ),
        encoding="utf-8",
    )
    tokens = _read_repo_addopts_tokens(tmp_path)
    # benchmark loadable, memray not.
    kept, stripped = _strip_unimportable_plugin_addopts(
        tokens,
        is_module_importable=_importable_only("pytest_benchmark"),
    )
    assert stripped == ["--memray"]
    assert kept == [
        "--benchmark-columns",
        "min,max,mean",
        "--strict-config",
        "--strict-markers",
    ]
