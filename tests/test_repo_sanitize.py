"""Upstream/version breadcrumb scrub for prepared commit0 repos."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from apex_omega.eval.repo_sanitize import (
    PLACEHOLDER_URL,
    PLACEHOLDER_VERSION,
    scrub_upstream_identifiers,
)


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


def _make_repo(*, test_pins_version: bool = False) -> Path:
    d = Path(tempfile.mkdtemp()) / "mimesis"
    (d / "mimesis").mkdir(parents=True)
    (d / "tests").mkdir()
    (d / "docs").mkdir()
    (d / "setup.py").write_text(
        "from setuptools import setup\n"
        "setup(name='mimesis', version='17.0.0',\n"
        "      url='https://github.com/lk-geimfari/mimesis',\n"
        "      api_version='2')\n"  # must NOT be touched (word boundary)
    )
    (d / "pyproject.toml").write_text(
        "[project]\nname = \"mimesis\"\nversion = \"17.0.0\"\n"
        "[project.urls]\nHomepage = \"https://pypi.org/project/mimesis/\"\n"
        "Docs = \"https://mimesis.readthedocs.io/\"\n"
    )
    (d / "mimesis" / "__init__.py").write_text(
        "__version__ = '17.0.0'\nminimum_version = '3'\n"
    )
    (d / "mimesis" / "_version.py").write_text("VERSION = \"17.0.0\"\n")
    (d / "README.md").write_text(
        "# mimesis 17.0.0\nInstall: pip install mimesis\n"
        "Source: https://github.com/lk-geimfari/mimesis\n"
    )
    (d / "docs" / "conf.py").write_text("version = '17.0'\nrelease = '17.0.0'\n")
    (d / "CHANGELOG.md").write_text("# 17.0.0\n- stuff\n# 16.0.0\n- old\n")
    egg = d / "mimesis.egg-info"
    egg.mkdir()
    (egg / "PKG-INFO").write_text("Metadata-Version: 2.1\nName: mimesis\nVersion: 17.0.0\n")
    if test_pins_version:
        (d / "tests" / "test_version.py").write_text(
            "from mimesis import __version__\n"
            "def test_v():\n    assert __version__ == '17.0.0'\n"
        )
    else:
        (d / "tests" / "test_core.py").write_text("def test_x():\n    assert True\n")

    _git(["init", "-q"], d)
    _git(["config", "user.email", "a@b.c"], d)
    _git(["config", "user.name", "x"], d)
    _git(["add", "-A"], d)
    _git(["commit", "-qm", "base"], d)
    _git(["checkout", "-qB", "apex-base"], d)
    _git(["tag", "v17.0.0"], d)
    _git(["tag", "v16.0.0"], d)
    _git(["remote", "add", "origin", "https://github.com/lk-geimfari/mimesis.git"], d)
    return d


def test_scrub_removes_version_and_upstream_tells():
    d = _make_repo()
    rep = scrub_upstream_identifiers(d)

    setup = (d / "setup.py").read_text()
    assert "17.0.0" not in setup and f"version='{PLACEHOLDER_VERSION}'" in setup
    assert "api_version='2'" in setup                      # word-boundary safe
    assert "github.com" not in setup and PLACEHOLDER_URL in setup

    pyproject = (d / "pyproject.toml").read_text()
    assert "17.0.0" not in pyproject
    assert "pypi.org" not in pyproject and "readthedocs.io" not in pyproject

    init = (d / "mimesis" / "__init__.py").read_text()
    assert "17.0.0" not in init and PLACEHOLDER_VERSION in init
    assert "minimum_version = '3'" in init                 # not clobbered
    assert "17.0.0" not in (d / "mimesis" / "_version.py").read_text()
    assert "17.0.0" not in (d / "docs" / "conf.py").read_text()

    assert not (d / "CHANGELOG.md").exists()
    assert not (d / "mimesis.egg-info").exists()

    # git tags + remote gone
    assert _git(["tag"], d).stdout.strip() == ""
    assert _git(["remote"], d).stdout.strip() == ""
    assert rep["tags_removed"] == 2 and rep["remotes_removed"] == ["origin"]
    assert rep["committed"] is True and rep["kept_package_version"] is False


def test_scrub_committed_onto_base_so_worktrees_inherit():
    d = _make_repo()
    scrub_upstream_identifiers(d)
    # a fresh worktree forked from apex-base must NOT contain the version tell
    wt = Path(tempfile.mkdtemp()) / "wt"
    res = _git(["worktree", "add", "--detach", str(wt), "apex-base"], d)
    assert res.returncode == 0, res.stderr
    assert "17.0.0" not in (wt / "setup.py").read_text()
    assert "17.0.0" not in (wt / "mimesis" / "__init__.py").read_text()
    assert not (wt / "CHANGELOG.md").exists()


def test_scrub_keeps_package_version_when_a_test_pins_it():
    # If a visible test asserts __version__, the exact version is part of the spec:
    # keep the package version literal, but still remove upstream locators + timeline.
    d = _make_repo(test_pins_version=True)
    rep = scrub_upstream_identifiers(d)
    assert rep["kept_package_version"] is True
    assert "__version__ = '17.0.0'" in (d / "mimesis" / "__init__.py").read_text()
    # locators + timeline still scrubbed
    assert "github.com" not in (d / "setup.py").read_text()
    assert not (d / "CHANGELOG.md").exists()
    assert _git(["tag"], d).stdout.strip() == ""


def test_scrub_idempotent_and_safe_on_rerun():
    d = _make_repo()
    scrub_upstream_identifiers(d)
    rep2 = scrub_upstream_identifiers(d)            # second pass: nothing left to do
    assert rep2["committed"] is False               # no changes -> no empty commit
    assert "17.0.0" not in (d / "setup.py").read_text()


def test_scrub_removes_nonbase_branches_and_reflog():
    # P0.3: the upstream solution can hide in a stale branch / the reflog even after
    # tags + remotes are stripped. Keep ONLY the eval base branch; expire the reflog.
    d = _make_repo()                                 # current branch == apex-base
    _git(["branch", "leak-branch"], d)               # a stale non-base branch
    before = _git(["for-each-ref", "--format=%(refname:short)", "refs/heads/"], d).stdout.split()
    assert "leak-branch" in before
    rep = scrub_upstream_identifiers(d)
    after = _git(["for-each-ref", "--format=%(refname:short)", "refs/heads/"], d).stdout.split()
    assert "leak-branch" not in after                # stale branch deleted
    assert "apex-base" in after                      # base branch preserved
    assert rep["branches_removed"] >= 1
    assert rep["reflog_scrubbed"] is True
