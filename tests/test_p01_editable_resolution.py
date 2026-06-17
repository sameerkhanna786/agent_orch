"""P0.1 regression: per-worktree editable resolution (src-layout false-zero fix).

v1 repo prep does ``pip install -e .`` against the BASE repo_dir, pinning the
editable importer at base/src/<pkg>. ``score_fn`` runs pytest in the candidate
WORKTREE but reused the base env, so for src-layout repos (e.g. jinja2) ``import
<pkg>`` resolved to the base STUB and the gate scored correct candidate code as
ZERO. The fix prepends ``<worktree>/src`` to a PER-CALL env's PYTHONPATH so the
worktree wins, and asserts resolution lands under the worktree (else indeterminate).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from apex_omega.eval.commit0_autogen import (
    _detect_src_pkg,
    _is_within,
    _resolve_pkg_origin,
)


def _mk_src_pkg(root: Path, pkg: str = "foo", body: str = "X = 1\n") -> Path:
    d = root / "src" / pkg
    d.mkdir(parents=True)
    (d / "__init__.py").write_text(body)
    return d


def test_detect_src_pkg_src_layout_vs_flat():
    root = Path(tempfile.mkdtemp())
    assert _detect_src_pkg(root) is None             # no src/ -> flat -> P0.1 no-op
    (root / "voluptuous").mkdir()                     # flat package at root
    (root / "voluptuous" / "__init__.py").write_text("")
    assert _detect_src_pkg(root) is None             # still flat
    _mk_src_pkg(root, "foo")
    assert _detect_src_pkg(root) == "foo"            # src/foo/__init__.py detected


def test_is_within():
    root = Path(tempfile.mkdtemp())
    inside = root / "src" / "foo" / "__init__.py"
    inside.parent.mkdir(parents=True)
    inside.write_text("")
    assert _is_within(str(inside), root)
    assert not _is_within("/etc/hostname", root)


def test_resolve_pkg_origin_worktree_wins_over_base():
    # Simulate the bug's setup: base stub + worktree edit both define `foo`; with
    # worktree/src PREPENDED to PYTHONPATH, resolution must land in the worktree.
    base = Path(tempfile.mkdtemp())
    _mk_src_pkg(base, "foo", "VALUE = 'base'\n")
    wt = Path(tempfile.mkdtemp())
    _mk_src_pkg(wt, "foo", "VALUE = 'worktree'\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(wt / "src") + os.pathsep + str(base / "src")
    origin = _resolve_pkg_origin(sys.executable, "foo", env)
    assert origin is not None
    assert _is_within(origin, wt)                     # worktree edit wins
    assert not _is_within(origin, base)               # NOT the base stub


def test_resolve_pkg_origin_missing_package_returns_none():
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    assert _resolve_pkg_origin(sys.executable, "no_such_pkg_xyz_123", env) is None
