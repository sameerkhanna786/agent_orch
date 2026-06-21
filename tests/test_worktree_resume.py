"""NEW-I1/I4/I8/I9: WorktreeProvider resume-hardening + apply conflict-marker guard.

These guard the lossless-resume foundation (the 2220->13 cluster): a crash/kill must leave a
re-acquirable cell (stale lock from a dead pid reclaimed, stale wt_<rid> hard-cleared) and a 3way
apply that leaves conflict markers must NOT be reported as a clean apply (poisoned-tree scoring).
"""
import os
import subprocess

import pytest

from apex_omega.isolation.worktree import (
    WorktreeProvider,
    apply_diff,
    apply_diff_partial,
    _diff_target_files,
    _has_conflict_markers,
)


def _git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def _init_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", cwd=path)
    _git("config", "user.email", "t@t", cwd=path)
    _git("config", "user.name", "t", cwd=path)
    (path / "f.py").write_text("x = 1\n")
    _git("add", "-A", cwd=path)
    _git("commit", "-qm", "base", cwd=path)


# --- conflict-marker helpers (NEW-I3/I9) --------------------------------------

def test_diff_target_files_extracts_b_side_skips_devnull():
    diff = (
        "diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-x\n+y\n"
        "diff --git a/gone.py b/gone.py\n--- a/gone.py\n+++ /dev/null\n@@ -1 +0 @@\n-z\n"
    )
    assert _diff_target_files(diff) == {"foo.py"}


def test_has_conflict_markers_detects_full_triplet(tmp_path):
    poisoned = tmp_path / "c.py"
    poisoned.write_text("a\n<<<<<<< HEAD\nb\n=======\nc\n>>>>>>> other\n")
    clean = tmp_path / "d.py"
    clean.write_text("a = 1\n")
    assert _has_conflict_markers(str(tmp_path), {"c.py"}) is True
    assert _has_conflict_markers(str(tmp_path), {"d.py"}) is False
    assert _has_conflict_markers(str(tmp_path), {"missing.py"}) is False


def test_apply_diff_clean_strict_apply(tmp_path):
    repo = tmp_path / "r"
    _init_repo(repo)
    diff = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
    assert apply_diff(str(repo), diff) is True
    assert (repo / "f.py").read_text() == "x = 2\n"


def test_apply_diff_partial_returns_dict_shape(tmp_path):
    """reduce_residuals depends on the dict {clean, applied_any, rejected_hunks} interface."""
    repo = tmp_path / "r"
    _init_repo(repo)
    diff = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x = 1\n+x = 9\n"
    r = apply_diff_partial(str(repo), diff)
    assert isinstance(r, dict)
    assert r["clean"] is True and r["applied_any"] is True and r["rejected_hunks"] == 0


def test_apply_diff_partial_empty_is_clean_noop(tmp_path):
    repo = tmp_path / "r"
    _init_repo(repo)
    r = apply_diff_partial(str(repo), "")
    assert r == {"clean": True, "applied_any": False, "rejected_hunks": 0}


# --- resume-hardening (NEW-I1/I4/I8) ------------------------------------------

def _provider(tmp_path):
    src = tmp_path / "src"
    _init_repo(src)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / ".locks").mkdir(parents=True, exist_ok=True)
    return WorktreeProvider(source_repo=str(src), workspace_dir=ws, run_scope="t")


def test_acquire_reclaims_stale_worktree_dir(tmp_path):
    """NEW-I1/I4: a leftover wt_<rid> orphan (crash) must be hard-cleared so acquire succeeds."""
    prov = _provider(tmp_path)
    h = prov.acquire("7")
    prov.release(h, confirm_patch_extracted=True)
    # simulate a crash orphan: a stale dir where the worktree path would be
    orphan = prov.workspace_dir / "wt_7"
    orphan.mkdir(parents=True, exist_ok=True)
    (orphan / "junk").write_text("stale")
    h2 = prov.acquire("7")  # must reclaim, not raise
    assert os.path.isdir(h2.path)
    prov.release(h2, confirm_patch_extracted=True)


def test_acquire_reclaims_lock_from_dead_pid(tmp_path):
    """NEW-I8: a lock file stamped with a dead pid must be reclaimable on resume."""
    prov = _provider(tmp_path)
    lock_path = prov._lock_path("9")
    # stamp a guaranteed-dead pid (no flock held -> simulates a crashed holder)
    with open(lock_path, "w") as fh:
        fh.write("999999")
    h = prov.acquire("9")  # dead-pid marker -> reclaimed
    assert os.path.isdir(h.path)
    prov.release(h, confirm_patch_extracted=True)


def test_marker_pid_alive_classifies(tmp_path):
    prov = _provider(tmp_path)
    fd = os.open(str(prov._lock_path("m")), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        os.write(fd, str(os.getpid()).encode())  # our own pid -> alive
        assert prov._marker_pid_alive(fd) is True
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, b"999999")  # dead pid -> stale
        assert prov._marker_pid_alive(fd) is False
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, b"")  # empty -> stale
        assert prov._marker_pid_alive(fd) is False
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
