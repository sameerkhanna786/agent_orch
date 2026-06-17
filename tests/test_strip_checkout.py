"""Regression guard for the ladder runner's disk-stripping pass.

The bug: ``_strip_checkout`` walked a finished cell's rundir and did
``Path.chmod(0o600)`` on every non-evidence file BEFORE unlinking it. A cell's
runtime venv contains ``runtime/.venv/bin/python*`` symlinks that point INTO the
SHARED uv-managed interpreter. ``Path.chmod`` FOLLOWS symlinks, so stripping the
first finished cell set the real interpreter to 0o600 (no execute), and every
later cell died with::

    RuntimeError: Failed to query Python interpreter at .../python3.10
      Caused by: Permission denied (os error 13)

This was concurrency-amplified (the first cell to finish poisoned the rest) and
recurring (a static ``chmod +x`` only patched the symptom). The fix: never chmod
a symlink — unlink the link itself, leaving the target untouched.
"""

from __future__ import annotations

import importlib.util
import stat
import tempfile
from pathlib import Path

_RUN_LADDER = Path(__file__).resolve().parents[1] / "scripts" / "run_ladder.py"


def _load_run_ladder():
    spec = importlib.util.spec_from_file_location("run_ladder", _RUN_LADDER)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _is_exec(p: Path) -> bool:
    return bool(p.stat().st_mode & stat.S_IXUSR)


def test_strip_checkout_does_not_clobber_symlinked_shared_interpreter():
    rl = _load_run_ladder()
    root = Path(tempfile.mkdtemp())

    # A SHARED interpreter living OUTSIDE the cell rundir (~/.local/share/uv/...).
    shared_bin = root / "shared_uv" / "bin"
    shared_bin.mkdir(parents=True)
    interp = shared_bin / "python3.10"
    interp.write_text("#!/bin/sh\necho hi\n")
    interp.chmod(0o755)
    assert _is_exec(interp)

    # A cell rundir whose runtime venv symlinks INTO the shared interpreter,
    # alongside a bulky non-evidence file and an evidence file.
    rundir = root / "cell"
    venv_bin = rundir / "runtime" / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    link = venv_bin / "python3.10"
    link.symlink_to(interp)                       # the dangerous symlink
    bulky = rundir / "src" / "module.so"
    bulky.parent.mkdir(parents=True)
    bulky.write_bytes(b"\x00" * 1024)
    evidence = rundir / "benchmark_report.json"
    evidence.write_text('{"solved_tasks": 1}')

    rl._strip_checkout(rundir)

    # The shared interpreter must survive UNTOUCHED — present AND still executable.
    assert interp.exists(), "shared interpreter was deleted via its symlink"
    assert _is_exec(interp), (
        "shared interpreter lost its execute bit — chmod followed the symlink"
    )
    # The symlink itself is gone; bulky file gone; evidence kept.
    assert not link.exists() and not link.is_symlink()
    assert not bulky.exists()
    assert evidence.exists()


def test_recover_checkpoint_finds_accepted_solve():
    # Tier-1.1: run_ladder recovers a verified solve from the acceptance checkpoint
    # when a cell is killed (subprocess timeout) before banking its winner.
    import json
    rl = _load_run_ladder()
    root = Path(tempfile.mkdtemp())
    assert rl._recover_checkpoint(root) is None                 # no checkpoint -> None
    sub = root / "cells" / "autogen_mimesis"
    sub.mkdir(parents=True)
    (sub / "accepted_checkpoint.json").write_text(json.dumps({"accepted": True, "candidate_id": "r3"}))
    rec = rl._recover_checkpoint(root)
    assert rec and rec["candidate_id"] == "r3"                  # nested accepted checkpoint found
    root2 = Path(tempfile.mkdtemp())
    (root2 / "accepted_checkpoint.json").write_text(json.dumps({"accepted": False}))
    assert rl._recover_checkpoint(root2) is None                # non-accepted ignored


def test_strip_checkout_keeps_evidence_and_removes_bulk():
    rl = _load_run_ladder()
    rundir = Path(tempfile.mkdtemp()) / "cell"
    rundir.mkdir(parents=True)
    keep = []
    for name in ("report.json", "narration.md", "fix.diff", "out.log", "x.patch", "n.txt", "j.jsonl"):
        p = rundir / name
        p.write_text("x")
        keep.append(p)
    drop = []
    for name in ("a.so", "b.pyc", "big.bin", "mod.py"):
        p = rundir / name
        p.write_text("x")
        drop.append(p)

    rl._strip_checkout(rundir)

    for p in keep:
        assert p.exists(), f"evidence {p.name} was wrongly deleted"
    for p in drop:
        assert not p.exists(), f"bulky {p.name} was not stripped"
