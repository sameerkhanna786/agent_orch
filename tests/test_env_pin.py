"""F1: pydantic-core <-> pydantic ABI pin guard.

A stale ``pydantic_core`` (e.g. 2.20.1 left behind by a partial venv rebuild)
against a newer ``pydantic`` crashes ``import datasets`` in v1 repo prep, taking
down EVERY commit0 eval cell before a single agent runs. The fix has two halves:

  1. a PERSISTED pin in requirements.txt (pydantic-core==2.46.1), and
  2. a loud pre-run CHECK (apex_omega.eval.commit0_autogen.check_pydantic_core_compat
     + its pure core _pydantic_core_verdict) that fails fast with a repin message.

These tests cover the check helper (mismatch -> raises; match -> ok) and assert the
requirements.txt pin stays in lockstep with the in-code pin. The check is a CHECK
ONLY — it never installs at runtime, so nothing here touches the environment.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from apex_omega.eval.commit0_autogen import (
    PYDANTIC_CORE_PIN,
    SKIP_PYDANTIC_CHECK_ENV,
    _pydantic_core_verdict,
    check_pydantic_core_compat,
)

_ROOT = Path(__file__).resolve().parents[1]


# --- the pure compare core (testable without pydantic installed) ---------------

def test_verdict_match_ok():
    ok, detail = _pydantic_core_verdict("2.46.1", "2.11.7", "2.46.1")
    assert ok is True
    assert "OK" in detail
    assert "2.46.1" in detail


def test_verdict_mismatch_raises_loud_with_repin_instruction():
    with pytest.raises(RuntimeError) as exc:
        _pydantic_core_verdict("2.20.1", "2.11.7", "2.46.1", fail_loud=True)
    msg = str(exc.value)
    # actionable: names both versions, the crash site, and how to repin
    assert "2.20.1" in msg and "2.46.1" in msg
    assert "import datasets" in msg
    assert "pydantic-core==2.46.1" in msg
    assert SKIP_PYDANTIC_CHECK_ENV in msg


def test_verdict_mismatch_soft_returns_false_no_raise():
    ok, detail = _pydantic_core_verdict("2.20.1", "2.11.7", "2.46.1", fail_loud=False)
    assert ok is False
    assert "MISMATCH" in detail


def test_verdict_mismatch_bypassed_by_env(monkeypatch):
    monkeypatch.setenv(SKIP_PYDANTIC_CHECK_ENV, "1")
    # even with fail_loud, the env bypass downgrades a mismatch to a soft (False, ...)
    ok, detail = _pydantic_core_verdict("2.20.1", "2.11.7", "2.46.1", fail_loud=True)
    assert ok is False
    assert "MISMATCH" in detail


def test_verdict_defaults_required_to_in_code_pin():
    # required=None falls back to PYDANTIC_CORE_PIN; installed==pin -> ok
    ok, _ = _pydantic_core_verdict(PYDANTIC_CORE_PIN, "2.11.7", None)
    assert ok is True


def test_verdict_unknown_installed_is_mismatch():
    ok, detail = _pydantic_core_verdict(None, None, "2.46.1", fail_loud=False)
    assert ok is False
    assert "unknown" in detail


# --- the public helper, exercised against the in-code pin ----------------------

def test_check_compat_matches_when_pin_satisfied(monkeypatch):
    """expected_pin lets us drive the helper deterministically regardless of what
    pydantic (if any) is installed: install==expected -> ok. We force this by
    pinning expected to the *installed* core when pydantic is present, else skip."""
    try:
        import pydantic_core  # noqa: F401
    except Exception:
        pytest.skip("pydantic not installed in this venv; covered by verdict tests")
    installed = pydantic_core.__version__
    ok, detail = check_pydantic_core_compat(expected_pin=installed)
    assert ok is True
    assert installed in detail


def test_check_compat_mismatch_raises(monkeypatch):
    try:
        import pydantic_core  # noqa: F401
    except Exception:
        pytest.skip("pydantic not installed in this venv; covered by verdict tests")
    monkeypatch.delenv(SKIP_PYDANTIC_CHECK_ENV, raising=False)
    with pytest.raises(RuntimeError):
        check_pydantic_core_compat(expected_pin="0.0.0-definitely-wrong")


# --- the persisted pin (requirements.txt) stays in lockstep --------------------

def test_requirements_pins_pydantic_core_to_in_code_pin():
    req = (_ROOT / "requirements.txt").read_text(encoding="utf-8")
    m = re.search(r"^pydantic-core==(\S+)\s*$", req, flags=re.MULTILINE)
    assert m is not None, "requirements.txt must pin pydantic-core==<version>"
    assert m.group(1) == PYDANTIC_CORE_PIN, (
        f"requirements.txt pins pydantic-core=={m.group(1)} but the in-code "
        f"PYDANTIC_CORE_PIN is {PYDANTIC_CORE_PIN}; keep them in lockstep")


def test_requirements_declares_compatible_pydantic():
    req = (_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"^pydantic[><=!~]", req, flags=re.MULTILINE), (
        "requirements.txt must declare a pydantic version constraint alongside "
        "the pydantic-core pin")
