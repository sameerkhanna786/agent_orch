"""Auth preflight: refresh + validate vendor auth BEFORE a run dispatches agents, so a stale
gateway token can never silently burn a whole run on 0-token infra_nonresult results."""

from __future__ import annotations

import pytest

from apex_omega.executor import auth_env


def test_refresh_skips_unknown_vendor_gracefully():
    ok, detail = auth_env.refresh_vendor_auth("nonexistent_vendor_xyz")
    assert ok is True and "skipped" in detail          # no binary -> graceful skip, never fatal


def test_preflight_fails_loud_on_broken_auth(monkeypatch):
    # a vendor that cannot produce a result aborts the run (no wasted burn) ...
    monkeypatch.setattr(auth_env, "refresh_vendor_auth", lambda v, **k: (False, "stale token"))
    monkeypatch.delenv("APEX_OMEGA_SKIP_AUTH_PREFLIGHT", raising=False)
    with pytest.raises(RuntimeError) as ei:
        auth_env.preflight_vendor_auth(["codex_cli"])
    assert "preflight FAILED" in str(ei.value) and "codex_cli" in str(ei.value)


def test_preflight_skip_env_bypasses_failure(monkeypatch):
    # ... unless the operator explicitly opts out.
    monkeypatch.setattr(auth_env, "refresh_vendor_auth", lambda v, **k: (False, "stale token"))
    monkeypatch.setenv("APEX_OMEGA_SKIP_AUTH_PREFLIGHT", "1")
    res = auth_env.preflight_vendor_auth(["codex_cli"])
    assert res["codex_cli"][0] is False                # surfaced, but not fatal


def test_preflight_passes_when_all_ok(monkeypatch):
    monkeypatch.setattr(auth_env, "refresh_vendor_auth", lambda v, **k: (True, "ok"))
    res = auth_env.preflight_vendor_auth(["codex_cli", "codex_cli", "claude_cli"])
    assert set(res) == {"codex_cli", "claude_cli"} and all(ok for ok, _ in res.values())
