"""Normalized Executor + capability negotiation (plan §03)."""

from __future__ import annotations

import tempfile

from apex_omega.executor import FakeExecutor, negotiate, STATIC_CAPABILITY_TABLE
from apex_omega.types import ScopedTask


def test_static_capability_table_covers_vendors():
    for v in ("codex_cli", "claude_cli", "gemini_cli", "opencode_cli"):
        assert v in STATIC_CAPABILITY_TABLE


def test_negotiate_declared_wins_and_degrades():
    p = negotiate("codex_cli", "gpt-5.5", probe=False)
    assert p.vendor == "codex_cli" and p.native_schema is True
    assert "read-only" in p.sandbox_levels
    # unknown vendor degrades to conservative no-capability profile, never crashes
    u = negotiate("nonexistent_cli", "x", probe=False)
    assert u.native_schema is False and u.sandbox_levels == ()


def test_fake_executor_roundtrip():
    fx = FakeExecutor()
    sess = fx.spawn(tempfile.mkdtemp(), "codex_cli", "m", "v1")
    res = sess.run(ScopedTask(prompt="hello", model="m", vendor="codex_cli"))
    assert res.ok and res.vendor == "codex_cli" and res.usage.total > 0
    assert fx.calls == 1


def test_fake_executor_typed_failure_never_raises():
    def boom(task, session):
        raise RuntimeError("vendor exploded")

    fx = FakeExecutor(boom)
    sess = fx.spawn(tempfile.mkdtemp(), "codex_cli", "m")
    res = sess.run(ScopedTask(prompt="x"))
    assert res.ok is False and res.finalization_status == "infra_nonresult"
    assert "vendor exploded" in (res.error or "")
