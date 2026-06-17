"""Vendor-CLI auth/env provisioning for the Meta gateway (host mode).

apex's launcher points the vendor CLIs at the Meta plugboard gateway when running
on a Meta host. We replicate the *minimal* env so APEX-Ω's in-process (Mode C) and
subprocess (Mode A) workers authenticate the same way apex does:

  * CODEX_BASE_URL -> plugboard (codex `responses` provider endpoint)
  * OPENAI_API_KEY -> a dummy (plugboard handles real auth via x2p network identity)
  * APEX_TARGET_RUNTIME_CLI_AUTH_MODE=host_cli -> so codex's provider `-c` config lands
    on the HOST command (otherwise it routes to the container path and codex reports
    "Model provider `responses` not found").

Everything is ``setdefault`` — an operator's real environment always wins, so this is
safe in a non-Meta env (just set your own CODEX_BASE_URL / keys, or none for the CLIs
that self-auth like gemini/opencode). Verified to make codex authenticate in the
Claude Code sandbox (gemini + opencode self-auth with no env at all).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Iterable, MutableMapping, Optional

# The Meta gateway codex endpoint host (normalized to plugboardv2 internally by apex).
PLUGBOARD_BASE_URL = "http://plugboard.x2p.facebook.net"
_DUMMY_OPENAI_KEY = "apex-omega-proxy-dummy"

# Per-vendor model names APEX's config actually accepts (SUPPORTED_MODELS_BY_BACKEND).
DEFAULT_MODELS = {
    "codex_cli": "gpt-5.5",
    "claude_cli": "opus",
    "gemini_cli": "gemini-3.1-pro",       # NOT gemini-2.5-pro (apex rejects it)
    "opencode_cli": "meta/avocado-tester",
    "metacode_cli": "meta/avocado-tester",
}


def ensure_vendor_auth_env(env: Optional[MutableMapping[str, str]] = None) -> MutableMapping[str, str]:
    """setdefault the Meta-gateway host-mode auth env onto ``env`` (or os.environ).
    Idempotent; never overrides values the operator already set."""
    target = env if env is not None else os.environ
    target.setdefault("CODEX_BASE_URL", PLUGBOARD_BASE_URL)
    target.setdefault("OPENAI_API_KEY", _DUMMY_OPENAI_KEY)
    target.setdefault("APEX_TARGET_RUNTIME_CLI_AUTH_MODE", "host_cli")
    return target


def default_model(vendor: str) -> str:
    return DEFAULT_MODELS.get(vendor, "gpt-5.5")


# The vendor CLI binary for each backend (mirrors capability._CLI_COMMAND).
_CLI_BINARY = {
    "codex_cli": "codex", "claude_cli": "claude", "gemini_cli": "gemini",
    "opencode_cli": "opencode", "metacode_cli": "metacode",
}

# Process-local memo of vendors whose auth has been warmed this process (so a cell-subprocess
# probes once, not per-agent). Keyed by vendor.
_WARMED: set[str] = set()


def _probe_argv(cli: str) -> list[str]:
    """A TRIVIAL model call that warms/refreshes the host gateway auth (or a version probe for
    self-auth CLIs). The codex/claude variants make a real model call — that is the point: it
    re-establishes the gateway token so the run's sandboxed agents don't hit a stale-auth wall."""
    if cli == "codex":
        # Run UNSANDBOXED + against the HOST codex home (no CODEX_HOME override) so it warms the
        # same host auth the sandboxed rollout agents then reuse. --skip-git-repo-check lets it run
        # in a throwaway non-git cwd ("Not inside a trusted directory" otherwise). Keep it tiny.
        return [cli, "exec", "--dangerously-disable-osx-sandbox", "--skip-git-repo-check",
                "Reply with the single word READY and nothing else."]
    if cli == "claude":
        return [cli, "-p", "Reply with the single word READY and nothing else."]
    return [cli, "--version"]   # best-effort for self-auth CLIs (gemini/opencode/metacode)


def refresh_vendor_auth(vendor: str = "codex_cli", *, timeout: int = 120,
                        force: bool = False) -> tuple[bool, str]:
    """Refresh + validate a vendor CLI's auth-state BEFORE a run dispatches agents, so a stale
    gateway token can never silently yield 0-token ``infra_nonresult`` for the whole run. Makes a
    trivial model call (which re-establishes the host auth) and checks it produced output. Returns
    ``(ok, detail)``. Memoized per-process per vendor unless ``force=True``. Never raises."""
    ensure_vendor_auth_env()
    key = str(vendor)
    if not force and key in _WARMED:
        return True, "already warmed this process"
    cli = _CLI_BINARY.get(key)
    if not cli or not shutil.which(cli):
        return True, f"no auth probe for vendor {key!r} (skipped)"
    try:
        # Run in a throwaway cwd so the probe never pollutes the repo (codex may touch ./.codex);
        # close stdin so codex doesn't block reading additional input.
        proc = subprocess.run(_probe_argv(cli), capture_output=True, text=True, timeout=timeout,
                              cwd=tempfile.mkdtemp(), env=dict(os.environ),
                              stdin=subprocess.DEVNULL)
    except Exception as exc:                       # noqa: BLE001 - typed-failure contract
        return False, f"{type(exc).__name__}: {exc}"
    out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    ok = proc.returncode == 0 and bool(out)
    if ok:
        _WARMED.add(key)
        return True, "ok"
    return False, f"rc={proc.returncode}; {out[-400:]}"


def preflight_vendor_auth(vendors: Iterable[str], *, fail_loud: bool = True,
                          timeout: int = 120) -> dict[str, tuple[bool, str]]:
    """Warm + validate auth for each distinct vendor before a run begins. With ``fail_loud`` (and
    no ``APEX_OMEGA_SKIP_AUTH_PREFLIGHT=1`` override) a vendor that cannot produce a result aborts
    the run with a clear RuntimeError instead of letting it burn on 0-token non-results. Returns
    ``{vendor: (ok, detail)}``."""
    results: dict[str, tuple[bool, str]] = {}
    for v in dict.fromkeys(vendors):               # distinct, order-preserving
        results[v] = refresh_vendor_auth(v, timeout=timeout)
    broken = {v: d for v, (ok, d) in results.items() if not ok}
    if broken and fail_loud and os.environ.get("APEX_OMEGA_SKIP_AUTH_PREFLIGHT") != "1":
        raise RuntimeError(
            "vendor auth preflight FAILED for " + ", ".join(broken)
            + f" -> {broken}. Refresh the vendor CLI auth (e.g. run `codex exec \"ok\"` once) and "
            "retry, or set APEX_OMEGA_SKIP_AUTH_PREFLIGHT=1 to bypass. Aborting before the run "
            "burns agents on stale-auth non-results.")
    return results
