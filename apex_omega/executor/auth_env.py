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
from typing import MutableMapping, Optional

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
