"""Consolidated security guards for the ACI tool surface.

This module exists so the ACI's security boundary is reviewable in one
place rather than scattered across :mod:`apex.tools.aci`. Three concerns
live here:

1. ``TestCommandRejectedError`` and ``validate_test_command`` — reject
   shell-injection metacharacters in any shell-bound command before we
   hand it to bash.
2. ``HARD_BLOCK_PATTERNS`` — the always-on denylist (backticks,
   ``$(...)``, process substitution, embedded newlines, carriage
   returns). These remain forbidden even when a caller explicitly opts
   into chaining (`allow_shell_chaining=True`), because backticks and
   ``$(...)`` are command substitution rather than chaining and embedded
   newlines smuggle a second statement past chaining-token scanners.
3. ``resolve_bash_invocation`` — pick between ``bash -c`` (default) and
   ``bash -lc`` (login shell, opt-in). Sourcing login profiles can
   re-introduce host secrets and shell-rc side effects into the agent's
   environment, so we default to the non-login form. Callers that
   genuinely need login-profile semantics (PATH munging, conda init,
   custom prompt setup) opt in via ``allow_login_shell=True``.

The :mod:`apex.tools.aci` module re-exports the public names from here,
so existing imports (``from apex.tools.aci import validate_test_command``)
keep working.
"""

from __future__ import annotations

import logging
from typing import Optional

_security_logger = logging.getLogger("apex.security")


class TestCommandRejectedError(ValueError):
    """Raised when a test command contains shell-injection metacharacters."""

    # Tell pytest to skip this exception class during test collection.
    __test__ = False

    def __init__(self, command: str, reason: str) -> None:
        super().__init__(reason)
        self.command = command
        self.reason = reason


# Always-blocked patterns: command substitution and unbounded chains. These
# enable shell injection regardless of whether basic chaining is permitted.
#
# Documented security boundary:
#   * Backticks and ``$(...)`` are command substitution, not chaining.
#     Allowing them would let a model-supplied "test" command execute
#     arbitrary code captured into a variable that the chaining-token
#     scanner cannot see.
#   * Process substitution (``<(...)``, ``>(...)``) is similarly an
#     always-on shell feature that opens read/write file descriptors to
#     subshells we never sandboxed.
#   * Embedded newlines / carriage returns separate statements inside
#     ``bash -c`` (and ``bash -lc``); a model-supplied command like
#     ``"pytest -q\nrm -rf /"`` would smuggle a second statement past
#     chaining-token scanners that only look at ``;``/``&&``/``||``.
HARD_BLOCK_PATTERNS: tuple[tuple[str, str], ...] = (
    ("`", "backtick command substitution"),
    ("$(", "$(...) command substitution"),
    ("<(", "<(...) process substitution"),
    (">(", ">(...) process substitution"),
    ("\n", "embedded newline (multi-line command)"),
    ("\r", "embedded carriage return (multi-line command)"),
)

# Patterns that imply chaining/redirection. Allowed only when callers pass
# allow_shell_chaining=True. Single-character tokens like `&` are validated
# more carefully via _scan_test_command_chaining below to avoid false positives
# (for example C-style escapes inside quoted strings handed to python -c).
CHAINING_TOKENS: tuple[str, ...] = (";", "&&", "||")


def scan_test_command_chaining(command: str) -> Optional[str]:
    """Return the offending token if ``command`` chains/redirects shell ops.

    The scanner walks the string left-to-right, skipping over quoted regions,
    so we don't trip on ``--option='a;b'`` style arguments. Returns None when
    no chaining/redirection token is found at top level.

    NOTE: We deliberately do NOT honor backslash-escaped quotes inside the
    quoted region (e.g. treating ``"a\\"b"`` as continuation). Doing so
    would let an attacker close a quote with ``\\`` and smuggle a ``;`` —
    over-rejecting on contrived inputs is the safer failure mode.
    """

    i = 0
    length = len(command)
    while i < length:
        ch = command[i]
        if ch in ("'", '"'):
            close = command.find(ch, i + 1)
            if close == -1:
                # Unbalanced quote; treat as suspicious.
                return f"unbalanced quote starting at index {i}"
            i = close + 1
            continue
        if ch == "\\" and i + 1 < length:
            i += 2
            continue
        for token in CHAINING_TOKENS:
            if command.startswith(token, i):
                return token
        if ch in ("|", ">", "<"):
            # Allow >> and << only with explicit chaining permission too.
            return ch
        if ch == "&":
            # Background `&` (single, not part of `&&` which we handled above).
            if i + 1 >= length or command[i + 1] != "&":
                return "& (background)"
        i += 1
    return None


def validate_test_command(
    command: Optional[str],
    *,
    allow_shell_chaining: bool = False,
    source: str = "test_command",
) -> str:
    """Validate ``command`` for shell-injection metacharacters.

    Returns the command unchanged when it is empty or accepted. Raises
    :class:`TestCommandRejectedError` (and emits a structured warning) when the
    command contains a forbidden pattern. ``allow_shell_chaining`` opts into
    accepting `&&`, `||`, `;`, and basic redirection — required for the
    benchmark configs that legitimately chain ``export ... && pytest``.

    Even when chaining is allowed, the patterns in :data:`HARD_BLOCK_PATTERNS`
    remain forbidden. See the module docstring for the documented security
    boundary.
    """

    if command is None:
        return ""
    text = str(command)
    if not text.strip():
        return text

    for pattern, description in HARD_BLOCK_PATTERNS:
        if pattern in text:
            reason = f"forbidden shell construct: {description}"
            _security_logger.warning(
                "test_command_rejected",
                extra={
                    "event": "apex.security.test_command_rejected",
                    "source": source,
                    "reason": reason,
                    "command_preview": text[:200],
                },
            )
            raise TestCommandRejectedError(text, reason)

    if not allow_shell_chaining:
        offender = scan_test_command_chaining(text)
        if offender is not None:
            reason = f"shell chaining/redirection token disallowed: {offender!r}"
            _security_logger.warning(
                "test_command_rejected",
                extra={
                    "event": "apex.security.test_command_rejected",
                    "source": source,
                    "reason": reason,
                    "command_preview": text[:200],
                },
            )
            raise TestCommandRejectedError(text, reason)
    return text


def resolve_bash_invocation(*, allow_login_shell: bool = False) -> list[str]:
    """Return the bash argv prefix used for shell-bound commands.

    Defaults to ``["bash", "-c"]`` because sourcing login profiles
    (``-l``) can re-introduce host secrets, ``HOME`` overrides, and
    interactive-prompt side effects into the agent's environment. The
    long-running ACI process already controls ``env=`` explicitly, so the
    extra profile-sourcing buys no convenience while opening a real
    contamination surface.

    Set ``allow_login_shell=True`` to opt back into ``["bash", "-lc"]``
    for the rare callers that actually need ``~/.bash_profile`` /
    ``~/.profile`` / ``conda init`` side effects (typically benchmark
    configs that ship a curated login profile inside a docker image).
    """

    if allow_login_shell:
        return ["bash", "-lc"]
    return ["bash", "-c"]


__all__ = [
    "CHAINING_TOKENS",
    "HARD_BLOCK_PATTERNS",
    "TestCommandRejectedError",
    "resolve_bash_invocation",
    "scan_test_command_chaining",
    "validate_test_command",
]
