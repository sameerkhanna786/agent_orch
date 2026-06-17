"""Coarse outcome status enum for an APEX run.

Phase 3.2 hoist: previously defined inline in ``apex/orchestrator.py``.
Promoted to ``apex.core.status`` so ``ModeResult`` (apex.modes) and the
in-container V5 agent can import it without pulling in the entire
orchestrator module.
"""

from __future__ import annotations

import enum


class Status(str, enum.Enum):
    """Coarse outcome status for an APEX run.

    Phase 2C 2.2: this replaces the dual ``success/salvaged_for_external_scoring``
    pair as the canonical surface. ``success`` is preserved as a derived
    bool for backward-compatibility (``success == status is SOLVED``).
    """

    SOLVED = "solved"
    ABSTAINED = "abstained"
    FAILED = "failed"
    ENV_SKIPPED = "env_skipped"


__all__ = ["Status"]
