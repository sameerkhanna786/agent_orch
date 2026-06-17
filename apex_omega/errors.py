"""APEX-Ω error types.  The cardinal one is ``FailLoud`` — the engine fails
loud, never fakes (Principle 7).  Salvage is never success; a rejected mechanism
form can never be silently downgraded into an adopted one."""

from __future__ import annotations


class ApexOmegaError(Exception):
    """Base class for all APEX-Ω errors."""


class FailLoud(ApexOmegaError):
    """Raised at config-load / wiring time when an invariant would be violated
    (e.g. shipping a Reject-disposition mechanism form without explicit opt-in).
    Surfacing loudly is mandatory — a silent downgrade is a correctness defect."""


class ConcurrentWorktreeError(ApexOmegaError):
    """Raised when a per-rollout fcntl lock is already held — never silently
    share or nuke a sibling worktree (Cardinal Safety isolation invariant)."""


class CapabilityNegotiationError(ApexOmegaError):
    """Raised only when degradation is impossible (a hard capability gap with no
    fallback path).  The default is *degrade, do not crash*."""


class PlateauStop(ApexOmegaError):
    """Raised by ``ctx.parallel`` when the RunGovernor halts escalation (plateau or
    budget/ceiling reached). A CLEAN stop, NOT a defect: the orchestration unwinds and
    the host selects the best banked candidate. This is what makes a default-unbounded,
    no-clock run (incl. a pathological ``while True``) terminate (Backbone 2.1).

    Carries a ``reason`` (the governor verdict string, e.g. ``"stop:agent-ceiling"`` or
    ``"plateau:no-progress"``) so the ledger/reclassifier records WHY escalation stopped."""

    @property
    def reason(self) -> str:
        return str(self.args[0]) if self.args else ""


class CutLosses(PlateauStop):
    """A genuine NON-PROGRESS cut: the run is objectively stuck (a dead/sterile state or
    a budget-aware patience window exhausted with no distance-to-solve gain) and further
    work would only burn tokens. A subclass of ``PlateauStop`` (so every host that already
    catches PlateauStop handles it), but a DISTINCT taxonomy bucket: a cut is a declared
    FAILURE with a ``cut:<reason>`` — NOT an infra/timeout non-result, and NOT an honest
    "explored, no headroom left" stop. Carries the specific ``cut:<reason>`` verbatim."""
