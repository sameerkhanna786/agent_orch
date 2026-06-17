"""SafetyModeConfig — the single typed object where "adopt-modified, not the
gate form" becomes load-bearing in code (Fusion Ledger §18.4.1).

The admission rule (``validate_safety_modes``) makes it *impossible* to silently
ship a Reject-disposition form:
  * R2 static-AST CTDG gate → ``prune_hard`` requires dynamic coverage + backstop.
  * R3 plan-score hard gate → no ``gate`` enum value exists by construction.
  * R4 raw share-all blackboard → requires an explicit research-ablation opt-in.
  * M5 verifier isolation → structural, asserted at wiring time.
This is the mechanical guarantee that the canonical ``accepted_mechanisms`` array
(§18.5) is never contradicted (pitfall 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..errors import FailLoud


# --- enums as frozen string sets (kept simple + JSON-friendly) -------------
CTDG_ADVISORY = "advisory"
CTDG_PRUNE_WITH_BACKSTOP = "prune_with_backstop"
CTDG_PRUNE_HARD = "prune_hard"
CTDG_MODES = (CTDG_ADVISORY, CTDG_PRUNE_WITH_BACKSTOP, CTDG_PRUNE_HARD)

PLAN_SCORE_PRIORITIZE_ONLY = "prioritize_only"
PLAN_SCORE_DOWNGRADE_ONLY = "downgrade_only"
PLAN_SCORE_MODES = (PLAN_SCORE_PRIORITIZE_ONLY, PLAN_SCORE_DOWNGRADE_ONLY)  # NOTE: no 'gate' by construction (R3)

BB_OFF = "off"
BB_PHASED_NEGATIVE = "phased_negative"
BB_SHARE_ALL = "share_all"           # Reject (R4); opt-in research ablation only
BB_DELIVERIES = (BB_OFF, BB_PHASED_NEGATIVE, BB_SHARE_ALL)


@dataclass
class SafetyModeConfig:
    ctdg_mode: str = CTDG_PRUNE_WITH_BACKSTOP
    plan_score_mode: str = PLAN_SCORE_DOWNGRADE_ONLY
    blackboard_delivery: str = BB_PHASED_NEGATIVE
    blackboard_phase_gate: bool = True       # first exploratory wave fully isolated
    branch_collapse_floor: float = 0.6       # feedback-confidence below which M1 → best-of-N
    full_cap_floor_K: int = 16               # thin-feedback floor only
    default_adaptive_K: int = 5              # difficulty-adaptive low-K target
    # rejected-form toggles (R5/A11): default to the SAFE value; relaxing any of
    # these requires research_ablation_optin (gated in validate_safety_modes).
    cardinal_contract_enforced: bool = True
    economy_difficulty_floor_enforced: bool = True
    economy_frontier_review_gate: bool = True
    # explicit opt-in required to run any rejected form (share-all, cardinal
    # relaxation, economy-floor disablement) as a negative-control ablation.
    research_ablation_optin: bool = False
    # structural verifier isolation (M5): asserted, not user-configurable to False
    verifier_isolated_from_producer: bool = True

    def to_dict(self) -> dict:
        return {
            "ctdg_mode": self.ctdg_mode,
            "plan_score_mode": self.plan_score_mode,
            "blackboard_delivery": self.blackboard_delivery,
            "blackboard_phase_gate": self.blackboard_phase_gate,
            "branch_collapse_floor": self.branch_collapse_floor,
            "full_cap_floor_K": self.full_cap_floor_K,
            "default_adaptive_K": self.default_adaptive_K,
            "cardinal_contract_enforced": self.cardinal_contract_enforced,
            "economy_difficulty_floor_enforced": self.economy_difficulty_floor_enforced,
            "economy_frontier_review_gate": self.economy_frontier_review_gate,
            "research_ablation_optin": self.research_ablation_optin,
            "verifier_isolated_from_producer": self.verifier_isolated_from_producer,
        }


def validate_safety_modes(
    cfg: SafetyModeConfig,
    *,
    dynamic_coverage_available: Optional[Callable[[], bool]] = None,
) -> None:
    """Fail-loud admission rule.  Call at config-load / wiring time.  Never
    silently downgrades a Reject into an Adopt (pitfall 2)."""
    if cfg.ctdg_mode not in CTDG_MODES:
        raise FailLoud(f"ctdg_mode {cfg.ctdg_mode!r} not in {CTDG_MODES}")
    if cfg.plan_score_mode not in PLAN_SCORE_MODES:
        # By construction there is no 'gate' value (R3); an unknown value is a defect.
        raise FailLoud(f"plan_score_mode {cfg.plan_score_mode!r} not in {PLAN_SCORE_MODES}")
    if cfg.blackboard_delivery not in BB_DELIVERIES:
        raise FailLoud(f"blackboard_delivery {cfg.blackboard_delivery!r} not in {BB_DELIVERIES}")

    # R2: static-AST gate is rejected; prune_hard requires DYNAMIC coverage.
    if cfg.ctdg_mode == CTDG_PRUNE_HARD:
        have_dyn = dynamic_coverage_available() if dynamic_coverage_available else False
        if not have_dyn:
            raise FailLoud(
                "ctdg prune_hard requires dynamic coverage; static-AST gating is rejected (R2). "
                "Use prune_with_backstop, or supply dynamic_coverage_available()."
            )

    # R4: share_all is a rejected form; only permitted behind an explicit opt-in.
    if cfg.blackboard_delivery == BB_SHARE_ALL and not cfg.research_ablation_optin:
        raise FailLoud(
            "blackboard share_all is rejected (R4, -3.7pp); set research_ablation_optin=True "
            "to run it as a negative-control ablation only."
        )

    # A11/R5: relaxing the Cardinal Contract or disabling the economy difficulty
    # floor / frontier-review gate are rejected forms; opt-in required.
    if (not cfg.cardinal_contract_enforced
            or not cfg.economy_difficulty_floor_enforced
            or not cfg.economy_frontier_review_gate) and not cfg.research_ablation_optin:
        raise FailLoud(
            "Cardinal-Contract relaxation / economy-floor disablement is a rejected form; "
            "set research_ablation_optin=True to run it as a negative control only."
        )

    # M5: verifier isolation is structural, not a soft preference.
    if not cfg.verifier_isolated_from_producer:
        raise FailLoud("verifier_isolated_from_producer must remain True (M5; anti-collective-delusion)")

    # branch_collapse_floor sanity
    if not (0.0 <= cfg.branch_collapse_floor <= 1.0):
        raise FailLoud(f"branch_collapse_floor {cfg.branch_collapse_floor} out of [0,1]")
