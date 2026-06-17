"""SafetyModeConfig admission rules + ablation arm integrity (§18.4.1, §20.3)."""

from __future__ import annotations

import pytest

from dataclasses import asdict

from apex_omega.ablation import ARMS, AblationConfig, build_ablation_config, deep_merge, get_arm
from apex_omega.ablation.safety_modes import (
    BB_PHASED_NEGATIVE,
    BB_SHARE_ALL,
    CTDG_PRUNE_HARD,
    SafetyModeConfig,
    validate_safety_modes,
)
from apex_omega.errors import FailLoud


def test_default_safety_modes_valid():
    validate_safety_modes(SafetyModeConfig())  # default config admits


def test_share_all_rejected_without_optin():
    with pytest.raises(FailLoud):
        validate_safety_modes(SafetyModeConfig(blackboard_delivery=BB_SHARE_ALL))
    # opt-in permits it as a negative-control ablation
    validate_safety_modes(SafetyModeConfig(blackboard_delivery=BB_SHARE_ALL, research_ablation_optin=True))


def test_prune_hard_requires_dynamic_coverage():
    with pytest.raises(FailLoud):
        validate_safety_modes(SafetyModeConfig(ctdg_mode=CTDG_PRUNE_HARD),
                              dynamic_coverage_available=lambda: False)
    validate_safety_modes(SafetyModeConfig(ctdg_mode=CTDG_PRUNE_HARD),
                          dynamic_coverage_available=lambda: True)


def test_verifier_isolation_structural():
    with pytest.raises(FailLoud):
        validate_safety_modes(SafetyModeConfig(verifier_isolated_from_producer=False))


def test_every_arm_safety_validates():
    for aid, arm in ARMS.items():
        cfg = build_ablation_config(arm)
        # negative controls opt in to research; assume dynamic coverage available for the check
        validate_safety_modes(cfg.to_safety_modes(), dynamic_coverage_available=lambda: True)


def test_baseline_arm_is_default_config():
    cfg = build_ablation_config(get_arm("baseline"))
    assert cfg.cardinal_contract_enforced is True
    assert cfg.economy_enabled is False  # default off (frontier-everywhere)
    assert cfg.allocation_adaptive_low_k is True


def test_negative_controls_relax_an_invariant():
    nc = build_ablation_config(get_arm("A11_cardinal_relaxed_NC"))
    assert nc.cardinal_contract_enforced is False and nc.research_ablation_optin is True
    share = build_ablation_config(get_arm("A5_share_all_NC"))
    assert share.blackboard_delivery == BB_SHARE_ALL and share.research_ablation_optin is True


def test_deep_merge_overlay_wins():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    over = {"a": {"y": 9}, "c": 4}
    assert deep_merge(base, over) == {"a": {"x": 1, "y": 9}, "b": 3, "c": 4}
    assert base == {"a": {"x": 1, "y": 2}, "b": 3}  # base not mutated


def test_cardinal_relaxation_requires_optin():
    # Regression (review finding #8): relaxing the Cardinal Contract / economy
    # floors without the research opt-in must fail loud (not silently ship).
    with pytest.raises(FailLoud):
        validate_safety_modes(AblationConfig(cardinal_contract_enforced=False).to_safety_modes())
    with pytest.raises(FailLoud):
        validate_safety_modes(AblationConfig(economy_difficulty_floor_enforced=False).to_safety_modes())
    # with opt-in, permitted as a negative control
    validate_safety_modes(AblationConfig(cardinal_contract_enforced=False,
                                         research_ablation_optin=True).to_safety_modes())


def test_no_ablation_arm_byte_identical_to_baseline():
    # Regression (review findings #7, #9): every clean ablation arm must flip at
    # least one mechanism (A8 heterogeneous, A6 economy on, etc.).
    base = asdict(build_ablation_config(get_arm("baseline")))
    for aid, arm in ARMS.items():
        if arm.kind == "ablation":
            assert asdict(build_ablation_config(arm)) != base, f"{aid} flips zero mechanisms"


def test_arms_map_to_matrix_ids():
    # every A* ablation maps to a §20 matrix id
    for aid, arm in ARMS.items():
        if aid.startswith("A") and arm.kind in ("ablation", "negative_control"):
            assert arm.maps_to, f"{aid} missing §20 matrix mapping"
