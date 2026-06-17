"""Fail-open ablation flags + SafetyModeConfig admission + the experiment matrix."""

from .arms import (
    ARMS,
    AblationArm,
    AblationConfig,
    arms_by_kind,
    build_ablation_config,
    deep_merge,
    get_arm,
    v1_runnable_arms,
)
from .safety_modes import SafetyModeConfig, validate_safety_modes

__all__ = [
    "AblationConfig",
    "AblationArm",
    "ARMS",
    "get_arm",
    "arms_by_kind",
    "build_ablation_config",
    "deep_merge",
    "v1_runnable_arms",
    "SafetyModeConfig",
    "validate_safety_modes",
]
