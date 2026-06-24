"""Commit0 evaluation + ablation over the APEX-Ω target repos."""

from . import registry
from .commit0_autogen import difficulty_to_max_agents, run_autogen_cell
from .commit0_driver import Commit0EvalDriver, build_arm_config_dict, load_base_config
from .scoring import (
    decide_from_counts,
    load_expected_ids,
    verification_from_commit0_evaluation,
)

# SWE-rebench (Mode-C, gated by APEX_OMEGA_BENCHMARK=swerebench). Lazily importable
# siblings of the commit0 path; importing them here keeps the commit0 surface
# byte-identical (no behavior change) while exposing the swerebench registry/runner.
from . import swerebench_registry
from . import swerebench_runner

__all__ = [
    "registry",
    "Commit0EvalDriver",
    "load_base_config",
    "build_arm_config_dict",
    "decide_from_counts",
    "load_expected_ids",
    "verification_from_commit0_evaluation",
    "run_autogen_cell",
    "difficulty_to_max_agents",
    "swerebench_registry",
    "swerebench_runner",
]
