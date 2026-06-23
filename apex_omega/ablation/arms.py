"""Ablation flag surface + arm registry (plan §20.2/§20.3, Fusion Ledger).

Two coupled artifacts:
  * ``AblationConfig`` — the canonical APEX-Ω flag surface.  EVERY flag fails open
    to the heuristic/v1 baseline: ``applied=False ⇒ value==baseline`` (§20.3).
  * ``ARMS`` — the experiment matrix.  Each arm flips exactly one *logical*
    mechanism (a single flag, or a coordinated negative-control preset that
    deliberately violates an invariant to demonstrate degradation).

Arms carry an optional ``v1_overlay`` — a deep-merge dict applied onto a base
ApexConfig JSON — so the same arm can also be run through v1's
``Commit0BenchmarkRunner`` (Mode A) when a faithful v1 flag mapping exists.
Arms without a v1 mapping are exercised by the engine-native path (Mode B).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from .safety_modes import (
    BB_OFF,
    BB_PHASED_NEGATIVE,
    BB_SHARE_ALL,
    CTDG_ADVISORY,
    CTDG_PRUNE_HARD,
    CTDG_PRUNE_WITH_BACKSTOP,
    SafetyModeConfig,
)


# --- the canonical flag surface --------------------------------------------
@dataclass
class AblationConfig:
    # Phase-1 allocation / cost levers
    allocation_adaptive_low_k: bool = True          # A1 (off -> full-cap-16 baseline)
    allocation_feedback_floor: float = 0.35
    allocation_full_cap_k: int = 16
    allocation_default_k: int = 5
    allocation_min_distinct_profiles: int = 2
    # Phase-2 search / branching
    search_enabled: bool = True                     # A2 master
    search_activation_min_nodes: int = 8
    search_feedback_confidence_floor: float = 0.55
    search_speculate: bool = True
    search_multi_llm_routing: bool = True           # A8/A9 enabler
    # Phase-1 futility / efficiency
    localization_futility_gate: bool = True         # A3
    snowball_detector: bool = True
    prefix_cache_require_check: bool = True
    pipeline_streaming: bool = True
    worktree_pool: bool = True
    # Phase-2 CTDG test-impact (A4)
    ctdg_enabled: bool = True
    ctdg_safety_mode: str = CTDG_PRUNE_WITH_BACKSTOP
    ctdg_full_suite_backstop: bool = True
    ctdg_test_impact_prune: str = "reorder_and_dynamic_coverage"   # vs "static_ast_gate" (reject)
    # Phase-2 blackboard (A5)
    blackboard_enabled: bool = True
    blackboard_delivery: str = BB_PHASED_NEGATIVE
    blackboard_min_isolated_waves: int = 1
    blackboard_admit_threshold: float = 0.85
    blackboard_abstraction_negatives_only: bool = True
    # Phase-2 model economy (A6)
    economy_enabled: bool = False                   # default off (frontier-everywhere = v1)
    economy_difficulty_floor_enforced: bool = True
    economy_frontier_review_gate: bool = True
    economy_max_rewrite_cycles: int = 2
    # orchestration strategy: fixed best-of-N vs the generated-code orchestrator
    # (a planner authors a tailored, frozen orchestrate(ctx); fails open to best_of_n)
    orchestrator: str = "best_of_n"                 # best_of_n | autogen
    # verifier (A7)
    hybrid_verifier: str = "execution_plus_critic"  # execution_plus_critic | execution_only | critic_only
    # Phase-3 controller
    controller_library_enabled: bool = True
    controller_bandit_enabled: bool = True
    controller_confidence_floor: float = 0.55
    controller_capability_profiles: bool = True     # A9 (vs one-hot)
    controller_held_out_vendor: bool = False        # A10 (eval-only)
    # Cardinal Contract (A11 negative control relaxes this)
    cardinal_contract_enforced: bool = True
    # vendor pool (A8)
    vendor_pool: tuple[str, ...] = ("codex_cli",)
    # research opt-in (required to run rejected forms as negative controls)
    research_ablation_optin: bool = False

    def to_safety_modes(self) -> SafetyModeConfig:
        return SafetyModeConfig(
            ctdg_mode=self.ctdg_safety_mode,
            blackboard_delivery=self.blackboard_delivery,
            blackboard_phase_gate=self.blackboard_min_isolated_waves > 0,
            branch_collapse_floor=self.search_feedback_confidence_floor,
            full_cap_floor_K=self.allocation_full_cap_k,
            default_adaptive_K=self.allocation_default_k,
            cardinal_contract_enforced=self.cardinal_contract_enforced,
            economy_difficulty_floor_enforced=self.economy_difficulty_floor_enforced,
            economy_frontier_review_gate=self.economy_frontier_review_gate,
            research_ablation_optin=self.research_ablation_optin,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["vendor_pool"] = list(self.vendor_pool)
        return d


@dataclass
class AblationArm:
    id: str
    kind: str                       # baseline | ablation | negative_control
    isolates: str                   # the one mechanism this arm measures
    overrides: dict = field(default_factory=dict)     # AblationConfig field overrides
    v1_overlay: Optional[dict] = None                 # deep-merge onto base ApexConfig (Mode A); None if engine-native only
    maps_to: str = ""               # §20 matrix id (A1..A11, B0..B4)
    description: str = ""
    # A single-model 1-shot anchor (B0): instead of hardcoding a vendor, it INHERITS the eval's
    # base-config llm_configs (the model/harness orchestration is being evaluated on), truncated to
    # exactly ONE model. So `--base-config` is the single knob that moves the anchor alongside the
    # orchestrated arms. An APEX_OMEGA_BASELINE_BACKEND[/_MODEL] env override can decouple the anchor
    # (e.g. a frontier anchor while orchestrating on a cheaper model). Resolved in build_arm_config_dict.
    single_model: bool = False


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursive dict deep-merge (overlay wins).  Returns a new dict."""
    out = copy.deepcopy(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def build_ablation_config(arm: "AblationArm", base: Optional[AblationConfig] = None) -> AblationConfig:
    cfg = copy.deepcopy(base) if base is not None else AblationConfig()
    for k, v in arm.overrides.items():
        if not hasattr(cfg, k):
            raise KeyError(f"arm {arm.id!r} sets unknown AblationConfig field {k!r}")
        setattr(cfg, k, v)
    return cfg


# v1 overlay fragments (deep-merged onto a base ApexConfig JSON for Mode A) -----
# NOTE: B0 no longer pins a vendor — it inherits the base-config backend (single_model=True). The
# old _V1_SINGLE_VENDOR codex pin was removed so the 1-shot anchor tracks whatever model/harness we
# are currently evaluating orchestration on. _V1_VENDOR_MIX stays (the cross-vendor arms are
# deliberately codex+claude regardless of base-config).
_V1_FULL_CAP_16 = {"rollout": {"enable_adaptive_allocation": False, "num_rollouts": 16,
                               "min_rollouts": 16, "max_rollouts": 16}}
_V1_ADAPTIVE_K = {"rollout": {"enable_adaptive_allocation": True, "min_rollouts": 1,
                              "max_rollouts": 8, "rollout_buckets": [1, 4, 8]}}
_V1_SEARCH_ON = {"search": {"mode": "puct", "max_expansions": 6, "max_depth": 4, "max_frontier_branching": 8}}
_V1_SEARCH_OFF = {"search": {"mode": "off", "max_expansions": 0}}
_V1_CRITIC_ON = {"selection": {"use_critic": True, "enable_critic_reranking": True}}
_V1_CRITIC_OFF = {"selection": {"use_critic": False, "enable_critic_reranking": False}}
_V1_VENDOR_MIX = {"llm_configs": [
    {"backend": "codex_cli", "model": "gpt-5.5"},
    {"backend": "claude_cli", "model": "opus", "cli_model_id": "claude-opus-4-8[1m]"},
]}


# --- the experiment matrix --------------------------------------------------
ARMS: dict[str, AblationArm] = {
    # ---- baselines (B0..B4, §20.2) ----
    "baseline": AblationArm(
        "baseline", "baseline", "the full APEX-Ω config (all mechanisms at plan defaults)",
        overrides={}, v1_overlay=None, maps_to="-",
        description="All Phase-1 levers on, Phase-2 amplifiers at safe defaults, economy off."),
    "B0_single_model": AblationArm(
        "B0_single_model", "baseline", "one strong single model, single shot",
        overrides={"allocation_default_k": 1, "allocation_adaptive_low_k": False,
                   "search_enabled": False, "vendor_pool": ("codex_cli",)},
        v1_overlay={"rollout": {"enable_adaptive_allocation": False,
                   "num_rollouts": 1, "min_rollouts": 1, "max_rollouts": 1}},
        maps_to="B0", single_model=True,
        description="Single-model 1-shot counter-anchor. Inherits the eval's base-config backend "
                    "(the model/harness orchestration is being evaluated on), truncated to one model; "
                    "override with APEX_OMEGA_BASELINE_BACKEND[/_MODEL] for a decoupled anchor."),
    "B2_v1_full_cap16": AblationArm(
        "B2_v1_full_cap16", "baseline", "APEX v1 as-shipped (full-cap-16, caps off) cost pathology",
        overrides={"allocation_adaptive_low_k": False, "allocation_default_k": 16},
        v1_overlay=_V1_FULL_CAP_16, maps_to="B2",
        description="The strong incumbent / cost-pathology witness."),
    "B4_static_cross_vendor": AblationArm(
        "B4_static_cross_vendor", "baseline", "cross-vendor best-of-N, STATIC routing (no learned controller)",
        overrides={"vendor_pool": ("codex_cli", "claude_cli"), "controller_library_enabled": False},
        v1_overlay=_V1_VENDOR_MIX, maps_to="B4",
        description="Isolates the learned controller's marginal value over a heterogeneous pool."),
    # ---- ablations (A1..A11, §20.3) ----
    "A1_adaptive_k": AblationArm(
        "A1_adaptive_k", "ablation", "difficulty-adaptive low-K vs full-cap-16",
        overrides={"allocation_adaptive_low_k": False, "allocation_default_k": 16},
        v1_overlay=_V1_FULL_CAP_16, maps_to="A1",
        description="off => full-cap-16; expect near-equal solve at large cost cut."),
    "A2_branching": AblationArm(
        "A2_branching", "ablation", "bounded adaptive-branching vs collapse-to-best-of-N",
        overrides={"search_enabled": False}, v1_overlay=_V1_SEARCH_OFF, maps_to="A2",
        description="off => pure adaptive low-K best-of-N."),
    "A3_futility_gate": AblationArm(
        "A3_futility_gate", "ablation", "early localization-futility gate + snowball detector",
        overrides={"localization_futility_gate": False, "snowball_detector": False},
        v1_overlay={"rollout": {"progressive_stop_on_strong_signal": False}}, maps_to="A3",
        description="off => spawn K identical attempts on a possibly-dead frontier."),
    "A4_ctdg_runall": AblationArm(
        "A4_ctdg_runall", "ablation", "CTDG prioritize+dynamic-prune vs run-all",
        overrides={"ctdg_enabled": False, "ctdg_safety_mode": CTDG_ADVISORY},
        v1_overlay={"selection": {"enable_regression_pruning": False}}, maps_to="A4",
        description="off => run all tests (no test-impact pruning)."),
    "A5_blackboard_off": AblationArm(
        "A5_blackboard_off", "ablation", "blackboard 2.0 vs no-sharing",
        overrides={"blackboard_enabled": False, "blackboard_delivery": BB_OFF},
        v1_overlay={"rollout": {"enable_cross_rollout_discovery_reuse": False,
                                "enable_cross_solve_episodic_memory": False}}, maps_to="A5",
        description="off => pure isolated rollouts."),
    "A6_economy_on": AblationArm(
        "A6_economy_on", "ablation", "model economy cascade vs heavy-everywhere",
        overrides={"economy_enabled": True}, v1_overlay=None, maps_to="A6",
        description="on => difficulty-gated economy cascade vs frontier-everywhere (baseline=v1 off)."),
    "A7_verifier": AblationArm(
        "A7_verifier", "ablation", "hybrid verifier (execution+critic) vs execution-only",
        overrides={"hybrid_verifier": "execution_only"}, v1_overlay=_V1_CRITIC_OFF, maps_to="A7",
        description="execution-only => critic disabled (still execution-authoritative)."),
    "A8_vendor_mix": AblationArm(
        "A8_vendor_mix", "ablation", "heterogeneous pool vs single-vendor baseline (controller held constant)",
        overrides={"vendor_pool": ("codex_cli", "claude_cli")}, v1_overlay=_V1_VENDOR_MIX, maps_to="A8",
        description="heterogeneous pool => cross-vendor decorrelation vs single-vendor baseline."),
    "A9_one_hot": AblationArm(
        "A9_one_hot", "ablation", "learned capability/cost profiles vs one-hot vendor ids (H1)",
        overrides={"controller_capability_profiles": False}, v1_overlay=None, maps_to="A9",
        description="one-hot => cannot route an unseen vendor."),
    "A10_held_out_vendor": AblationArm(
        "A10_held_out_vendor", "ablation", "held-out-vendor generalization, no retraining",
        overrides={"controller_held_out_vendor": True, "vendor_pool": ("codex_cli", "claude_cli", "gemini_cli")},
        v1_overlay=None, maps_to="A10",
        description="test pool contains a vendor absent at train time; route via profile."),
    "autogen_orchestrator": AblationArm(
        "autogen_orchestrator", "ablation", "generated-code orchestrator vs fixed best-of-N",
        overrides={"orchestrator": "autogen"}, v1_overlay=None, maps_to="-",
        description="planner authors a tailored, frozen orchestrate(ctx) (1000s-of-agents capable); "
                    "fails open to verified best-of-N."),
    # ---- negative controls (deliberately violate an invariant) ----
    "A4_static_gate_NC": AblationArm(
        "A4_static_gate_NC", "negative_control", "static-AST CTDG gate (R2) — drops fault-revealing tests",
        overrides={"ctdg_test_impact_prune": "static_ast_gate", "ctdg_full_suite_backstop": False,
                   "research_ablation_optin": True}, v1_overlay=None, maps_to="A4",
        description="expected: fault-revealing-test loss -> solve-rate drop."),
    "A5_share_all_NC": AblationArm(
        "A5_share_all_NC", "negative_control", "raw share-all blackboard (R4) — homogenizes attempts",
        overrides={"blackboard_delivery": BB_SHARE_ALL, "blackboard_min_isolated_waves": 0,
                   "blackboard_admit_threshold": 0.0, "blackboard_abstraction_negatives_only": False,
                   "research_ablation_optin": True}, v1_overlay=None, maps_to="A5",
        description="expected: ~-3.7pp and diversity collapse."),
    "A6_thin_everywhere_NC": AblationArm(
        "A6_thin_everywhere_NC", "negative_control", "thin-executor-everywhere (R5) — cheap on hard SWE",
        overrides={"economy_enabled": True, "economy_difficulty_floor_enforced": False,
                   "economy_frontier_review_gate": False, "research_ablation_optin": True},
        v1_overlay=None, maps_to="A6",
        description="expected: worst resolve-rate drop (HyperAgent finding)."),
    "A11_cardinal_relaxed_NC": AblationArm(
        "A11_cardinal_relaxed_NC", "negative_control", "Cardinal-Contract relaxation (H2) — soft signals promote",
        overrides={"cardinal_contract_enforced": False, "research_ablation_optin": True},
        v1_overlay=None, maps_to="A11",
        description="expected: solve-rate inversion (false-positive ship)."),
}


def get_arm(arm_id: str) -> AblationArm:
    if arm_id not in ARMS:
        raise KeyError(f"unknown arm {arm_id!r}; known: {sorted(ARMS)}")
    return ARMS[arm_id]


def arms_by_kind(kind: str) -> list[AblationArm]:
    return [a for a in ARMS.values() if a.kind == kind]


def v1_runnable_arms() -> list[str]:
    """Arms that have a faithful v1-config overlay (runnable via Mode A)."""
    return [a.id for a in ARMS.values() if a.v1_overlay is not None or a.id == "baseline"]
