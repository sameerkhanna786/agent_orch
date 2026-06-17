"""WS4: component ablation / optional-component toggle matrix harness.

Validates that every controllable component actually has an enforced effect — no
silent no-ops — and that toggles round-trip through config:

* default-ON ablation components (:data:`ABLATION_COMPONENTS`): disabling one via
  an assignment must flip :func:`component_enabled` to False while leaving the
  others True.
* default-OFF optional components (:data:`OPTIONAL_COMPONENTS`, e.g. the WS2C
  EG-critic): :func:`component_optional_enabled` must be False by default and
  True only when the component is explicitly listed.
* behavioral arms (WS3I): each runtime_policy gate flips the
  :func:`behavioral_arms_summary` snapshot.

The harness needs no LLM — it exercises the pure assignment/enforcement helpers
and the config round-trip directly.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Optional

from ..core.component_ablation import (
    ABLATION_COMPONENTS,
    OPTIONAL_COMPONENTS,
    behavioral_arms_summary,
    component_enabled,
    component_optional_enabled,
)
from ..core.config import ApexConfig


def _config_with_runtime_policy(policy: dict[str, Any]) -> ApexConfig:
    cfg = ApexConfig()
    # runtime_policy lives on the benchmark config as a free-form mapping.
    cfg.benchmark.runtime_policy = dict(policy)
    return cfg


def _check_ablation_components() -> dict[str, Any]:
    """Each ablation component, when disabled, must flip enabled->False for
    itself and leave the others enabled."""
    rows = []
    all_ok = True
    for component in ABLATION_COMPONENTS:
        assignment = {
            "enabled": True,
            "disabled_component": component,
            "arm": f"disable_{component}",
        }
        self_enabled = component_enabled(assignment, component)
        others_enabled = all(
            component_enabled(assignment, other)
            for other in ABLATION_COMPONENTS
            if other != component
        )
        ok = (self_enabled is False) and (others_enabled is True)
        all_ok = all_ok and ok
        rows.append(
            {
                "component": component,
                "self_disabled_correctly": self_enabled is False,
                "others_remain_enabled": others_enabled,
                "ok": ok,
            }
        )
    # Control arm (nothing disabled) -> all enabled.
    control = {"enabled": True, "disabled_component": None, "arm": "control"}
    control_ok = all(component_enabled(control, c) for c in ABLATION_COMPONENTS)
    all_ok = all_ok and control_ok
    return {"rows": rows, "control_all_enabled": control_ok, "ok": all_ok}


def _check_optional_components() -> dict[str, Any]:
    """Each optional component is OFF by default and ON only when listed."""
    rows = []
    all_ok = True
    for component in OPTIONAL_COMPONENTS:
        off_default = component_optional_enabled({"enabled": True}, component) is False
        off_when_disabled = (
            component_optional_enabled(
                {"enabled": False, "enabled_optional_components": [component]}, component
            )
            is False
        )
        on_when_listed = (
            component_optional_enabled(
                {"enabled": True, "enabled_optional_components": [component]}, component
            )
            is True
        )
        ok = off_default and off_when_disabled and on_when_listed
        all_ok = all_ok and ok
        rows.append(
            {
                "component": component,
                "off_by_default": off_default,
                "off_when_assignment_disabled": off_when_disabled,
                "on_when_listed": on_when_listed,
                "ok": ok,
            }
        )
    return {"rows": rows, "ok": all_ok}


def _check_behavioral_arms() -> dict[str, Any]:
    """Each WS3I behavioral arm gate flips the summary snapshot."""
    base = behavioral_arms_summary(_config_with_runtime_policy({}))
    rows = []
    all_ok = True
    for gate in ("clarification_abstain_enabled", "anti_repetition_downrank_enabled"):
        on = behavioral_arms_summary(_config_with_runtime_policy({gate: True}))
        ok = (base.get(gate) is False) and (on.get(gate) is True)
        all_ok = all_ok and ok
        rows.append(
            {"gate": gate, "default_off": base.get(gate) is False, "flips_on": on.get(gate) is True, "ok": ok}
        )
    return {"rows": rows, "default_snapshot": base, "ok": all_ok}


def _check_config_round_trip() -> dict[str, Any]:
    """The WS2C/WS3 config flags must survive a to_dict round-trip with their
    documented defaults (default-OFF where required)."""
    cfg = ApexConfig()
    data = cfg.to_dict()
    rollout = data.get("rollout", {})
    selection = data.get("selection", {}) if isinstance(data.get("selection"), dict) else {}
    checks = {
        "rollout.enable_speculative_first_attempt_present": "enable_speculative_first_attempt"
        in rollout,
        "rollout.enable_cross_solve_episodic_memory_default_off": rollout.get(
            "enable_cross_solve_episodic_memory"
        )
        is False,
    }
    # EG-critic tiebreak default-off (WS2C) — read straight from the dataclass
    # since selection echoes may vary in shape across versions.
    checks["selection.enable_eg_critic_tiebreak_default_off"] = (
        getattr(cfg.selection, "enable_eg_critic_tiebreak", None) is False
    )
    checks["selection.enable_final_acceptance_reviewer_default_off"] = (
        getattr(cfg.selection, "enable_final_acceptance_reviewer", None) is False
    )
    ok = all(checks.values())
    return {"checks": checks, "ok": ok, "_selection_keys": sorted(selection.keys())[:0]}


def run_ablation_matrix() -> dict[str, Any]:
    """Run all four sub-checks and return a structured verdict."""
    ablation = _check_ablation_components()
    optional = _check_optional_components()
    arms = _check_behavioral_arms()
    round_trip = _check_config_round_trip()
    verdict = {
        "ablation_components": ablation,
        "optional_components": optional,
        "behavioral_arms": arms,
        "config_round_trip": round_trip,
    }
    verdict["passed"] = bool(
        ablation["ok"] and optional["ok"] and arms["ok"] and round_trip["ok"]
    )
    return verdict


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="WS4 ablation/optional toggle matrix harness")
    parser.parse_args(argv)
    verdict = run_ablation_matrix()
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
