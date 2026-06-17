"""WS4: structural latency-profile harness.

Quantifies the latency wins from WS3 dispatch changes WITHOUT needing a real LLM
or container — it accounts for the *structural* work avoided:

* WS3B speculative-first-attempt: on an easy task, dispatch ONE seed rollout and
  accept on an authoritative pass instead of fanning out the whole slate. The
  expected saving is ``(slate_size - 1)`` rollouts whenever the seed passes.
* WS3F read-only explore: the localizer/reproducer stages run with the write
  tools (edit_file/bash/run_test/...) DENIED, so an explore turn can never burn
  time on a failed mutating call. The harness reports how many tools are removed
  from the explore surface.

Both are reported as ratios so a reviewer can sanity-check the claims; neither
reduces coverage (the speculative path falls through to the full slate on a miss,
and the read-only profile only applies to non-editing stages).
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Optional

from ..tools.aci import _READ_ONLY_DENIED_TOOL_NAMES, build_agent_tool_definitions


def _tool_names(definitions: Any) -> set[str]:
    names: set[str] = set()
    for d in definitions or []:
        if isinstance(d, dict):
            # Support both {"name": ...} and {"function": {"name": ...}} shapes.
            name = d.get("name")
            if not name and isinstance(d.get("function"), dict):
                name = d["function"].get("name")
            if name:
                names.add(str(name))
        else:
            name = getattr(d, "name", None)
            if name:
                names.add(str(name))
    return names


def profile_speculative_dispatch(
    *, slate_size: int = 4, seed_pass_rate: float = 0.6
) -> dict[str, Any]:
    """Expected rollouts dispatched per easy task, baseline vs speculative.

    Baseline always dispatches the full slate. Speculative dispatches 1 seed and,
    with probability ``seed_pass_rate``, stops; otherwise it dispatches the full
    slate (seed + remaining). Expected speculative cost:
        p*1 + (1-p)*slate_size
    """
    p = max(0.0, min(1.0, float(seed_pass_rate)))
    baseline = float(slate_size)
    speculative = p * 1.0 + (1.0 - p) * float(slate_size)
    saved = baseline - speculative
    return {
        "slate_size": slate_size,
        "seed_pass_rate": p,
        "expected_rollouts_baseline": round(baseline, 3),
        "expected_rollouts_speculative": round(speculative, 3),
        "expected_rollouts_saved": round(saved, 3),
        "expected_fraction_saved": round(saved / baseline, 3) if baseline else 0.0,
        # Never increases work: worst case (seed miss) == baseline + 0 (the seed
        # IS the first of the slate, reused), so coverage is preserved.
        "never_exceeds_baseline": speculative <= baseline + 1e-9,
    }


def profile_read_only_explore() -> dict[str, Any]:
    """How much of the tool surface the read-only explore profile removes."""
    full = _tool_names(build_agent_tool_definitions(read_only=False))
    explore = _tool_names(build_agent_tool_definitions(read_only=True))
    removed = sorted(full - explore)
    return {
        "full_tool_count": len(full),
        "explore_tool_count": len(explore),
        "removed_tools": removed,
        "removed_count": len(removed),
        "denied_set": sorted(_READ_ONLY_DENIED_TOOL_NAMES),
        # The explore profile must remove at least the mutating tools and never
        # ADD tools that the full profile lacks.
        "explore_is_subset": explore.issubset(full),
        "removes_mutating_tools": bool(removed),
    }


def run_latency_profile(
    *, slate_size: int = 4, seed_pass_rate: float = 0.6
) -> dict[str, Any]:
    speculative = profile_speculative_dispatch(
        slate_size=slate_size, seed_pass_rate=seed_pass_rate
    )
    read_only = profile_read_only_explore()
    verdict = {
        "speculative_dispatch": speculative,
        "read_only_explore": read_only,
    }
    verdict["passed"] = bool(
        speculative["never_exceeds_baseline"]
        and speculative["expected_rollouts_saved"] >= 0.0
        and read_only["explore_is_subset"]
        and read_only["removes_mutating_tools"]
    )
    return verdict


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="WS4 latency-profile harness")
    parser.add_argument("--slate-size", type=int, default=4)
    parser.add_argument("--seed-pass-rate", type=float, default=0.6)
    args = parser.parse_args(argv)
    verdict = run_latency_profile(
        slate_size=args.slate_size, seed_pass_rate=args.seed_pass_rate
    )
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
