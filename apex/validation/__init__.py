"""WS4 — APEX validation harnesses.

Three falsifiable harnesses that validate the step-change designs landed in the
V5 / WS2 / WS3 work:

* :mod:`apex.validation.decay_resistance` — longitudinal A/B that proves the V5
  append-only, value-preserving working memory keeps exact identifiers across
  many sessions (infinite information half-life) while a lossy-compaction
  baseline decays (finite half-life). Motivated by arxiv 2605.26302.
* :mod:`apex.validation.ablation_matrix` — enumerates every default-ON ablation
  component and default-OFF optional component, proving each toggle round-trips
  through config and flips the enforcement helpers (no silent no-ops).
* :mod:`apex.validation.latency_profile` — structural latency accounting for the
  WS3B speculative-first-attempt dispatch and the WS3F read-only explore tool
  profile (no real LLM / container needed).

Each module exposes a ``run(...) -> dict`` returning a structured report and a
``main(argv)`` CLI entry point.
"""

from __future__ import annotations

from .ablation_matrix import run_ablation_matrix
from .decay_resistance import run_decay_resistance
from .latency_profile import run_latency_profile

__all__ = [
    "run_decay_resistance",
    "run_ablation_matrix",
    "run_latency_profile",
]
