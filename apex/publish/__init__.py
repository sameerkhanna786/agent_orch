"""Reviewer-mode benchmark publication bundle (Phase 6.7).

The ``apex publish-benchmark <run_dir>`` CLI subcommand wraps
:class:`apex.publish.bundle.PublicationBundle` to produce a self-contained,
reviewer-friendly directory containing:

* ``RESULTS.md`` — score table per benchmark, per-task results, optional
  fairness-audit deltas, abstention rates, Pareto-frontier reference
  (if ``pareto_frontier.json`` exists).
* ``MANIFEST.json`` — copy of ``run_manifest.json`` from Phase 0.2.
* ``OVERRIDES_DISCLOSURE.md`` — copy of the Phase 1.7 disclosure.
* ``predictions/`` — one ``predictions.jsonl`` per benchmark in upstream-
  canonical schema (Commit0, SWT-Bench, TestGenEval).
* ``REPRODUCE.sh`` — one-command bash script that pins APEX + harness
  versions from the manifest and re-runs the benchmark slice.
* ``README.md`` — top-level explainer for external reviewers.

Bundles are designed to be tarred and shared — there are no symlinks,
no absolute paths, and every artifact is self-contained.
"""

from __future__ import annotations

from .bundle import (
    BUNDLE_FILES,
    BundleValidationError,
    PublicationBundle,
)

__all__ = [
    "BUNDLE_FILES",
    "BundleValidationError",
    "PublicationBundle",
]
