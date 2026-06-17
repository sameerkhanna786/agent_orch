"""WS2C: offline harvest -> fit -> validate pipeline for the EG-Critic.

Reads a harvested JSONL of execution-grounded features + gold pass/fail labels,
fits an :class:`ExecutionGroundedCritic`, and validates that the fitted head's
held-out selection accuracy BEATS the heuristic floor before the artifact is
allowed to move live rankings (the falsifiable gate from the design dossier).

JSONL row schema (one candidate per line):
    {"task_id": "...", "gold_label": 0|1, "features": {<FEATURE_NAMES floats>}}

Usage:
    python -m apex.scripts.fit_eg_critic --harvest harvest.jsonl --out weights.json
The artifact is written ONLY when the fitted head beats the floor (unless
--force), so a regression can never be shipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..selection.learned_critic import (
    FEATURE_NAMES,
    ExecutionGroundedCritic,
    save_eg_critic,
)


def _features_from_row(row: dict[str, Any]) -> dict[str, float]:
    raw = row.get("features")
    feats: dict[str, float] = {}
    if isinstance(raw, dict):
        for name in FEATURE_NAMES:
            try:
                feats[name] = float(raw.get(name, 0.0))
            except (TypeError, ValueError):
                feats[name] = 0.0
    return feats


def load_training_groups(
    jsonl_path: "str | Path",
) -> tuple[list[tuple[dict[str, float], float]], list[list[tuple[dict[str, float], float]]]]:
    """Return ``(flat_samples, per_task_groups)`` from a harvest JSONL.

    Rows with ``gold_label is None`` are dropped. Groups are per task_id (the
    execution-tied candidate set the critic must rank)."""
    flat: list[tuple[dict[str, float], float]] = []
    by_task: dict[str, list[tuple[dict[str, float], float]]] = {}
    text = Path(jsonl_path).read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("gold_label") is None:
            continue
        try:
            label = float(row["gold_label"])
        except (TypeError, ValueError):
            continue
        feats = _features_from_row(row)
        sample = (feats, label)
        flat.append(sample)
        by_task.setdefault(str(row.get("task_id") or ""), []).append(sample)
    groups = [g for g in by_task.values() if g]
    return flat, groups


def fit_and_validate(
    train: list[tuple[dict[str, float], float]],
    groups: list[list[tuple[dict[str, float], float]]],
    *,
    holdout_frac: float = 0.3,
) -> dict[str, Any]:
    """Fit on the first ``1-holdout_frac`` of groups, validate selection accuracy
    on the held-out tail against the heuristic floor. Returns a report dict."""
    n_groups = len(groups)
    split = max(1, int(round(n_groups * (1.0 - holdout_frac)))) if n_groups > 1 else n_groups
    train_groups = groups[:split]
    holdout_groups = groups[split:] or groups[:1]
    train_samples = [s for g in train_groups for s in g] or train

    floor = ExecutionGroundedCritic()
    fitted = ExecutionGroundedCritic().fit(train_samples)
    floor_acc = floor.selection_accuracy(holdout_groups)
    fitted_acc = fitted.selection_accuracy(holdout_groups)
    return {
        "critic": fitted,
        "floor_accuracy": floor_acc,
        "fitted_accuracy": fitted_acc,
        "beats_floor": fitted_acc >= floor_acc,
        "n_train_samples": len(train_samples),
        "n_holdout_groups": len(holdout_groups),
    }


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Fit + validate the EG-Critic offline.")
    parser.add_argument("--harvest", required=True, help="harvest JSONL path")
    parser.add_argument("--out", required=True, help="output weights JSON path")
    parser.add_argument("--holdout-frac", type=float, default=0.3)
    parser.add_argument(
        "--force", action="store_true", help="write the artifact even if it does not beat the floor"
    )
    args = parser.parse_args(argv)
    flat, groups = load_training_groups(args.harvest)
    if not flat:
        print("no labeled training rows found")
        return 2
    report = fit_and_validate(flat, groups, holdout_frac=args.holdout_frac)
    print(
        f"floor_accuracy={report['floor_accuracy']:.3f} "
        f"fitted_accuracy={report['fitted_accuracy']:.3f} "
        f"beats_floor={report['beats_floor']}"
    )
    if report["beats_floor"] or args.force:
        save_eg_critic(report["critic"], args.out)
        print(f"wrote fitted EG-critic to {args.out}")
        return 0
    print("fitted head did NOT beat the floor; artifact NOT written (use --force to override)")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
