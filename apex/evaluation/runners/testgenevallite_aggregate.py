"""Aggregate TestGenEvalLite scores from a directory of harness eval logs.

Phase 1.4 update
----------------
The upstream ``generate_report.py`` historically crashed with
``KeyError("baseline_covs")`` on certain ``kjain14/testgenevallite``
rows, so APEX shipped this parallel log-marker re-aggregator as the
authoritative pass@1 / mutation / coverage source. As of Phase 1.4
the upstream report is fixed defensively (see
``apex/evaluation/upstream_patches/testgeneval/baseline_covs_keyerror.patch``)
and is now the headline-number source of truth. The parallel runner
orchestration in this module is still useful for speed, but its
pass@1 computation is now treated as a *legacy diagnostic* whose only
purpose is to feed the fairness-audit delta:
``AggregateScores.legacy_pass_at_1_filtered`` /
``legacy_pass_at_1_unfiltered`` are exposed alongside the canonical
fields and the runner records them in the audit's ``notes``.

Why we still keep the log-marker classifier:
  * pass@1 in TestGenEval = ``All Tests Passed`` marker present
    (the *filtered* subset passes after dropping individual failing
    tests). The final ``>>>>> ...`` line is often
    ``Unfiltered Tests Failed`` even for tasks that pass@1=1, so a
    naive ``last marker`` reader undercounts pass@1 dramatically.
  * The harness writes one log per (task_id × model_name) pair.
    Resumed runs may produce duplicates under the bare ``instance_id``;
    we match against the dataset's canonical ``id`` first, then fall
    back to ``instance_id``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TaskScore:
    id: str
    status: str  # unfiltered_pass | filtered_pass | failed | no_signal | no_log
    mutation: Optional[float] = None
    coverage: Optional[float] = None
    log: str = ""
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "status": self.status,
            "mutation": self.mutation,
            "coverage": self.coverage,
            "log": self.log,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class AggregateScores:
    task_count: int = 0
    pass_at_1_filtered: float = 0.0
    pass_at_1_unfiltered: float = 0.0
    mutation_score_mean: float = 0.0
    mutation_n: int = 0
    coverage_mean: float = 0.0
    coverage_n: int = 0
    failed: int = 0
    no_signal: int = 0
    no_log: int = 0
    # Audit H1: surface scoring-tool gaps so users see when their run
    # would have N=0 because the host venv is missing coverage / mutation
    # tooling, not because the candidates were bad. ``coverage_missing``
    # counts tasks that scored 0 despite the log existing because no
    # ``CoverageLOG`` line was emitted. Same for mutation.
    coverage_missing: int = 0
    mutation_missing: int = 0
    coverage_present_count: int = 0
    mutation_present_count: int = 0
    coverage_denominator: int = 0
    mutation_denominator: int = 0
    stale_or_duplicate_log_candidates: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "task_count": self.task_count,
            "pass_at_1_filtered": round(self.pass_at_1_filtered, 4),
            "pass_at_1_unfiltered": round(self.pass_at_1_unfiltered, 4),
            # Phase 1.4 alias: same numeric values, but flagged as legacy
            # so downstream callers (and the fairness audit notes) can
            # tell at a glance that they came from the parallel
            # log-marker classifier rather than the upstream report.
            "legacy_pass_at_1_filtered": round(self.pass_at_1_filtered, 4),
            "legacy_pass_at_1_unfiltered": round(self.pass_at_1_unfiltered, 4),
            "mutation_score_mean": round(self.mutation_score_mean, 2),
            "mutation_n": self.mutation_n,
            "coverage_mean": round(self.coverage_mean, 2),
            "coverage_n": self.coverage_n,
            "failed": self.failed,
            "no_signal": self.no_signal,
            "no_log": self.no_log,
            "coverage_missing": self.coverage_missing,
            "mutation_missing": self.mutation_missing,
            "coverage_present_count": self.coverage_present_count,
            "mutation_present_count": self.mutation_present_count,
            "coverage_denominator": self.coverage_denominator,
            "mutation_denominator": self.mutation_denominator,
            "stale_or_duplicate_log_candidates": self.stale_or_duplicate_log_candidates,
        }

    def as_legacy_diagnostic(self) -> dict[str, object]:
        """Return a dict tagged as ``legacy_*`` for fairness-audit notes.

        Phase 1.4 stops trusting the parallel pass@1 computation as the
        headline-number source. We still want it surfaced so humans can
        see the divergence vs. the upstream report — this is the format
        the runner attaches to the fairness audit's ``notes`` and to
        the runner status payload's ``parallel_aggregator_legacy`` key.
        """
        return {
            "legacy_pass_at_1_filtered": round(self.pass_at_1_filtered, 4),
            "legacy_pass_at_1_unfiltered": round(self.pass_at_1_unfiltered, 4),
            "legacy_mutation_score_mean": round(self.mutation_score_mean, 2),
            "legacy_coverage_mean": round(self.coverage_mean, 2),
            "task_count": self.task_count,
            "no_log": self.no_log,
            "scored_via": "parallel_log_marker_aggregator",
            "headline_source_of_truth": "upstream_generate_report",
        }


@dataclass(frozen=True)
class OfficialEvalLogParse:
    status: str
    mutation: Optional[float] = None
    coverage: Optional[float] = None
    has_mutation_log: bool = False
    has_coverage_log: bool = False

    @property
    def has_scored_status(self) -> bool:
        return self.status in {"unfiltered_pass", "filtered_pass", "failed"}

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "mutation": self.mutation,
            "coverage": self.coverage,
            "has_mutation_log": self.has_mutation_log,
            "has_coverage_log": self.has_coverage_log,
            "has_scored_status": self.has_scored_status,
        }


def aggregate_eval_logs(
    *,
    log_dir: Path,
    model_name: str,
    dataset_name: str = "kjain14/testgenevallite",
    split: str = "test",
) -> tuple[AggregateScores, list[TaskScore]]:
    """Read eval logs in ``log_dir`` and produce the aggregate + per-task scores.

    A ``filtered_pass`` task has ``All Tests Passed`` somewhere in the log
    (the harness's filtered-subset re-run succeeded). An
    ``unfiltered_pass`` task additionally has all original tests passing
    on the first try — these are a strict subset of filtered_pass.
    """

    from datasets import load_dataset

    suffix = f".{model_name}.full.eval.log"
    log_files = [fn for fn in os.listdir(log_dir) if fn.endswith(suffix)]
    files_by_token: dict[str, str] = {}
    stale_or_duplicate = 0
    for fn in log_files:
        token = fn[: -len(suffix)]
        existing = files_by_token.get(token)
        if existing:
            stale_or_duplicate += 1
            existing_path = Path(log_dir) / existing
            candidate_path = Path(log_dir) / fn
            try:
                if candidate_path.stat().st_mtime > existing_path.stat().st_mtime:
                    files_by_token[token] = fn
            except OSError:
                pass
        else:
            files_by_token[token] = fn

    ds = load_dataset(dataset_name, split=split)
    per_task: list[TaskScore] = []
    mut_values: list[float] = []
    cov_values: list[float] = []
    counts = {
        "unfiltered_pass": 0,
        "filtered_pass": 0,
        "failed": 0,
        "no_signal": 0,
        "no_log": 0,
        "coverage_missing": 0,
        "mutation_missing": 0,
        "coverage_present_count": 0,
        "mutation_present_count": 0,
        "coverage_denominator": 0,
        "mutation_denominator": 0,
    }

    for row in ds:
        did = str(row.get("id") or row["instance_id"])
        iid = str(row["instance_id"])
        fn = files_by_token.get(did) or files_by_token.get(iid)
        if not fn:
            per_task.append(TaskScore(id=did, status="no_log"))
            counts["no_log"] += 1
            continue
        # Audit M4: a truncated/missing log used to crash the whole
        # aggregator; we now degrade to "no_log" status for that one
        # task so the rest of the report still emits.
        log_path = Path(log_dir) / fn
        if not log_path.exists():
            per_task.append(TaskScore(id=did, status="no_log", log=fn))
            counts["no_log"] += 1
            continue
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            per_task.append(TaskScore(id=did, status="no_log", log=fn))
            counts["no_log"] += 1
            continue
        parsed = parse_official_eval_log(text)
        status = parsed.status
        counts[status] += 1
        mscore = parsed.mutation
        if mscore is not None:
            mut_values.append(mscore)
            counts["mutation_present_count"] += 1
        if parsed.has_scored_status:
            counts["mutation_denominator"] += 1
        if parsed.has_scored_status and mscore is None:
            counts["mutation_missing"] += 1
        cscore = parsed.coverage
        if cscore is not None:
            cov_values.append(cscore)
            counts["coverage_present_count"] += 1
        if parsed.has_scored_status:
            counts["coverage_denominator"] += 1
        if parsed.has_scored_status and cscore is None:
            counts["coverage_missing"] += 1
        per_task.append(
            TaskScore(
                id=did,
                status=status,
                mutation=mscore,
                coverage=cscore,
                log=fn,
                diagnostics={"eval_log_parse": parsed.to_dict()},
            )
        )

    total = len(per_task)
    filtered = counts["unfiltered_pass"] + counts["filtered_pass"]
    aggregate = AggregateScores(
        task_count=total,
        pass_at_1_filtered=filtered / total if total else 0.0,
        pass_at_1_unfiltered=counts["unfiltered_pass"] / total if total else 0.0,
        mutation_score_mean=(sum(mut_values) / len(mut_values) if mut_values else 0.0),
        mutation_n=len(mut_values),
        coverage_mean=sum(cov_values) / len(cov_values) if cov_values else 0.0,
        coverage_n=len(cov_values),
        failed=counts["failed"],
        no_signal=counts["no_signal"],
        no_log=counts["no_log"],
        coverage_missing=counts["coverage_missing"],
        mutation_missing=counts["mutation_missing"],
        coverage_present_count=counts["coverage_present_count"],
        mutation_present_count=counts["mutation_present_count"],
        coverage_denominator=counts["coverage_denominator"],
        mutation_denominator=counts["mutation_denominator"],
        stale_or_duplicate_log_candidates=stale_or_duplicate,
    )
    return aggregate, per_task


def parse_official_eval_log(text: str) -> OfficialEvalLogParse:
    mutation_match = re.search(r"MutationLOG:\s*([\d.]+)%", text)
    coverage_match = re.search(r"CoverageLOG:\s*([\d.]+)%", text)
    return OfficialEvalLogParse(
        status=_classify(text),
        mutation=float(mutation_match.group(1)) if mutation_match else None,
        coverage=float(coverage_match.group(1)) if coverage_match else None,
        has_mutation_log=bool(mutation_match),
        has_coverage_log=bool(coverage_match),
    )


def _classify(text: str) -> str:
    """Map raw eval-log content to a canonical status.

    Order matters: ``Unfiltered Tests Passed`` implies
    ``All Tests Passed`` so we check the strict variant first.
    """

    if "Unfiltered Tests Passed" in text:
        return "unfiltered_pass"
    if "All Tests Passed" in text:
        return "filtered_pass"
    if "Unfiltered Tests Failed" in text or "Tests Errored" in text or "Some Tests Failed" in text:
        return "failed"
    return "no_signal"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate TestGenEvalLite eval logs into a pass@1 / mutation / "
            "coverage report. Use after run_evaluation.py finishes."
        )
    )
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-name", default="kjain14/testgenevallite")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the aggregate JSON report.",
    )
    args = parser.parse_args(argv)

    aggregate, per_task = aggregate_eval_logs(
        log_dir=Path(args.log_dir),
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        split=args.split,
    )
    payload = {
        "aggregate": aggregate.to_dict(),
        "tasks": [t.to_dict() for t in per_task],
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(json.dumps(aggregate.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
