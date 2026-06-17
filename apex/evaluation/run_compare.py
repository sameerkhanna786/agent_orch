"""Compare two test-generation benchmark run directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

V3_ACCEPTANCE_THRESHOLDS = {
    "max_regressed_tasks": 0,
    "min_publishable_pass_delta": 0.0,
    "min_charged_all_pass_delta": 0.0,
}


def compare_testgen_runs(
    baseline_run_dir: str | Path,
    candidate_run_dir: str | Path,
) -> dict[str, Any]:
    baseline = _load_run_records(Path(baseline_run_dir))
    candidate = _load_run_records(Path(candidate_run_dir))
    baseline_tasks = baseline.get("tasks", {})
    candidate_tasks = candidate.get("tasks", {})
    all_ids = sorted(set(baseline_tasks) | set(candidate_tasks))
    deltas: list[dict[str, Any]] = []
    for task_id in all_ids:
        before = dict(baseline_tasks.get(task_id) or {})
        after = dict(candidate_tasks.get(task_id) or {})
        deltas.append(
            {
                "instance_id": task_id,
                "baseline_pass_at_1": float(before.get("pass_at_1") or 0.0),
                "candidate_pass_at_1": float(after.get("pass_at_1") or 0.0),
                "baseline_all_pass_at_1": float(before.get("all_pass_at_1") or 0.0),
                "candidate_all_pass_at_1": float(after.get("all_pass_at_1") or 0.0),
                "baseline_failure_class": _failure_class(before),
                "candidate_failure_class": _failure_class(after),
            }
        )
    improved_tasks = sum(
        1
        for item in deltas
        if item["candidate_all_pass_at_1"] > item["baseline_all_pass_at_1"]
        or item["candidate_pass_at_1"] > item["baseline_pass_at_1"]
    )
    regressed_tasks = sum(
        1
        for item in deltas
        if item["candidate_all_pass_at_1"] < item["baseline_all_pass_at_1"]
        or item["candidate_pass_at_1"] < item["baseline_pass_at_1"]
    )
    payload = {
        "baseline": baseline.get("summary", {}),
        "candidate": candidate.get("summary", {}),
        "task_count": len(all_ids),
        "improved_tasks": improved_tasks,
        "regressed_tasks": regressed_tasks,
        "deltas": deltas,
    }
    payload["scorecard"] = _build_v3_acceptance_scorecard(payload)
    return payload


def render_testgen_run_comparison(payload: dict[str, Any]) -> str:
    lines = [
        "# Test Generation Run Comparison",
        "",
        f"- Tasks: {payload.get('task_count', 0)}",
        f"- Improved tasks: {payload.get('improved_tasks', 0)}",
        f"- Regressed tasks: {payload.get('regressed_tasks', 0)}",
    ]
    scorecard = dict(payload.get("scorecard") or {})
    if scorecard:
        lines.extend(
            [
                f"- V3 acceptance: {'pass' if scorecard.get('accepted') else 'fail'}",
                "",
                "## Acceptance Scorecard",
                "",
                "| Check | Status | Detail |",
                "| --- | --- | --- |",
            ]
        )
        for check in scorecard.get("checks") or []:
            if not isinstance(check, dict):
                continue
            lines.append(
                "| {name} | {status} | {detail} |".format(
                    name=check.get("name") or "-",
                    status=check.get("status") or "-",
                    detail=str(check.get("detail") or "").replace("|", "\\|"),
                )
            )
    lines.extend(
        [
            "",
            "| Task | pass@1 Δ | all_pass@1 Δ | Failure Before | Failure After |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for item in payload.get("deltas", [])[:200]:
        pass_delta = float(item.get("candidate_pass_at_1") or 0.0) - float(
            item.get("baseline_pass_at_1") or 0.0
        )
        all_delta = float(item.get("candidate_all_pass_at_1") or 0.0) - float(
            item.get("baseline_all_pass_at_1") or 0.0
        )
        if pass_delta == 0 and all_delta == 0:
            continue
        lines.append(
            "| {task} | {pass_delta:+.3f} | {all_delta:+.3f} | {before} | {after} |".format(
                task=item.get("instance_id") or "-",
                pass_delta=pass_delta,
                all_delta=all_delta,
                before=item.get("baseline_failure_class") or "-",
                after=item.get("candidate_failure_class") or "-",
            )
        )
    lines.append("")
    return "\n".join(lines)


def _load_run_records(run_dir: Path) -> dict[str, Any]:
    for name in ("report.json", "summary.json", "testgeneval_report.json"):
        path = run_dir / name
        if path.exists():
            payload = _load_json(path)
            return _records_from_payload(payload)
    for path in sorted((run_dir / "official_reports").glob("*summary.json")):
        payload = _load_json(path)
        if payload:
            records = _records_from_payload(payload)
            full_path = path.with_name(path.name.replace("_summary.json", "_full.json"))
            if full_path.exists():
                try:
                    from apex.evaluation.run_artifacts import records_from_official_full_json

                    tasks = {
                        str(record.get("instance_id")): record
                        for record in records_from_official_full_json(full_path)
                        if record.get("instance_id")
                    }
                    records["tasks"] = tasks
                except Exception:
                    pass
            return records
    jsonl = run_dir / "predictions.jsonl"
    if not jsonl.exists():
        jsonl = next(iter(sorted((run_dir / "preds").glob("*.jsonl"))), jsonl)
    if jsonl.exists():
        tasks = {}
        for line in jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = str(item.get("instance_id") or item.get("task_id") or "")
            if task_id:
                tasks[task_id] = item
        return {"summary": {"task_count": len(tasks)}, "tasks": tasks}
    return {"summary": {}, "tasks": {}}


def _records_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload.get("task_results") or payload.get("tasks") or []
    tasks = {}
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            task_id = str(record.get("instance_id") or record.get("task_id") or "")
            if task_id:
                tasks[task_id] = record
    return {"summary": payload, "tasks": tasks}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _failure_class(record: dict[str, Any]) -> str:
    diagnostics = dict(record.get("diagnostics") or {})
    validation = dict(diagnostics.get("apex_validation") or record.get("apex_validation") or {})
    return str(
        validation.get("failure_class")
        or (diagnostics.get("failure_classification") or {}).get("failure_class")
        or ""
    )


def _build_v3_acceptance_scorecard(payload: dict[str, Any]) -> dict[str, Any]:
    deltas = [item for item in payload.get("deltas", []) if isinstance(item, dict)]
    publishable_delta = sum(
        float(item.get("candidate_pass_at_1") or 0.0) - float(item.get("baseline_pass_at_1") or 0.0)
        for item in deltas
    )
    charged_delta = sum(
        float(item.get("candidate_all_pass_at_1") or 0.0)
        - float(item.get("baseline_all_pass_at_1") or 0.0)
        for item in deltas
    )
    regressed_tasks = int(payload.get("regressed_tasks") or 0)
    checks = [
        {
            "name": "no_regressions",
            "status": (
                "pass"
                if regressed_tasks <= V3_ACCEPTANCE_THRESHOLDS["max_regressed_tasks"]
                else "fail"
            ),
            "detail": f"{regressed_tasks} regressed task(s)",
        },
        {
            "name": "publishable_pass_delta",
            "status": (
                "pass"
                if publishable_delta >= V3_ACCEPTANCE_THRESHOLDS["min_publishable_pass_delta"]
                else "fail"
            ),
            "detail": f"{publishable_delta:+.3f} aggregate pass@1 delta",
        },
        {
            "name": "charged_all_pass_delta",
            "status": (
                "pass"
                if charged_delta >= V3_ACCEPTANCE_THRESHOLDS["min_charged_all_pass_delta"]
                else "fail"
            ),
            "detail": f"{charged_delta:+.3f} aggregate all_pass@1 delta",
        },
    ]
    return {
        "accepted": all(check["status"] == "pass" for check in checks),
        "thresholds": dict(V3_ACCEPTANCE_THRESHOLDS),
        "publishable_pass_delta": publishable_delta,
        "charged_all_pass_delta": charged_delta,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two test-generation runs.")
    parser.add_argument("baseline_run_dir", nargs="?")
    parser.add_argument("candidate_run_dir", nargs="?")
    parser.add_argument("--baseline", dest="baseline_opt")
    parser.add_argument("--candidate", dest="candidate_opt")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    baseline_run_dir = args.baseline_opt or args.baseline_run_dir
    candidate_run_dir = args.candidate_opt or args.candidate_run_dir
    if not baseline_run_dir or not candidate_run_dir:
        parser.error("baseline and candidate run directories are required")
    payload = compare_testgen_runs(baseline_run_dir, candidate_run_dir)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_testgen_run_comparison(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
