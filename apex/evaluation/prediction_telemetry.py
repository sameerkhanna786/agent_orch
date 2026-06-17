"""Prediction JSONL telemetry helpers for test-generation benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkpointing import atomic_write_text


def enrich_testgeneval_prediction_jsonl(path: str | Path) -> dict[str, Any]:
    """Ensure every prediction record has a top-level ``apex_validation`` block."""

    target = Path(path).expanduser()
    if not target.exists():
        return {"status": "missing", "path": str(target), "records": 0}
    lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
    rendered: list[str] = []
    changed = 0
    records = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            rendered.append(line)
            continue
        if not isinstance(record, dict):
            rendered.append(line)
            continue
        records += 1
        enriched = enrich_testgeneval_prediction_record(record)
        if enriched != record:
            changed += 1
        rendered.append(json.dumps(enriched, sort_keys=True))
    if changed:
        atomic_write_text(target, "\n".join(rendered) + "\n")
    return {
        "status": "updated" if changed else "already_enriched",
        "path": str(target),
        "records": records,
        "changed": changed,
    }


def enrich_testgeneval_prediction_record(record: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(record)
    validation = dict(enriched.get("apex_validation") or {})
    diagnostics = dict(enriched.get("diagnostics") or {})
    if not validation and isinstance(diagnostics.get("apex_validation"), dict):
        validation.update(diagnostics["apex_validation"])
    if not validation:
        validation.update(_validation_from_generation_diagnostics(enriched))
    validation.setdefault(
        "prediction_quality",
        _infer_prediction_quality(enriched, validation),
    )
    validation = reconcile_authoritative_validation(validation)
    validation.setdefault("schema_version", 1)
    enriched["apex_validation"] = validation
    return enriched


def reconcile_authoritative_validation(validation: dict[str, Any]) -> dict[str, Any]:
    """Let benchmark-environment gates override host-env preflight failures.

    The tier-2 import/collect probes run in Apex's host Python environment.
    They are useful as cheap preflights, but for project-specific benchmarks
    they can fail simply because dependencies are only present in the Docker
    harness. Once the Docker final gate ships an artifact, that gate is the
    authoritative validation signal and host tier-2 failures must become
    advisory telemetry rather than the record's active quality/failure label.
    """

    reconciled = dict(validation or {})
    raw_docker_gate = reconciled.get("docker_final_acceptance_gate")
    docker_gate = dict(raw_docker_gate) if isinstance(raw_docker_gate, dict) else {}
    if str(docker_gate.get("status") or "").strip().lower() != "ship":
        return reconciled

    previous_quality = str(reconciled.get("prediction_quality") or "").strip()
    if previous_quality and previous_quality != "clean":
        reconciled.setdefault(
            "pre_authoritative_prediction_quality",
            previous_quality,
        )

    previous_failure_class = str(reconciled.get("failure_class") or "").strip()
    if previous_failure_class and previous_failure_class != "none":
        reconciled.setdefault("local_failure_class", previous_failure_class)
        reconciled["failure_class"] = "none"

    previous_repair_action = str(reconciled.get("repair_action") or "").strip()
    if previous_repair_action and previous_repair_action != "none":
        reconciled.setdefault("local_repair_action", previous_repair_action)
        reconciled["repair_action"] = "none"

    local_tier_statuses = {
        str(reconciled.get("tier_2_import") or "").strip().lower(),
        str(reconciled.get("tier_2_collect") or "").strip().lower(),
    }
    if previous_quality.startswith("tier_2_") or local_tier_statuses.intersection(
        {
            "fail",
            "failed",
            "error",
            "errored",
            "deferred_to_docker",
            "deferred_to_adapter_environment",
        }
    ):
        reconciled["local_tier_2_advisory"] = True
        reconciled["local_tier_2_authority"] = "host_environment"

    reconciled["prediction_quality"] = "clean"
    reconciled["authoritative_validation_gate"] = "docker_final_acceptance_gate"
    reconciled["authoritative_validation_status"] = "ship"
    return reconciled


def _validation_from_generation_diagnostics(record: dict[str, Any]) -> dict[str, Any]:
    generation = dict(record.get("apex_generation") or {})
    diagnostics_path = generation.get("diagnostics_path")
    if not diagnostics_path:
        return {}
    path = Path(str(diagnostics_path)).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    validation: dict[str, Any] = {}
    static = payload.get("static_validation")
    if isinstance(static, dict):
        validation["static_validation"] = static
        tier_1 = dict(static.get("tier_1_static") or {})
        if tier_1.get("status"):
            validation["tier_1_static"] = tier_1.get("status")
    if isinstance(payload.get("style_profile"), dict):
        validation["style_profile"] = payload["style_profile"]
    if "doctest_seed_count" in payload:
        validation["doctest_seed_count"] = payload.get("doctest_seed_count")
    if "artifact_count" in payload:
        validation["artifact_count"] = payload.get("artifact_count")
    return validation


def _infer_prediction_quality(
    record: dict[str, Any],
    validation: dict[str, Any],
) -> str:
    if validation.get("prediction_quality"):
        return str(validation["prediction_quality"])
    tier_3 = dict(validation.get("tier_3_run") or {})
    if float(tier_3.get("all_pass_at_1") or 0.0) >= 1.0:
        return "clean"
    if float(tier_3.get("pass_at_1") or 0.0) > 0.0:
        return "filtered_only"
    if validation.get("failure_class"):
        return "failed"
    preds = record.get("preds")
    if isinstance(preds, dict) and preds:
        return "unknown_unvalidated"
    return "unknown"
