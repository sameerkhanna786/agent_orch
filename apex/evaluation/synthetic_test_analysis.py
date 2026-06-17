"""
Offline synthesized-vs-gold test comparison helpers for SWE-Bench Pro.

These helpers are intentionally evaluation-only. They should be used after a
run completes to compare Apex-generated synthetic tests against benchmark gold
tests without exposing benchmark-private tests to the solving pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from ..core.cli_backend import CLIModelClient, extract_total_tokens
from ..core.config import LLMConfig
from ..core.filesystem import copy_tree
from ..core.llm import LLMClient, Message
from ..test_portfolio import (
    apply_test_portfolio_promotion,
    evaluate_field_path_negative_shape_coverage,
    extract_data_contract_field_paths,
    extract_issue_contract_targets,
    normalize_test_suite_artifact_payload,
)
from .swebench_pro_benchmark import (
    SWEBENCH_PRO_DATASET_NAME,
    SWEBENCH_PRO_DATASET_SPLIT,
    SWEBenchProHarness,
    SWEBenchProTask,
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_generated_test_portfolio(
    apex_result_path: str | Path,
    *,
    rollout_id: Optional[int] = None,
) -> dict[str, Any]:
    payload = _load_json(apex_result_path)
    selected_rollout_id = (
        int(rollout_id) if rollout_id is not None else int(payload.get("selected_rollout_id") or 0)
    )
    for rollout in list(payload.get("rollout_summaries") or []):
        if int(rollout.get("rollout_id") or -1) != selected_rollout_id:
            continue
        return normalize_test_suite_artifact_payload(rollout.get("test_suite_artifact") or {})
    return normalize_test_suite_artifact_payload({})


def _parse_patch_paths(patch_text: str) -> list[str]:
    paths: list[str] = []
    for line in str(patch_text or "").splitlines():
        if not line.startswith("diff --git a/"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        new_path = parts[3].removeprefix("b/")
        if new_path and new_path != "/dev/null":
            paths.append(new_path)
    return sorted(set(paths))


def _extract_patch_sections(patch_text: str) -> dict[str, str]:
    sections = [
        section for section in str(patch_text or "").split("diff --git ") if section.strip()
    ]
    by_path: dict[str, str] = {}
    for section in sections:
        header, _, body = section.partition("\n")
        header_parts = header.split()
        if len(header_parts) < 2:
            continue
        new_path = header_parts[1].removeprefix("b/")
        by_path[new_path] = f"diff --git {header}\n{body}"
    return by_path


def materialize_gold_test_files(
    *,
    base_repo_root: str | Path,
    task: SWEBenchProTask,
) -> dict[str, Any]:
    repo_root = Path(base_repo_root).resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"Base repo root does not exist: {repo_root}")

    benchmark_paths = list(task.benchmark_test_files or []) or _parse_patch_paths(task.test_patch)
    patch_sections = _extract_patch_sections(task.test_patch)

    with tempfile.TemporaryDirectory(prefix="apex-gold-tests-") as temp_dir:
        temp_repo = Path(temp_dir) / "repo"
        snapshot_ignore = shutil.ignore_patterns(
            ".git",
            ".hg",
            ".jj",
            ".sl",
            ".svn",
            ".venv",
            "venv",
            "env",
            "node_modules",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
        )
        copy_tree(
            repo_root,
            temp_repo,
            ignore=snapshot_ignore,
            restrict_symlinks_to_root=True,
        )
        apply_error = ""
        if str(task.test_patch or "").strip():
            result = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "-"],
                cwd=str(temp_repo),
                input=task.test_patch,
                text=True,
                capture_output=True,
            )
            if result.returncode != 0:
                apply_error = (result.stderr or result.stdout or "").strip()

        gold_files: dict[str, dict[str, Any]] = {}
        for rel_path in benchmark_paths:
            file_payload: dict[str, Any] = {
                "path": rel_path,
                "patch": patch_sections.get(rel_path, ""),
            }
            materialized = temp_repo / rel_path
            if apply_error:
                file_payload["materialization_error"] = apply_error
            if materialized.exists():
                file_payload["content"] = materialized.read_text()
            gold_files[rel_path] = file_payload
        return {
            "benchmark_test_files": benchmark_paths,
            "gold_test_files": gold_files,
        }


def summarize_test_portfolio(portfolio: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_test_suite_artifact_payload(portfolio or {})
    return {
        "artifact_count": len(list(normalized.get("test_artifacts") or [])),
        "paths": [
            str(item.get("path") or "")
            for item in list(normalized.get("test_artifacts") or [])
            if str(item.get("path") or "").strip()
        ],
        "targets": sorted(
            {
                str(target)
                for item in list(normalized.get("test_artifacts") or [])
                for target in list(
                    item.get("contract_targets") or item.get("reference_targets") or []
                )
                if str(target).strip()
            }
        ),
    }


_SEMANTIC_REVIEW_STATUS_VALUES = {
    "covered",
    "partially_covered",
    "missing",
    "contradictory",
}
_SEMANTIC_REVIEW_STRENGTH_VALUES = {
    "stronger",
    "equivalent",
    "weaker",
    "unclear",
}
_SEMANTIC_REVIEW_SEVERITY_VALUES = {
    "critical",
    "moderate",
    "minor",
}
_SEMANTIC_REVIEW_VERDICT_VALUES = {
    "equivalent_or_stronger",
    "mostly_equivalent",
    "material_gaps",
    "contradictory",
}
_SEMANTIC_REVIEW_METADATA_FIELDS = (
    "judge_backend",
    "judge_model",
    "judge_duration_seconds",
    "judge_usage",
    "judge_total_tokens",
)
_SEMANTIC_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "overall_verdict": {
            "type": "string",
            "enum": sorted(_SEMANTIC_REVIEW_VERDICT_VALUES),
        },
        "confidence": {"type": "number"},
        "gold_behavior_obligations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "obligation": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": sorted(_SEMANTIC_REVIEW_STATUS_VALUES),
                    },
                    "assertion_strength": {
                        "type": "string",
                        "enum": sorted(_SEMANTIC_REVIEW_STRENGTH_VALUES),
                    },
                    "severity": {
                        "type": "string",
                        "enum": sorted(_SEMANTIC_REVIEW_SEVERITY_VALUES),
                    },
                    "generated_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "gold_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "notes": {"type": "string"},
                },
                "required": [
                    "obligation",
                    "status",
                    "assertion_strength",
                    "severity",
                    "generated_evidence",
                    "gold_evidence",
                ],
            },
        },
        "critical_gaps": {
            "type": "array",
            "items": {"type": "string"},
        },
        "weaker_assertions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "extra_generated_behaviors": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "summary",
        "overall_verdict",
        "confidence",
        "gold_behavior_obligations",
        "critical_gaps",
        "weaker_assertions",
        "extra_generated_behaviors",
    ],
}


def _truncate_semantic_review_text(text: str, *, max_chars: int) -> str:
    raw = str(text or "").strip()
    if len(raw) <= max_chars:
        return raw
    head = raw[: max_chars - 16].rstrip()
    return f"{head}\n...[truncated]"


def _semantic_review_artifact_payload(
    entry: dict[str, Any],
    *,
    max_content_chars: int = 5000,
) -> dict[str, Any]:
    content = str(entry.get("content") or "")
    return {
        "path": str(entry.get("path") or ""),
        "summary": str(entry.get("summary") or ""),
        "justification": str(entry.get("justification") or ""),
        "contract_targets": [
            str(item) for item in list(entry.get("contract_targets") or []) if str(item).strip()
        ],
        "contract_axes": [
            str(item) for item in list(entry.get("contract_axes") or []) if str(item).strip()
        ],
        "test_descriptions": [
            str(item) for item in list(entry.get("test_descriptions") or []) if str(item).strip()
        ][:8],
        "properties": [
            str(item) for item in list(entry.get("properties") or []) if str(item).strip()
        ][:8],
        "content_excerpt": _truncate_semantic_review_text(
            content,
            max_chars=max_content_chars,
        ),
        "content_truncated": len(content) > max_content_chars,
    }


def _semantic_review_portfolio_payload(
    portfolio: dict[str, Any],
    *,
    max_artifacts: int = 12,
    max_content_chars: int = 5000,
) -> dict[str, Any]:
    entries = list(portfolio.get("test_artifacts") or [])
    rendered_entries = [
        _semantic_review_artifact_payload(
            dict(entry or {}),
            max_content_chars=max_content_chars,
        )
        for entry in entries[:max_artifacts]
    ]
    omitted_paths = [
        str(entry.get("path") or "")
        for entry in entries[max_artifacts:]
        if str(entry.get("path") or "").strip()
    ]
    return {
        "summary": str(portfolio.get("summary") or ""),
        "portfolio_summary": str(portfolio.get("portfolio_summary") or ""),
        "promotion_summary": str(portfolio.get("promotion_summary") or ""),
        "contract_hypotheses": [
            str(item)
            for item in list(portfolio.get("contract_hypotheses") or [])
            if str(item).strip()
        ][:16],
        "test_descriptions": [
            str(item)
            for item in list(portfolio.get("test_descriptions") or [])
            if str(item).strip()
        ][:16],
        "artifacts": rendered_entries,
        "omitted_artifact_paths": omitted_paths,
    }


def build_semantic_test_review_prompt(
    *,
    packet: dict[str, Any],
    generated_portfolio: dict[str, Any],
    reference_portfolio: dict[str, Any],
    reference_label: str = "gold suite",
) -> str:
    required_targets = [
        str(target)
        for target in list(packet.get("required_contract_targets") or [])
        if str(target).strip()
    ]
    normalized_reference_label = str(reference_label or "").strip() or "gold suite"
    prompt_payload = {
        "instance_id": str(packet.get("instance_id") or ""),
        "repo": str(packet.get("repo") or ""),
        "required_contract_targets": required_targets,
        "reference_label": normalized_reference_label,
        "issue_description": str(packet.get("issue_description") or ""),
        "generated_suite": _semantic_review_portfolio_payload(generated_portfolio),
        "reference_suite": _semantic_review_portfolio_payload(reference_portfolio),
    }
    return "\n".join(
        [
            (
                "Review whether the generated suite semantically covers the same "
                f"behavioral obligations as the {normalized_reference_label}."
            ),
            "",
            "Rules:",
            (
                "- Extract the distinct behavioral obligations from the "
                f"{normalized_reference_label}."
            ),
            "- For each obligation, mark one status: covered, partially_covered, missing, or contradictory.",
            (
                "- Judge assertion strength relative to the "
                f"{normalized_reference_label}: stronger, equivalent, weaker, or unclear."
            ),
            (
                "- Additional generated tests are fine unless they contradict the "
                f"{normalized_reference_label} behavior."
            ),
            "- Be conservative: overlapping target names are not enough without matching assertions.",
            "- Cite concrete evidence from both suites using short path/snippet strings.",
            "- Mark severity as critical when missing coverage would plausibly let a bad patch pass.",
            (
                "- Return the reference-suite obligations in the "
                "`gold_behavior_obligations` field and the reference-suite citations "
                "in each obligation's `gold_evidence` field."
            ),
            "",
            "Return only JSON matching the provided schema.",
            "",
            json.dumps(prompt_payload, indent=2),
        ]
    )


def _bounded_review_float(
    value: Any,
    *,
    default: float = 0.0,
    lower: float = 0.0,
    upper: float = 1.0,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(lower, min(parsed, upper))


def _semantic_review_requires_obligations(portfolio: dict[str, Any]) -> bool:
    if str(portfolio.get("summary") or "").strip():
        return True
    if list(portfolio.get("contract_hypotheses") or []):
        return True
    if list(portfolio.get("test_descriptions") or []):
        return True
    for artifact in list(portfolio.get("test_artifacts") or []):
        artifact_entry = dict(artifact or {})
        for key in ("path", "summary", "justification", "content"):
            if str(artifact_entry.get(key) or "").strip():
                return True
        if list(artifact_entry.get("test_descriptions") or []):
            return True
        if list(artifact_entry.get("properties") or []):
            return True
    return False


def _validate_semantic_review_payload(
    payload: dict[str, Any],
    *,
    require_obligations: bool,
) -> None:
    if not isinstance(payload, dict):
        raise RuntimeError("Semantic review judge returned a non-object payload.")

    missing_top_level = [
        key
        for key in (
            "summary",
            "overall_verdict",
            "confidence",
            "gold_behavior_obligations",
            "critical_gaps",
            "weaker_assertions",
            "extra_generated_behaviors",
        )
        if key not in payload
    ]
    if missing_top_level:
        raise RuntimeError(
            "Semantic review judge omitted required fields: " + ", ".join(sorted(missing_top_level))
        )

    verdict = str(payload.get("overall_verdict") or "").strip().lower()
    if verdict not in _SEMANTIC_REVIEW_VERDICT_VALUES:
        raise RuntimeError(
            f"Semantic review judge returned invalid verdict: {payload.get('overall_verdict')!r}"
        )
    try:
        float(payload.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Semantic review judge returned a non-numeric confidence.") from exc

    obligations = payload.get("gold_behavior_obligations")
    if not isinstance(obligations, list):
        raise RuntimeError("Semantic review judge returned non-list obligations.")
    if require_obligations and not obligations:
        raise RuntimeError(
            "Semantic review judge returned zero obligations despite non-empty gold tests."
        )
    for index, raw in enumerate(obligations):
        if not isinstance(raw, dict):
            raise RuntimeError(f"Semantic review obligation {index} is not an object.")
        missing_obligation_fields = [
            key
            for key in (
                "obligation",
                "status",
                "assertion_strength",
                "severity",
                "generated_evidence",
                "gold_evidence",
            )
            if key not in raw
        ]
        if missing_obligation_fields:
            raise RuntimeError(
                f"Semantic review obligation {index} omitted fields: "
                + ", ".join(sorted(missing_obligation_fields))
            )
        if str(raw.get("status") or "").strip().lower() not in _SEMANTIC_REVIEW_STATUS_VALUES:
            raise RuntimeError(f"Semantic review obligation {index} returned invalid status.")
        if (
            str(raw.get("assertion_strength") or "").strip().lower()
            not in _SEMANTIC_REVIEW_STRENGTH_VALUES
        ):
            raise RuntimeError(
                f"Semantic review obligation {index} returned invalid assertion strength."
            )
        if str(raw.get("severity") or "").strip().lower() not in _SEMANTIC_REVIEW_SEVERITY_VALUES:
            raise RuntimeError(f"Semantic review obligation {index} returned invalid severity.")
        if not isinstance(raw.get("generated_evidence"), list):
            raise RuntimeError(
                f"Semantic review obligation {index} returned non-list generated evidence."
            )
        if not isinstance(raw.get("gold_evidence"), list):
            raise RuntimeError(
                f"Semantic review obligation {index} returned non-list gold evidence."
            )

    for field_name in ("critical_gaps", "weaker_assertions", "extra_generated_behaviors"):
        if not isinstance(payload.get(field_name), list):
            raise RuntimeError(f"Semantic review judge returned non-list {field_name}.")


def _normalize_semantic_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "summary": str(payload.get("summary") or ""),
        "overall_verdict": str(payload.get("overall_verdict") or "").strip().lower(),
        "confidence": _bounded_review_float(payload.get("confidence")),
        "gold_behavior_obligations": [],
        "critical_gaps": [
            str(item).strip()
            for item in list(payload.get("critical_gaps") or [])
            if str(item).strip()
        ],
        "weaker_assertions": [
            str(item).strip()
            for item in list(payload.get("weaker_assertions") or [])
            if str(item).strip()
        ],
        "extra_generated_behaviors": [
            str(item).strip()
            for item in list(payload.get("extra_generated_behaviors") or [])
            if str(item).strip()
        ],
    }
    if normalized["overall_verdict"] not in _SEMANTIC_REVIEW_VERDICT_VALUES:
        normalized["overall_verdict"] = "mostly_equivalent"

    for raw in list(payload.get("gold_behavior_obligations") or []):
        entry = dict(raw or {})
        status = str(entry.get("status") or "").strip().lower()
        strength = str(entry.get("assertion_strength") or "").strip().lower()
        severity = str(entry.get("severity") or "").strip().lower()
        normalized["gold_behavior_obligations"].append(
            {
                "obligation": str(entry.get("obligation") or "").strip(),
                "status": (status if status in _SEMANTIC_REVIEW_STATUS_VALUES else "missing"),
                "assertion_strength": (
                    strength if strength in _SEMANTIC_REVIEW_STRENGTH_VALUES else "unclear"
                ),
                "severity": (
                    severity if severity in _SEMANTIC_REVIEW_SEVERITY_VALUES else "moderate"
                ),
                "generated_evidence": [
                    str(item).strip()
                    for item in list(entry.get("generated_evidence") or [])
                    if str(item).strip()
                ][:8],
                "gold_evidence": [
                    str(item).strip()
                    for item in list(entry.get("gold_evidence") or [])
                    if str(item).strip()
                ][:8],
                "notes": str(entry.get("notes") or "").strip(),
            }
        )
    for field_name in _SEMANTIC_REVIEW_METADATA_FIELDS:
        if field_name == "judge_usage":
            normalized[field_name] = dict(payload.get(field_name) or {})
            continue
        if field_name == "judge_duration_seconds":
            normalized[field_name] = round(
                _bounded_review_float(payload.get(field_name), upper=1000000.0),
                4,
            )
            continue
        if field_name == "judge_total_tokens":
            try:
                normalized[field_name] = max(0, int(payload.get(field_name) or 0))
            except (TypeError, ValueError):
                normalized[field_name] = 0
            continue
        normalized[field_name] = str(payload.get(field_name) or "").strip()
    return normalized


def summarize_semantic_test_review(review: dict[str, Any]) -> dict[str, Any]:
    obligations = list(dict(review or {}).get("gold_behavior_obligations") or [])
    total = len(obligations)
    covered = sum(1 for item in obligations if str(item.get("status") or "") == "covered")
    partial = sum(1 for item in obligations if str(item.get("status") or "") == "partially_covered")
    missing = sum(1 for item in obligations if str(item.get("status") or "") == "missing")
    contradictory = sum(
        1 for item in obligations if str(item.get("status") or "") == "contradictory"
    )
    weaker = sum(1 for item in obligations if str(item.get("assertion_strength") or "") == "weaker")
    critical = sum(
        1
        for item in obligations
        if str(item.get("severity") or "") == "critical"
        and str(item.get("status") or "") in {"missing", "contradictory"}
    )
    strict_recall = (covered / total) if total else 1.0
    lenient_recall = ((covered + partial) / total) if total else 1.0
    verdict = str(review.get("overall_verdict") or "").strip().lower()
    has_material_gaps = (
        verdict in {"material_gaps", "contradictory"}
        or missing > 0
        or contradictory > 0
        or critical > 0
        or bool(list(review.get("critical_gaps") or []))
    )
    return {
        "semantic_review_verdict": (
            verdict if verdict in _SEMANTIC_REVIEW_VERDICT_VALUES else "mostly_equivalent"
        ),
        "semantic_review_confidence": _bounded_review_float(review.get("confidence")),
        "semantic_review_obligation_count": total,
        "semantic_review_covered_obligation_count": covered,
        "semantic_review_partially_covered_obligation_count": partial,
        "semantic_review_missing_obligation_count": missing,
        "semantic_review_contradictory_obligation_count": contradictory,
        "semantic_review_weaker_assertion_count": weaker,
        "semantic_review_critical_gap_count": critical,
        "semantic_review_strict_behavioral_recall": round(strict_recall, 4),
        "semantic_review_lenient_behavioral_recall": round(lenient_recall, 4),
        "semantic_review_no_material_gaps": not has_material_gaps,
        "semantic_review_no_weaker_assertions": weaker == 0,
    }


def attach_semantic_review_to_comparison(
    comparison: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(comparison or {})
    normalized_review = _normalize_semantic_review_payload(review)
    enriched["semantic_review"] = normalized_review
    enriched["semantic_review_summary"] = summarize_semantic_test_review(normalized_review)
    coverage_summary = dict(enriched.get("coverage_summary") or {})
    coverage_summary.update(enriched["semantic_review_summary"])
    enriched["coverage_summary"] = coverage_summary
    return enriched


def review_generated_vs_reference_test_semantics(
    *,
    packet: dict[str, Any],
    generated_portfolio: dict[str, Any],
    reference_portfolio: dict[str, Any],
    judge_config: LLMConfig,
    working_dir: str,
    system_prompt: Optional[str] = None,
    reference_label: str = "reference suite",
) -> dict[str, Any]:
    prompt = build_semantic_test_review_prompt(
        packet=packet,
        generated_portfolio=generated_portfolio,
        reference_portfolio=reference_portfolio,
        reference_label=reference_label,
    )
    normalized_reference_label = str(reference_label or "").strip() or "reference suite"
    effective_system_prompt = system_prompt or (
        "You are a rigorous semantic test-suite reviewer. "
        "Determine whether the generated tests cover the same behavioral obligations "
        f"as the {normalized_reference_label}. Be conservative, cite evidence, and "
        "return only JSON."
    )
    if judge_config.is_cli_backend:
        result = CLIModelClient(judge_config).run_structured_prompt(
            prompt=prompt,
            working_dir=working_dir,
            schema=_SEMANTIC_REVIEW_SCHEMA,
            system_prompt=effective_system_prompt,
            allow_edits=False,
        )
        if not result.success or not isinstance(result.parsed_json, dict):
            raise RuntimeError(result.error or "Semantic review judge failed.")
        _validate_semantic_review_payload(
            result.parsed_json,
            require_obligations=_semantic_review_requires_obligations(reference_portfolio),
        )
        normalized = _normalize_semantic_review_payload(result.parsed_json)
        usage = dict(result.usage or {})
        normalized["judge_backend"] = str(judge_config.backend.value)
        normalized["judge_model"] = str(judge_config.model or "")
        normalized["judge_duration_seconds"] = round(float(result.duration_seconds or 0.0), 4)
        normalized["judge_usage"] = usage
        normalized["judge_total_tokens"] = int(extract_total_tokens(usage))
        return normalized

    response = LLMClient(judge_config).chat(
        [
            Message(role="system", content=effective_system_prompt),
            Message(role="user", content=prompt),
        ]
    )
    try:
        parsed = json.loads(str(response.content or "").strip() or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("Semantic review judge did not return valid JSON.") from exc
    _validate_semantic_review_payload(
        parsed,
        require_obligations=_semantic_review_requires_obligations(reference_portfolio),
    )
    normalized = _normalize_semantic_review_payload(parsed)
    usage = dict(response.usage or {})
    normalized["judge_backend"] = str(judge_config.backend.value)
    normalized["judge_model"] = str(judge_config.model or "")
    normalized["judge_duration_seconds"] = round(float(response.latency_ms or 0.0) / 1000.0, 4)
    normalized["judge_usage"] = usage
    normalized["judge_total_tokens"] = int(usage.get("total_tokens") or 0)
    return normalized


def review_generated_vs_gold_test_semantics(
    *,
    packet: dict[str, Any],
    generated_portfolio: dict[str, Any],
    gold_portfolio: dict[str, Any],
    judge_config: LLMConfig,
    working_dir: str,
    system_prompt: Optional[str] = None,
) -> dict[str, Any]:
    return review_generated_vs_reference_test_semantics(
        packet=packet,
        generated_portfolio=generated_portfolio,
        reference_portfolio=gold_portfolio,
        judge_config=judge_config,
        working_dir=working_dir,
        system_prompt=system_prompt,
        reference_label="gold suite",
    )


def build_swebench_gold_comparison_packet_for_task(
    *,
    task: SWEBenchProTask,
    base_repo_root: str | Path,
    generated_portfolio: dict[str, Any],
    test_command: Optional[str] = None,
    issue_description: Optional[str] = None,
) -> dict[str, Any]:
    normalized_generated = normalize_test_suite_artifact_payload(generated_portfolio or {})
    effective_issue_description = issue_description
    if effective_issue_description is None:
        effective_issue_description = task.build_issue_description(
            test_command,
            include_benchmark_guardrails=False,
            include_benchmark_metadata=False,
            include_selected_test_targets=False,
            include_required_tests=False,
        )
    return {
        "instance_id": task.instance_id,
        "repo": task.repo,
        "generated_portfolio": normalized_generated,
        "generated_summary": summarize_test_portfolio(normalized_generated),
        "gold_tests": materialize_gold_test_files(
            base_repo_root=base_repo_root,
            task=task,
        ),
        "issue_description": effective_issue_description,
        "issue_field_paths": extract_data_contract_field_paths([effective_issue_description]),
        "required_contract_targets": extract_issue_contract_targets(effective_issue_description),
    }


def build_swebench_gold_comparison_packet(
    *,
    apex_result_path: str | Path,
    instance_id: str,
    rollout_id: Optional[int] = None,
    worktree_path: str | Path | None = None,
    dataset_name: str = SWEBENCH_PRO_DATASET_NAME,
    dataset_split: str = SWEBENCH_PRO_DATASET_SPLIT,
) -> dict[str, Any]:
    apex_payload = _load_json(apex_result_path)
    generated_portfolio = load_generated_test_portfolio(
        apex_result_path,
        rollout_id=rollout_id,
    )
    worktree_value = str(
        worktree_path
        if worktree_path is not None
        else apex_payload.get("selected_worktree_path") or ""
    ).strip()
    if not worktree_value:
        raise ValueError("A worktree path is required to reconstruct gold test files.")
    effective_worktree = Path(worktree_value).resolve()

    with tempfile.TemporaryDirectory(prefix="apex-swebench-task-") as temp_dir:
        harness = SWEBenchProHarness(
            output_dir=temp_dir,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
        )
        task = harness.load_task(instance_id=instance_id)
        base_repo_root = effective_worktree
        if not base_repo_root.exists():
            base_repo_root = Path(temp_dir) / "prepared_repo"
            harness.prepare_repo(task, base_repo_root)
        build_agent_test_command = getattr(harness, "build_agent_test_command", None)
        test_command = (
            build_agent_test_command(task, base_repo_root)
            if callable(build_agent_test_command)
            else None
        )
        return build_swebench_gold_comparison_packet_for_task(
            task=task,
            base_repo_root=base_repo_root,
            generated_portfolio=generated_portfolio,
            test_command=test_command,
        )


def _infer_gold_file_contract_targets(
    rel_path: str,
    content: str,
    *,
    required_contract_targets: Optional[list[str]] = None,
) -> list[str]:
    required_targets = [
        str(target).strip()
        for target in list(required_contract_targets or [])
        if str(target).strip()
    ]
    if len(required_targets) <= 1:
        return required_targets

    searchable = f"{rel_path}\n{content}".lower()
    scored: list[tuple[int, str]] = []
    for target in required_targets:
        parts = [part for part in re.split(r"[.#:()]+", target) if part]
        full_target = target.lower()
        score = 0
        if full_target and full_target in searchable:
            score += 3
        if parts:
            leaf = parts[-1].lower()
            if leaf and re.search(rf"\b{re.escape(leaf)}\b", searchable):
                score += 2
        if len(parts) >= 2:
            container = parts[-2].lower()
            if container and re.search(rf"\b{re.escape(container)}\b", searchable):
                score += 1
        if score > 0:
            scored.append((score, target))
    if not scored:
        return required_targets

    best_score = max(score for score, _ in scored)
    return sorted({target for score, target in scored if score == best_score})


def _camelize_identifier(value: str) -> str:
    return "".join(
        part[:1].upper() + part[1:] for part in re.split(r"[_\-\s]+", str(value or "")) if part
    )


def _gold_target_search_terms(contract_targets: list[str]) -> tuple[list[str], list[str]]:
    strong_terms: list[str] = []
    fallback_terms: list[str] = []
    seen_strong: set[str] = set()
    seen_fallback: set[str] = set()

    def add_strong(value: str) -> None:
        token = str(value or "").strip().lower()
        if not token or token in seen_strong:
            return
        strong_terms.append(token)
        seen_strong.add(token)

    def add_fallback(value: str) -> None:
        token = str(value or "").strip().lower()
        if not token or token in seen_fallback or token in seen_strong:
            return
        fallback_terms.append(token)
        seen_fallback.add(token)

    for target in contract_targets:
        parts = [part for part in re.split(r"[.#:()/]+", str(target or "")) if part]
        if not parts:
            continue
        full_target = str(target or "").strip().lower()
        if full_target:
            add_strong(full_target)
        if len(parts) >= 2:
            add_strong(".".join(part.lower() for part in parts[-2:]))
        leaf = parts[-1].lower()
        camel_leaf = _camelize_identifier(leaf).lower()
        collapsed_leaf = re.sub(r"[_-]+", "", leaf)
        if "_" in leaf or "-" in leaf or len(leaf) >= 8:
            add_strong(leaf)
        if camel_leaf and camel_leaf != leaf:
            add_strong(camel_leaf)
            add_strong(f"test{camel_leaf}")
        if collapsed_leaf and collapsed_leaf not in {leaf, camel_leaf}:
            add_strong(collapsed_leaf)
            add_strong(f"test{collapsed_leaf}")
        add_fallback(leaf)
        if len(parts) >= 2:
            add_fallback(parts[-2].lower())
    return strong_terms, fallback_terms


def _matching_line_indices(lines: list[str], terms: list[str]) -> list[int]:
    if not terms:
        return []
    matches: list[int] = []
    for index, line in enumerate(lines):
        lowered = line.lower()
        if any(term in lowered for term in terms):
            matches.append(index)
    return matches


_GOLD_BLOCK_HEADER_PATTERN = re.compile(
    r"^\s*(?:async\s+def|def|class)\s+\w+|^\s*(?:it|test|describe)\s*\(",
    re.IGNORECASE,
)


def _gold_block_span(lines: list[str], index: int) -> tuple[int, int]:
    header_index: Optional[int] = None
    search_start = max(0, index - 16)
    for candidate in range(index, search_start - 1, -1):
        if _GOLD_BLOCK_HEADER_PATTERN.search(lines[candidate]):
            header_index = candidate
            break

    if header_index is None:
        start = max(0, index - 4)
        for candidate in range(index - 1, start - 1, -1):
            if not lines[candidate].strip():
                start = candidate + 1
                break
        return start, min(len(lines), index + 33)

    start = header_index
    while start > 0 and lines[start - 1].lstrip().startswith("@"):
        start -= 1

    header_indent = len(lines[header_index]) - len(lines[header_index].lstrip(" "))
    end = len(lines)
    for candidate in range(header_index + 1, len(lines)):
        stripped = lines[candidate].strip()
        if not stripped:
            continue
        indent = len(lines[candidate]) - len(lines[candidate].lstrip(" "))
        if indent <= header_indent and _GOLD_BLOCK_HEADER_PATTERN.search(lines[candidate]):
            end = candidate
            break
    return start, end


def _extract_target_scoped_gold_content(
    content: str,
    contract_targets: list[str],
) -> str:
    lines = str(content or "").splitlines()
    if not lines or not contract_targets:
        return str(content or "")

    strong_terms, fallback_terms = _gold_target_search_terms(contract_targets)
    match_indices = _matching_line_indices(lines, strong_terms)
    if not match_indices:
        match_indices = _matching_line_indices(lines, fallback_terms)
    if not match_indices:
        return str(content or "")

    spans: list[tuple[int, int]] = []
    for index in match_indices:
        start, end = _gold_block_span(lines, index)
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))

    scoped_lines: list[str] = []
    for span_index, (start, end) in enumerate(spans):
        if span_index:
            scoped_lines.append("")
        scoped_lines.extend(lines[start:end])
    scoped_content = "\n".join(scoped_lines).strip()
    if not scoped_content or len(scoped_content) >= int(0.9 * len(str(content or ""))):
        return str(content or "")
    return scoped_content


def build_gold_test_portfolio(
    packet: dict[str, Any],
    *,
    required_contract_targets: Optional[list[str]] = None,
) -> dict[str, Any]:
    gold_tests = dict(packet.get("gold_tests") or {})
    gold_files = dict(gold_tests.get("gold_test_files") or {})
    artifact_entries: list[dict[str, Any]] = []
    for rel_path, payload in sorted(gold_files.items()):
        content = str(dict(payload or {}).get("content") or "")
        if not content.strip():
            continue
        contract_targets = _infer_gold_file_contract_targets(
            rel_path,
            content,
            required_contract_targets=required_contract_targets,
        )
        artifact_entries.append(
            {
                "path": rel_path,
                "content": _extract_target_scoped_gold_content(content, contract_targets),
                "summary": f"Gold benchmark test file {rel_path}",
                "strategy": "regression",
                "contract_targets": contract_targets,
                "contract_sources": ["existing_tests"],
                "generator_vendor": "benchmark_gold",
                "generator_role": "gold_reference",
                "materialization_mode": "replace",
            }
        )
    normalized = normalize_test_suite_artifact_payload(
        {
            "summary": "Gold SWE-Bench Pro benchmark tests materialized offline.",
            "test_artifacts": artifact_entries,
        }
    )
    return apply_test_portfolio_promotion(
        normalized,
        required_contract_targets=list(required_contract_targets or []),
    )


def compare_generated_and_gold_portfolios(
    packet: dict[str, Any],
) -> dict[str, Any]:
    generated = normalize_test_suite_artifact_payload(packet.get("generated_portfolio") or {})
    required_targets = list(packet.get("required_contract_targets") or [])
    if not required_targets:
        required_targets = list(
            dict(generated.get("validation_summary") or {}).get("issue_contract_targets") or []
        )
    if not required_targets:
        required_targets = list(packet.get("generated_summary", {}).get("targets") or [])
    generated_scored = apply_test_portfolio_promotion(
        generated,
        required_contract_targets=required_targets,
    )
    gold_scored = build_gold_test_portfolio(
        packet,
        required_contract_targets=required_targets,
    )

    generated_validation = dict(generated_scored.get("validation_summary") or {})
    gold_validation = dict(gold_scored.get("validation_summary") or {})
    generated_contract = dict(generated_validation.get("contract_matrix") or {})
    gold_contract = dict(gold_validation.get("contract_matrix") or {})

    generated_primary_targets = list(generated_contract.get("primary_targets") or [])
    gold_primary_targets = list(gold_contract.get("primary_targets") or [])
    generated_required_axes = list(generated_contract.get("required_axes") or [])
    gold_required_axes = list(gold_contract.get("required_axes") or [])
    generated_axes_by_target = {
        str(target): set(values or [])
        for target, values in dict(generated_contract.get("covered_axes_by_target") or {}).items()
    }
    gold_axes_by_target = {
        str(target): set(values or [])
        for target, values in dict(gold_contract.get("covered_axes_by_target") or {}).items()
    }

    target_union = sorted(
        {
            str(target)
            for target in list(generated_primary_targets) + list(gold_primary_targets)
            if str(target).strip()
        }
    )
    target_comparison: list[dict[str, Any]] = []
    shared_axis_hits = 0
    gold_axis_total = 0
    for target in target_union:
        generated_axes = generated_axes_by_target.get(target, set())
        gold_axes = gold_axes_by_target.get(target, set())
        gold_axis_total += len(gold_axes)
        shared_axes = sorted(generated_axes.intersection(gold_axes))
        shared_axis_hits += len(shared_axes)
        target_comparison.append(
            {
                "target": target,
                "gold_axes": sorted(gold_axes),
                "generated_axes": sorted(generated_axes),
                "missing_axes": sorted(gold_axes.difference(generated_axes)),
                "extra_axes": sorted(generated_axes.difference(gold_axes)),
                "shared_axes": shared_axes,
                "gold_coverage_ratio": round(
                    (len(shared_axes) / len(gold_axes)) if gold_axes else 1.0,
                    4,
                ),
            }
        )

    generated_field_paths = extract_data_contract_field_paths(
        [
            str(generated_scored.get("summary") or ""),
            str(generated_scored.get("portfolio_summary") or ""),
            str(generated_scored.get("promotion_summary") or ""),
            *[str(item) for item in list(generated_scored.get("contract_hypotheses") or [])],
            *[str(item) for item in list(generated_scored.get("test_descriptions") or [])],
            *[
                str(item)
                for artifact in list(generated_scored.get("test_artifacts") or [])
                for item in (
                    str(artifact.get("summary") or ""),
                    str(artifact.get("justification") or ""),
                    str(artifact.get("content") or ""),
                    *[str(value) for value in list(artifact.get("test_descriptions") or [])],
                    *[str(value) for value in list(artifact.get("properties") or [])],
                )
            ],
        ]
    )
    gold_field_paths = extract_data_contract_field_paths(
        [
            str(gold_scored.get("summary") or ""),
            str(gold_scored.get("portfolio_summary") or ""),
            str(gold_scored.get("promotion_summary") or ""),
            *[str(item) for item in list(gold_scored.get("contract_hypotheses") or [])],
            *[str(item) for item in list(gold_scored.get("test_descriptions") or [])],
            *[
                str(item)
                for artifact in list(gold_scored.get("test_artifacts") or [])
                for item in (
                    str(artifact.get("summary") or ""),
                    str(artifact.get("justification") or ""),
                    str(artifact.get("content") or ""),
                    *[str(value) for value in list(artifact.get("test_descriptions") or [])],
                    *[str(value) for value in list(artifact.get("properties") or [])],
                )
            ],
        ]
    )
    issue_field_paths = extract_data_contract_field_paths(
        [str(packet.get("issue_description") or "")]
    )
    if not issue_field_paths:
        issue_field_paths = extract_data_contract_field_paths(
            list(packet.get("issue_field_paths") or [])
        )
    authoritative_field_paths = list(issue_field_paths or gold_field_paths or generated_field_paths)
    generated_field_shape = evaluate_field_path_negative_shape_coverage(
        {
            "summary": generated_scored.get("summary"),
            "justification": generated_scored.get("promotion_summary"),
            "content": "\n\n".join(
                str(item.get("content") or "")
                for item in list(generated_scored.get("test_artifacts") or [])
            ),
            "test_descriptions": list(generated_scored.get("test_descriptions") or []),
        },
        authoritative_field_paths,
    )
    gold_field_shape = evaluate_field_path_negative_shape_coverage(
        {
            "summary": gold_scored.get("summary"),
            "justification": gold_scored.get("promotion_summary"),
            "content": "\n\n".join(
                str(item.get("content") or "")
                for item in list(gold_scored.get("test_artifacts") or [])
            ),
            "test_descriptions": list(gold_scored.get("test_descriptions") or []),
        },
        authoritative_field_paths,
    )

    generated_paths = {
        str(item.get("path") or "")
        for item in list(generated_scored.get("test_artifacts") or [])
        if str(item.get("path") or "").strip()
    }
    gold_paths = {
        str(path)
        for path in list(dict(packet.get("gold_tests") or {}).get("benchmark_test_files") or [])
        if str(path).strip()
    }

    generated_targets = {
        str(target)
        for target in list(generated_primary_targets)
        + list(packet.get("generated_summary", {}).get("targets") or [])
        if str(target).strip()
    }
    gold_targets = {str(target) for target in list(gold_primary_targets) if str(target).strip()}
    shared_targets = sorted(generated_targets.intersection(gold_targets))
    shared_paths = sorted(generated_paths.intersection(gold_paths))
    gold_field_path_set = set(gold_field_paths)
    generated_field_path_set = set(generated_field_paths)
    shared_field_paths = sorted(generated_field_path_set.intersection(gold_field_path_set))
    generated_field_path_shapes = list(generated_field_shape.get("covered_shapes") or [])
    gold_field_path_shapes = list(gold_field_shape.get("covered_shapes") or [])
    shared_gold_field_path_shapes = sorted(
        set(generated_field_path_shapes).intersection(set(gold_field_path_shapes))
    )
    gold_field_path_recall = round(
        (len(shared_field_paths) / len(gold_field_path_set)) if gold_field_path_set else 1.0,
        4,
    )
    # When the gold suite has no extracted field paths but the issue itself
    # documents some (the common case for tasks where the gold tests are
    # behavior-only without payload assertions), compute recall against the
    # *authoritative* set instead. Without this, the aggregator filter on
    # gold_field_path_count > 0 silently drops every such task and
    # mean_gold_field_path_recall becomes 0.0 even though per-task recall
    # was 1.0 by vacuous truth.
    authoritative_field_path_set = set(authoritative_field_paths)
    shared_authoritative_field_paths = sorted(
        generated_field_path_set.intersection(authoritative_field_path_set)
    )
    authoritative_field_path_recall = round(
        (len(shared_authoritative_field_paths) / len(authoritative_field_path_set))
        if authoritative_field_path_set
        else 1.0,
        4,
    )
    gold_field_path_shape_recall = round(
        (len(shared_gold_field_path_shapes) / len(gold_field_path_shapes))
        if gold_field_path_shapes
        else 1.0,
        4,
    )
    gold_target_recall = round(
        (len(shared_targets) / len(gold_targets)) if gold_targets else 1.0,
        4,
    )

    overall_contract_axis_recall = round(
        (shared_axis_hits / gold_axis_total) if gold_axis_total else 1.0,
        4,
    )

    # Per-task required-axis coverage: which of the canonical four contract
    # axes did the generated portfolio touch *anywhere* (across all targets)?
    # The harness asks the test_writer to fill all four slots — this score
    # surfaces whether the agent actually did, independent of which target
    # the axis attached to. Anchored on gold_required_axes (the ground
    # truth) when present, falling back to the canonical four otherwise.
    _CANONICAL_REQUIRED_AXES = {
        "positive_path",
        "missing_boundary",
        "negative_malformed",
        "multi_ordering",
    }
    required_axis_set = {
        str(axis).strip()
        for axis in (gold_required_axes or list(_CANONICAL_REQUIRED_AXES))
        if str(axis).strip()
    } or set(_CANONICAL_REQUIRED_AXES)
    generated_axis_universe: set[str] = set()
    for axes in generated_axes_by_target.values():
        generated_axis_universe.update(str(a).strip() for a in axes if str(a).strip())
    covered_required_axes = sorted(required_axis_set.intersection(generated_axis_universe))
    missing_required_axes = sorted(required_axis_set.difference(generated_axis_universe))
    required_axis_coverage_score = round(
        (len(covered_required_axes) / len(required_axis_set)) if required_axis_set else 1.0,
        4,
    )
    return {
        "instance_id": str(packet.get("instance_id") or ""),
        "repo": str(packet.get("repo") or ""),
        "generated_portfolio": generated_scored,
        "gold_portfolio": gold_scored,
        "generated_summary": dict(packet.get("generated_summary") or {}),
        "gold_summary": {
            "artifact_count": len(list(gold_scored.get("test_artifacts") or [])),
            "paths": sorted(gold_paths),
            "targets": sorted(gold_targets),
        },
        "coverage_summary": {
            "generated_artifact_count": len(list(generated_scored.get("test_artifacts") or [])),
            "gold_artifact_count": len(list(gold_scored.get("test_artifacts") or [])),
            "generated_primary_target_count": len(generated_primary_targets),
            "gold_primary_target_count": len(gold_primary_targets),
            "shared_gold_target_count": len(shared_targets),
            "gold_target_recall": gold_target_recall,
            "generated_required_axes": generated_required_axes,
            "gold_required_axes": gold_required_axes,
            "overall_contract_axis_recall": overall_contract_axis_recall,
            "required_axis_coverage_score": required_axis_coverage_score,
            "covered_required_axes": covered_required_axes,
            "missing_required_axes": missing_required_axes,
            "generated_contract_target_coverage_ratio": float(
                generated_validation.get("contract_matrix_target_coverage_ratio") or 0.0
            ),
            "gold_contract_target_coverage_ratio": float(
                gold_validation.get("contract_matrix_target_coverage_ratio") or 0.0
            ),
            "generated_field_path_negative_shape_coverage_ratio": float(
                generated_field_shape.get("coverage_ratio") or 0.0
            ),
            "gold_field_path_negative_shape_coverage_ratio": float(
                gold_field_shape.get("coverage_ratio") or 0.0
            ),
            "shared_gold_path_count": len(shared_paths),
            "gold_path_count": len(gold_paths),
            "missing_gold_targets": sorted(gold_targets.difference(generated_targets)),
            "extra_generated_targets": sorted(generated_targets.difference(gold_targets)),
            "missing_gold_paths": sorted(gold_paths.difference(generated_paths)),
            "extra_generated_paths": sorted(generated_paths.difference(gold_paths)),
            "gold_field_path_count": len(gold_field_path_set),
            "authoritative_field_path_count": len(authoritative_field_paths),
            "authoritative_field_paths": list(authoritative_field_paths),
            "shared_gold_field_path_count": len(shared_field_paths),
            "shared_authoritative_field_path_count": len(shared_authoritative_field_paths),
            "gold_field_path_recall": gold_field_path_recall,
            "authoritative_field_path_recall": authoritative_field_path_recall,
            "missing_gold_field_paths": sorted(
                gold_field_path_set.difference(generated_field_path_set)
            ),
            "missing_authoritative_field_paths": sorted(
                authoritative_field_path_set.difference(generated_field_path_set)
            ),
            "extra_generated_field_paths": sorted(
                generated_field_path_set.difference(gold_field_path_set)
            ),
            "gold_field_path_shape_count": len(gold_field_path_shapes),
            "shared_gold_field_path_shape_count": len(shared_gold_field_path_shapes),
            "gold_field_path_shape_recall": gold_field_path_shape_recall,
            "missing_gold_field_path_shapes": sorted(
                set(gold_field_path_shapes).difference(set(generated_field_path_shapes))
            ),
            "extra_generated_field_path_shapes": sorted(
                set(generated_field_path_shapes).difference(set(gold_field_path_shapes))
            ),
            "gold_field_path_shapes": gold_field_path_shapes,
            "generated_field_path_shapes": generated_field_path_shapes,
        },
        "target_comparison": target_comparison,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an offline SWE-Bench Pro synthesized-vs-gold test comparison packet.",
    )
    parser.add_argument("--apex-result", required=True, help="Path to apex_result.json")
    parser.add_argument("--instance-id", required=True, help="SWE-Bench Pro instance id")
    parser.add_argument(
        "--worktree-path", help="Prepared benchmark worktree to apply the gold test patch onto"
    )
    parser.add_argument(
        "--rollout-id",
        type=int,
        help="Optional rollout id to inspect instead of the selected rollout",
    )
    parser.add_argument(
        "--output-json", help="Optional path to write the comparison packet as JSON"
    )
    parser.add_argument("--dataset-name", default=SWEBENCH_PRO_DATASET_NAME)
    parser.add_argument("--dataset-split", default=SWEBENCH_PRO_DATASET_SPLIT)
    args = parser.parse_args(argv)

    packet = build_swebench_gold_comparison_packet(
        apex_result_path=args.apex_result,
        instance_id=args.instance_id,
        rollout_id=args.rollout_id,
        worktree_path=args.worktree_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
    )
    rendered = json.dumps(packet, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(rendered)
    else:
        print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
