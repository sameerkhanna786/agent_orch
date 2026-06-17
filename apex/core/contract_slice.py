"""Compact contract-slice construction for model prompts."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from ..agents.artifacts import coerce_localization_artifact
from .stub_scanner import scan_files_for_stubs


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _issue_list(payload: Any, attr: str) -> list[str]:
    if isinstance(payload, dict):
        values = payload.get(attr)
        if isinstance(values, (list, tuple, set)):
            return [str(value).strip() for value in values if str(value).strip()]
        return []
    values = getattr(payload, attr, None)
    if isinstance(values, (list, tuple, set)):
        return [str(value).strip() for value in values if str(value).strip()]
    return []


def _test_context(issue_plan: Any) -> Any:
    return getattr(issue_plan, "test_context", None)


def _path_from_test_id(test_id: str) -> str:
    return str(test_id or "").split("::", 1)[0].strip()


def _python_symbols(path: Path) -> dict[str, Any]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return {}
    exports: list[str] = []
    symbols: list[str] = []
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(node.name)
            if not node.name.startswith("_"):
                exports.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == "__all__":
                        try:
                            value = ast.literal_eval(node.value)
                        except Exception:
                            value = None
                        if isinstance(value, (list, tuple)):
                            exports.extend(str(item) for item in value if isinstance(item, str))
                    elif not target.id.startswith("_"):
                        exports.append(target.id)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = "." * int(node.level or 0) + str(node.module or "")
            imports.append(module)
    return {
        "symbols": _dedupe(symbols)[:12],
        "public_exports": _dedupe(exports)[:12],
        "imports": _dedupe(imports)[:12],
    }


def build_contract_slice(
    *,
    repo_root: str | Path,
    issue_plan: Any,
    localization_artifact: Any = None,
    quick_verification: dict[str, Any] | None = None,
    max_files: int = 8,
) -> dict[str, Any]:
    root = Path(repo_root)
    localization = coerce_localization_artifact(localization_artifact)
    test_context = _test_context(issue_plan)
    failing_tests = _dedupe(
        _issue_list(test_context, "failing_test_ids")
        + _issue_list(test_context, "expected_test_ids")
        + _issue_list(quick_verification or {}, "failed_tests")
    )[:12]
    file_candidates = _dedupe(
        (list(localization.files) if localization is not None else [])
        + _issue_list(issue_plan, "relevant_files")
        + _issue_list(issue_plan, "risk_files")
        + [_path_from_test_id(test_id) for test_id in failing_tests]
    )
    selected_files = [
        path
        for path in file_candidates
        if path and not Path(path).is_absolute() and (root / path).exists()
    ][:max_files]
    symbol_map: dict[str, Any] = {}
    for rel_path in selected_files:
        path = root / rel_path
        if path.suffix == ".py":
            payload = _python_symbols(path)
            if payload:
                symbol_map[rel_path] = payload
    stubs: list[dict[str, Any]] = []
    try:
        findings = scan_files_for_stubs(root, selected_files)
    except Exception:
        findings = []
    for finding in list(findings or [])[:12]:
        if hasattr(finding, "__dict__"):
            stubs.append(
                {
                    "path": str(getattr(finding, "path", "")),
                    "symbol": str(getattr(finding, "symbol", "")),
                    "reason": str(getattr(finding, "reason", "")),
                }
            )
        elif isinstance(finding, dict):
            stubs.append(
                {
                    "path": str(finding.get("path") or ""),
                    "symbol": str(finding.get("symbol") or ""),
                    "reason": str(finding.get("reason") or ""),
                }
            )
    return {
        "failing_tests": failing_tests,
        "files": selected_files,
        "symbols": list(localization.symbols)[:12] if localization is not None else [],
        "file_contracts": symbol_map,
        "stub_findings": stubs,
    }


def render_contract_slice(slice_payload: dict[str, Any]) -> str:
    if not any(slice_payload.get(key) for key in ("failing_tests", "files", "file_contracts", "stub_findings")):
        return ""
    lines: list[str] = ["# Contract Slice"]
    failing_tests = list(slice_payload.get("failing_tests") or [])
    if failing_tests:
        lines.append("Failing/expected tests:")
        lines.extend(f"- {item}" for item in failing_tests[:10])
    files = list(slice_payload.get("files") or [])
    if files:
        lines.append("Relevant contract files:")
        lines.extend(f"- {item}" for item in files[:8])
    contracts = slice_payload.get("file_contracts") if isinstance(slice_payload.get("file_contracts"), dict) else {}
    for path, payload in list(contracts.items())[:6]:
        if not isinstance(payload, dict):
            continue
        exports = ", ".join(list(payload.get("public_exports") or [])[:8])
        symbols = ", ".join(list(payload.get("symbols") or [])[:8])
        imports = ", ".join(list(payload.get("imports") or [])[:6])
        details = "; ".join(part for part in (f"exports: {exports}" if exports else "", f"symbols: {symbols}" if symbols else "", f"imports: {imports}" if imports else "") if part)
        if details:
            lines.append(f"- {path}: {details}")
    stubs = list(slice_payload.get("stub_findings") or [])
    if stubs:
        lines.append("Placeholder/stub findings:")
        for item in stubs[:8]:
            if isinstance(item, dict):
                lines.append(
                    f"- {item.get('path')}: {item.get('symbol')} ({item.get('reason')})"
                )
    return "\n".join(lines).strip()
