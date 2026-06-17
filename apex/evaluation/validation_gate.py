"""Closed-loop validation primitives for generated test artifacts."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from apex.core.generated_tests import normalize_generated_test_path

from .api_probe import ApiProbeResult, find_missing_public_imports
from .code_emission import roundtrip
from .failure_classifier import FailureClassification, classify_testgen_failure
from .signature_preflight import preflight_signatures, signatures_from_api_probe
from .splice_simulator import SpliceMode, SpliceSimulator, validate_splice_invariants
from .test_style import (
    TestStyleProfile,
    imports_forbidden_by_style,
)

_TARGET_ENVIRONMENT_SETUP_FAILURES = frozenset(
    {
        "artifact_failed",
        "collection_failed",
        "harness_error",
        "harness_log_missing",
        "setup_failed",
        "syntax_error",
    }
)


def _active_target_environment_adapter() -> Any | None:
    try:
        from .docker_acceptance_adapter import get_docker_task_context
    except Exception:  # pragma: no cover - defensive
        return None
    ctx = get_docker_task_context()
    return getattr(ctx, "adapter", None) if ctx is not None else None


def _run_target_environment_validation(
    *,
    adapter: Any,
    artifacts: list[dict[str, Any]],
    workdir: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    from .final_acceptance_gate import GeneratedArtifact

    adapter_name = str(getattr(adapter, "name", "") or "benchmark_adapter")
    runs: list[dict[str, Any]] = []
    setup_failure = False
    test_failure = False
    diagnostics: list[str] = []
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            continue
        content = str(artifact.get("content") or "")
        if not content.strip():
            continue
        path = normalize_generated_test_path(artifact.get("path")) or str(
            artifact.get("path") or f"tests/test_apex_generated_{index}.py"
        )
        item = GeneratedArtifact(path=path, content=content)
        try:
            try:
                raw_run = adapter.run_unfiltered(
                    item,
                    workdir,
                    timeout_seconds=float(timeout_seconds),
                )
            except TypeError:
                raw_run = adapter.run_unfiltered(item, workdir)
            run_payload = raw_run.to_dict() if hasattr(raw_run, "to_dict") else dict(raw_run or {})
        except Exception as exc:  # pragma: no cover - adapter boundary
            run_payload = {
                "status": "harness_error",
                "failure_taxonomy": "harness_error",
                "diagnostic": f"{type(exc).__name__}: {exc}",
            }
        run_payload["artifact_path"] = path
        runs.append(run_payload)
        status = str(run_payload.get("status") or "").strip().lower()
        taxonomy = str(run_payload.get("failure_taxonomy") or status).strip().lower()
        if (
            status in _TARGET_ENVIRONMENT_SETUP_FAILURES
            or taxonomy in _TARGET_ENVIRONMENT_SETUP_FAILURES
        ):
            setup_failure = True
        elif status and status not in {"pass", "passed", "ok", "ship", "shipped"}:
            test_failure = True
        diagnostic = str(run_payload.get("diagnostic") or "")
        if diagnostic:
            diagnostics.append(diagnostic)
    return {
        "status": "fail" if setup_failure or test_failure else "pass",
        "adapter": adapter_name,
        "target_environment_adapter": True,
        "target_environment_collection_failed": setup_failure,
        "target_environment_test_failed": test_failure,
        "target_environment_runs": runs,
        "diagnostic": "\n".join(diagnostics)[-4000:],
    }


@dataclass(frozen=True)
class ValidationTierResult:
    name: str
    status: str
    diagnostic: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationGateResult:
    status: str
    artifacts: list[dict[str, Any]]
    tier_1_static: ValidationTierResult
    tier_2_import: ValidationTierResult | None = None
    tier_2_collect: ValidationTierResult | None = None
    tier_3_run: dict[str, Any] | None = None
    failure_classification: FailureClassification | None = None
    repair_attempts: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "artifact_count": len(self.artifacts),
            "tier_1_static": self.tier_1_static.to_dict(),
            "tier_2_import": self.tier_2_import.to_dict() if self.tier_2_import else None,
            "tier_2_collect": (self.tier_2_collect.to_dict() if self.tier_2_collect else None),
            "tier_3_run": self.tier_3_run,
            "failure_classification": (
                self.failure_classification.to_dict() if self.failure_classification else None
            ),
            "repair_attempts": self.repair_attempts,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class ValidationPipelineResult:
    artifacts: list[dict[str, Any]]
    validation: ValidationGateResult
    generation_diagnostics: dict[str, Any] = field(default_factory=dict)
    repair_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    best_valid_artifacts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_count": len(self.artifacts),
            "validation": self.validation.to_dict(),
            "generation_diagnostics": dict(self.generation_diagnostics),
            "repair_diagnostics": list(self.repair_diagnostics),
            "best_valid_artifact_count": len(self.best_valid_artifacts),
        }


def validate_static_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    style: TestStyleProfile,
    api_probe: ApiProbeResult | None = None,
    focal_module: str = "",
    original_test_source: str = "",
    splice_simulator: SpliceSimulator | None = None,
) -> ValidationGateResult:
    started = time.time()
    if not artifacts:
        tier = ValidationTierResult(
            name="static",
            status="fail",
            diagnostic="no artifacts generated",
            duration_seconds=time.time() - started,
        )
        return ValidationGateResult(
            status="fail",
            artifacts=[],
            tier_1_static=tier,
            failure_classification=classify_testgen_failure(
                {"diagnostic": tier.diagnostic},
                style=style,
            ),
        )

    diagnostics: dict[str, Any] = {"artifact_static": []}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        path = normalize_generated_test_path(artifact.get("path"))
        source = str(artifact.get("content") or "")
        artifact_diag = {"path": path, "language": style.language}
        source_for_validation = source
        if splice_simulator is not None and original_test_source:
            source_for_validation = splice_simulator.synthesize_post_splice(
                original_test_source=original_test_source,
                artifact_text=source,
                style=style,
            )
            invariant = validate_splice_invariants(
                original_test_source=original_test_source,
                artifact_text=source,
                post_splice_source=source_for_validation,
                style=style,
                splice_mode=getattr(splice_simulator, "splice_mode", SpliceMode.APPEND),
            )
            artifact_diag["post_splice_validation"] = invariant.to_dict()
            if not invariant.passed:
                tier = ValidationTierResult(
                    name="static",
                    status="fail",
                    diagnostic=invariant.diagnostic,
                    duration_seconds=time.time() - started,
                )
                return ValidationGateResult(
                    status="fail",
                    artifacts=artifacts,
                    tier_1_static=tier,
                    failure_classification=classify_testgen_failure(
                        {"diagnostic": tier.diagnostic},
                        style=style,
                    ),
                    diagnostics=diagnostics,
                )
        if (style.language or "").lower() in {"python", "py", "python3"}:
            try:
                ast.parse(source_for_validation)
                compile(source_for_validation, path or "<generated-test>", "exec")
                roundtrip(source_for_validation, language=style.language)
            except SyntaxError as exc:
                tier = ValidationTierResult(
                    name="static",
                    status="fail",
                    diagnostic=f"SyntaxError in {path or '<generated-test>'}: {exc}",
                    duration_seconds=time.time() - started,
                )
                return ValidationGateResult(
                    status="fail",
                    artifacts=artifacts,
                    tier_1_static=tier,
                    failure_classification=classify_testgen_failure(
                        {"diagnostic": tier.diagnostic},
                        style=style,
                    ),
                    diagnostics=diagnostics,
                )
            if api_probe:
                signature_result = preflight_signatures(
                    source,
                    signatures_from_api_probe(api_probe),
                )
                artifact_diag["signature_preflight"] = signature_result.to_dict()
                if not signature_result.passed:
                    tier = ValidationTierResult(
                        name="static",
                        status="fail",
                        diagnostic="Signature preflight failed: "
                        + "; ".join(signature_result.diagnostics[:5]),
                        duration_seconds=time.time() - started,
                    )
                    return ValidationGateResult(
                        status="fail",
                        artifacts=artifacts,
                        tier_1_static=tier,
                        failure_classification=classify_testgen_failure(
                            {"diagnostic": tier.diagnostic},
                            style=style,
                        ),
                        diagnostics=diagnostics,
                    )
        forbidden = imports_forbidden_by_style(source, style)
        if forbidden:
            tier = ValidationTierResult(
                name="static",
                status="fail",
                diagnostic=(f"Forbidden import(s) for {style.runner}: " + ", ".join(forbidden)),
                duration_seconds=time.time() - started,
            )
            return ValidationGateResult(
                status="fail",
                artifacts=artifacts,
                tier_1_static=tier,
                failure_classification=classify_testgen_failure(
                    {"diagnostic": tier.diagnostic},
                    style=style,
                ),
                diagnostics=diagnostics,
            )
        if api_probe and focal_module and (style.language or "").lower().startswith("python"):
            missing = find_missing_public_imports(
                test_source=source,
                focal_module=focal_module,
                public_names=api_probe.public_names,
            )
            if missing:
                rendered = ", ".join(
                    f"{name} (closest: {closest or 'none'})"
                    for name, closest in sorted(missing.items())
                )
                tier = ValidationTierResult(
                    name="static",
                    status="fail",
                    diagnostic=f"Missing focal symbol import(s): {rendered}",
                    duration_seconds=time.time() - started,
                )
                return ValidationGateResult(
                    status="fail",
                    artifacts=artifacts,
                    tier_1_static=tier,
                    failure_classification=classify_testgen_failure(
                        {"diagnostic": tier.diagnostic},
                        style=style,
                    ),
                    diagnostics=diagnostics,
                )
        artifact_diag["forbidden_imports"] = forbidden
        diagnostics["artifact_static"].append(artifact_diag)

    tier = ValidationTierResult(
        name="static",
        status="pass",
        duration_seconds=time.time() - started,
    )
    return ValidationGateResult(
        status="pass",
        artifacts=artifacts,
        tier_1_static=tier,
        diagnostics=diagnostics,
    )


def import_validate_python_artifacts(
    *,
    workdir: Path,
    artifacts: list[dict[str, Any]],
    timeout_seconds: float = 10.0,
    python_executable: str | None = None,
) -> ValidationTierResult:
    started = time.time()
    executable = python_executable or sys.executable
    for artifact in artifacts:
        path = normalize_generated_test_path(artifact.get("path"))
        if not path:
            continue
        target = workdir / path
        if not target.exists():
            continue
        code = (
            "import importlib.util, pathlib; "
            f"p=pathlib.Path({str(target)!r}); "
            "spec=importlib.util.spec_from_file_location('apex_generated_test', p); "
            "m=importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(m)"
        )
        try:
            completed = subprocess.run(
                [executable, "-c", code],
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ValidationTierResult(
                name="import",
                status="fail",
                diagnostic=f"import validation timed out for {path}",
                duration_seconds=time.time() - started,
            )
        except OSError as exc:
            return ValidationTierResult(
                name="import",
                status="fail",
                diagnostic=f"import validation could not start for {path}: {type(exc).__name__}: {exc}",
                duration_seconds=time.time() - started,
            )
        if completed.returncode != 0:
            return ValidationTierResult(
                name="import",
                status="fail",
                diagnostic=(completed.stderr or completed.stdout or "")[-4000:],
                duration_seconds=time.time() - started,
            )
    return ValidationTierResult(
        name="import",
        status="pass",
        duration_seconds=time.time() - started,
    )


def collect_validate_artifacts(
    *,
    workdir: Path,
    artifacts: list[dict[str, Any]],
    style: TestStyleProfile,
    timeout_seconds: float = 20.0,
    python_executable: str | None = None,
) -> ValidationTierResult:
    """Run the lightest framework-native collect/list step available."""

    started = time.time()
    language = (style.language or "").lower()
    if language in {"python", "py", "python3"}:
        return _collect_validate_python_artifacts(
            workdir=workdir,
            artifacts=artifacts,
            style=style,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
            started=started,
        )
    if language in {"javascript", "typescript"}:
        return _collect_validate_js_artifacts(
            workdir=workdir,
            artifacts=artifacts,
            style=style,
            timeout_seconds=timeout_seconds,
            started=started,
        )
    return ValidationTierResult(
        name="collect",
        status="skipped",
        diagnostic=f"collect validation unavailable for {style.language}",
        duration_seconds=time.time() - started,
    )


def validate_testgen_portfolio_static(
    portfolio: dict[str, Any],
    *,
    style: TestStyleProfile,
    api_probe: ApiProbeResult | None = None,
    focal_module: str = "",
    original_test_source: str = "",
    splice_simulator: SpliceSimulator | None = None,
) -> dict[str, Any]:
    """Benchmark-adapter friendly wrapper for portfolio validation telemetry."""

    artifacts = [
        artifact
        for artifact in list((portfolio or {}).get("test_artifacts") or [])
        if isinstance(artifact, dict)
    ]
    validation = validate_static_artifacts(
        artifacts,
        style=style,
        api_probe=api_probe,
        focal_module=focal_module,
        original_test_source=original_test_source,
        splice_simulator=splice_simulator,
    )
    return {
        "status": validation.status,
        "tier_1_static": validation.tier_1_static.to_dict(),
        "failure_classification": (
            validation.failure_classification.to_dict()
            if validation.failure_classification
            else None
        ),
        "style_profile": style.to_dict(),
        "artifact_count": len(artifacts),
    }


def _collect_validate_python_artifacts(
    *,
    workdir: Path,
    artifacts: list[dict[str, Any]],
    style: TestStyleProfile,
    timeout_seconds: float,
    python_executable: str | None,
    started: float,
) -> ValidationTierResult:
    executable = python_executable or sys.executable
    paths = _materialized_artifact_paths(workdir, artifacts)
    if not paths:
        return ValidationTierResult(
            name="collect",
            status="skipped",
            diagnostic="no materialized Python test artifacts",
            duration_seconds=time.time() - started,
        )
    runner = (style.runner or "").lower()
    if runner in {"pytest", "sympy-bin-test"}:
        command = [
            executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *paths,
        ]
    elif runner in {"unittest", "django-runtests"}:
        command = [
            executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(_common_test_root(paths)),
            "-p",
            "test*.py",
        ]
    else:
        return ValidationTierResult(
            name="collect",
            status="skipped",
            diagnostic=f"collect validation unavailable for runner {style.runner}",
            duration_seconds=time.time() - started,
        )
    return _run_collect_command(
        command,
        cwd=workdir,
        timeout_seconds=timeout_seconds,
        started=started,
    )


def _collect_validate_js_artifacts(
    *,
    workdir: Path,
    artifacts: list[dict[str, Any]],
    style: TestStyleProfile,
    timeout_seconds: float,
    started: float,
) -> ValidationTierResult:
    paths = _materialized_artifact_paths(workdir, artifacts)
    if not paths:
        return ValidationTierResult(
            name="collect",
            status="skipped",
            diagnostic="no materialized JS/TS test artifacts",
            duration_seconds=time.time() - started,
        )
    if not _js_runner_config_present(workdir):
        return ValidationTierResult(
            name="collect",
            status="skipped",
            diagnostic="no JS/TS runner config found",
            duration_seconds=time.time() - started,
        )
    runner = (style.runner or "").lower()
    if runner == "vitest":
        command = ["npx", "vitest", "list", *paths]
    elif runner in {"jest", "js-test"}:
        command = ["npx", "jest", "--listTests", *paths]
    else:
        return ValidationTierResult(
            name="collect",
            status="skipped",
            diagnostic=f"collect validation unavailable for runner {style.runner}",
            duration_seconds=time.time() - started,
        )
    return _run_collect_command(
        command,
        cwd=workdir,
        timeout_seconds=timeout_seconds,
        started=started,
    )


def _js_runner_config_present(workdir: Path) -> bool:
    for name in (
        "package.json",
        "jest.config.js",
        "jest.config.ts",
        "vitest.config.js",
        "vitest.config.ts",
        "mocha.opts",
    ):
        if (workdir / name).exists():
            return True
    return False


def _run_collect_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    started: float,
) -> ValidationTierResult:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ValidationTierResult(
            name="collect",
            status="fail",
            diagnostic="collect validation timed out",
            duration_seconds=time.time() - started,
        )
    except OSError as exc:
        return ValidationTierResult(
            name="collect",
            status="skipped",
            diagnostic=f"collect validation could not start: {type(exc).__name__}: {exc}",
            duration_seconds=time.time() - started,
        )
    return ValidationTierResult(
        name="collect",
        status="pass" if completed.returncode == 0 else "fail",
        diagnostic=(
            ""
            if completed.returncode == 0
            else (completed.stderr or completed.stdout or "")[-4000:]
        ),
        duration_seconds=time.time() - started,
    )


def _materialized_artifact_paths(
    workdir: Path,
    artifacts: list[dict[str, Any]],
) -> list[str]:
    paths: list[str] = []
    for artifact in artifacts:
        path = normalize_generated_test_path(artifact.get("path"))
        if not path:
            continue
        if (workdir / path).exists():
            paths.append(path)
    return paths


def _common_test_root(paths: list[str]) -> Path:
    parents = [Path(path).parent for path in paths if path]
    if not parents:
        return Path(".")
    try:
        common = os.path.commonpath([str(parent) for parent in parents])
    except ValueError:
        return Path(".")
    return Path(common or ".")


def run_testgen_with_validation_and_repair(
    *,
    generate: Callable[[], tuple[list[dict[str, Any]], dict[str, Any]]],
    repair: Callable[
        [list[dict[str, Any]], dict[str, Any], int], tuple[list[dict[str, Any]], dict[str, Any]]
    ],
    style: TestStyleProfile,
    max_repair_attempts: int = 3,
    api_probe: ApiProbeResult | None = None,
    focal_module: str = "",
    workdir: Path | None = None,
    python_executable: str | None = None,
    run_import_validation: bool = False,
    run_collect_validation: bool = False,
    validation_timeout_seconds: float = 20.0,
    original_test_source: str = "",
    splice_simulator: SpliceSimulator | None = None,
    run_tier_3: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
    minimize_artifacts: Callable[
        [list[dict[str, Any]], dict[str, Any]],
        tuple[list[dict[str, Any]], dict[str, Any]],
    ]
    | None = None,
) -> ValidationPipelineResult:
    artifacts, generation_diagnostics = generate()
    static_result = _validate_artifacts_with_optional_tiers(
        artifacts,
        style=style,
        api_probe=api_probe,
        focal_module=focal_module,
        workdir=workdir,
        python_executable=python_executable,
        run_import_validation=run_import_validation,
        run_collect_validation=run_collect_validation,
        validation_timeout_seconds=validation_timeout_seconds,
        original_test_source=original_test_source,
        splice_simulator=splice_simulator,
    )
    best_valid = list(artifacts) if static_result.passed else []
    repair_diagnostics: list[dict[str, Any]] = []
    attempts = 0
    last_result = static_result
    last_error = static_result.tier_1_static.diagnostic
    repeated_static_failures = 0

    best_artifacts = (
        list(best_valid)
        if best_valid
        else (list(artifacts) if _artifacts_are_syntactically_valid(artifacts, style=style) else [])
    )
    best_score = _validation_attempt_score(last_result)

    while not last_result.passed and attempts < max(0, int(max_repair_attempts or 0)):
        attempts += 1
        repaired, repair_diag = repair(
            artifacts,
            _validation_failure_payload(last_result),
            attempts,
        )
        repair_diag = dict(repair_diag)
        repair_diag["validation_attempt"] = attempts
        repair_diagnostics.append(repair_diag)
        if not repaired:
            break
        artifacts = repaired
        last_result = _validate_artifacts_with_optional_tiers(
            artifacts,
            style=style,
            api_probe=api_probe,
            focal_module=focal_module,
            workdir=workdir,
            python_executable=python_executable,
            run_import_validation=run_import_validation,
            run_collect_validation=run_collect_validation,
            validation_timeout_seconds=validation_timeout_seconds,
            original_test_source=original_test_source,
            splice_simulator=splice_simulator,
        )
        attempt_score = _validation_attempt_score(last_result)
        if _artifacts_are_syntactically_valid(artifacts, style=style) and (
            not best_artifacts or attempt_score < best_score
        ):
            best_score = attempt_score
            best_artifacts = list(artifacts)
        if last_result.passed:
            best_valid = list(artifacts)
            break
        if last_result.tier_1_static.diagnostic == last_error:
            repeated_static_failures += 1
        else:
            repeated_static_failures = 0
            last_error = last_result.tier_1_static.diagnostic
        if repeated_static_failures >= 2:
            break

    if last_result.passed and run_tier_3 is not None:
        tier_3_payload = run_tier_3(artifacts)
        last_result = _apply_tier_3_result(
            artifacts=artifacts,
            validation=last_result,
            tier_3_payload=tier_3_payload,
            style=style,
        )
        if (
            not last_result.passed
            and minimize_artifacts is not None
            and _tier_3_partially_passes(tier_3_payload)
        ):
            minimized, minimizer_diag = minimize_artifacts(artifacts, tier_3_payload)
            if minimized:
                artifacts = minimized
                minimized_static = _validate_artifacts_with_optional_tiers(
                    artifacts,
                    style=style,
                    api_probe=api_probe,
                    focal_module=focal_module,
                    workdir=workdir,
                    python_executable=python_executable,
                    run_import_validation=run_import_validation,
                    run_collect_validation=run_collect_validation,
                    validation_timeout_seconds=validation_timeout_seconds,
                    original_test_source=original_test_source,
                    splice_simulator=splice_simulator,
                )
                if minimized_static.passed:
                    minimized_tier_3 = run_tier_3(artifacts)
                    last_result = _apply_tier_3_result(
                        artifacts=artifacts,
                        validation=minimized_static,
                        tier_3_payload=minimized_tier_3,
                        style=style,
                        extra_diagnostics={"minimizer": minimizer_diag},
                    )
                    if last_result.passed:
                        last_result.diagnostics.setdefault(
                            "prediction_quality",
                            "minimized",
                        )

    if not last_result.passed and best_artifacts:
        artifacts = best_artifacts
        fallback_validation = _validate_artifacts_with_optional_tiers(
            artifacts,
            style=style,
            api_probe=api_probe,
            focal_module=focal_module,
            workdir=workdir,
            python_executable=python_executable,
            run_import_validation=run_import_validation,
            run_collect_validation=run_collect_validation,
            validation_timeout_seconds=validation_timeout_seconds,
            original_test_source=original_test_source,
            splice_simulator=splice_simulator,
        )
        fallback_diagnostics = dict(fallback_validation.diagnostics)
        fallback_diagnostics.update(
            {
                "prediction_quality": (
                    "fallback_last_valid"
                    if fallback_validation.passed
                    else "failed_after_repair_budget"
                ),
                "best_validation_score": best_score,
                "fallback_revalidated": True,
                "underlying_status": last_result.status,
                "underlying_tier_1_static": last_result.tier_1_static.to_dict(),
            }
        )
        if last_result.tier_2_import is not None:
            fallback_diagnostics["underlying_tier_2_import"] = last_result.tier_2_import.to_dict()
        if last_result.tier_2_collect is not None:
            fallback_diagnostics["underlying_tier_2_collect"] = last_result.tier_2_collect.to_dict()
        last_result = ValidationGateResult(
            status=(
                "fallback_last_valid"
                if fallback_validation.passed
                else "failed_after_repair_budget"
            ),
            artifacts=artifacts,
            tier_1_static=fallback_validation.tier_1_static,
            tier_2_import=fallback_validation.tier_2_import,
            tier_2_collect=fallback_validation.tier_2_collect,
            failure_classification=(
                fallback_validation.failure_classification or last_result.failure_classification
            ),
            repair_attempts=attempts,
            diagnostics=fallback_diagnostics,
        )
    else:
        last_result = ValidationGateResult(
            status=last_result.status,
            artifacts=artifacts,
            tier_1_static=last_result.tier_1_static,
            tier_2_import=last_result.tier_2_import,
            tier_2_collect=last_result.tier_2_collect,
            tier_3_run=last_result.tier_3_run,
            failure_classification=last_result.failure_classification,
            repair_attempts=attempts,
            diagnostics=dict(last_result.diagnostics),
        )
    return ValidationPipelineResult(
        artifacts=artifacts,
        validation=last_result,
        generation_diagnostics=generation_diagnostics,
        repair_diagnostics=repair_diagnostics,
        best_valid_artifacts=best_valid,
    )


def _validate_artifacts_with_optional_tiers(
    artifacts: list[dict[str, Any]],
    *,
    style: TestStyleProfile,
    api_probe: ApiProbeResult | None,
    focal_module: str,
    workdir: Path | None,
    python_executable: str | None,
    run_import_validation: bool,
    run_collect_validation: bool,
    validation_timeout_seconds: float,
    original_test_source: str = "",
    splice_simulator: SpliceSimulator | None = None,
) -> ValidationGateResult:
    static_result = validate_static_artifacts(
        artifacts,
        style=style,
        api_probe=api_probe,
        focal_module=focal_module,
        original_test_source=original_test_source,
        splice_simulator=splice_simulator,
    )
    if not static_result.passed or workdir is None:
        return static_result
    target_adapter = _active_target_environment_adapter()
    if target_adapter is not None and (run_import_validation or run_collect_validation):
        target_payload = _run_target_environment_validation(
            adapter=target_adapter,
            artifacts=artifacts,
            workdir=workdir,
            timeout_seconds=validation_timeout_seconds,
        )
        import_result = (
            ValidationTierResult(
                name="import",
                status="pass",
                diagnostic="target environment adapter owns import validation",
            )
            if run_import_validation
            and (style.language or "").lower() in {"python", "py", "python3"}
            else None
        )
        collect_result = (
            ValidationTierResult(
                name="collect",
                status=(
                    "fail" if target_payload.get("target_environment_collection_failed") else "pass"
                ),
                diagnostic=str(target_payload.get("diagnostic") or ""),
            )
            if run_collect_validation
            else None
        )
        diagnostics = dict(static_result.diagnostics)
        diagnostics.update(
            {
                "target_environment_validation": target_payload,
                "dynamic_validation_environment": "target_environment",
                "host_dynamic_validation": "disabled",
            }
        )
        if target_payload.get("status") != "pass":
            return ValidationGateResult(
                status="fail",
                artifacts=artifacts,
                tier_1_static=static_result.tier_1_static,
                tier_2_import=import_result,
                tier_2_collect=collect_result,
                failure_classification=classify_testgen_failure(
                    {"diagnostic": target_payload.get("diagnostic") or ""},
                    style=style,
                ),
                diagnostics=diagnostics,
            )
        return ValidationGateResult(
            status="pass",
            artifacts=artifacts,
            tier_1_static=static_result.tier_1_static,
            tier_2_import=import_result,
            tier_2_collect=collect_result,
            tier_3_run=target_payload,
            diagnostics=diagnostics,
        )
    import_result: ValidationTierResult | None = None
    if run_import_validation and (style.language or "").lower() in {"python", "py", "python3"}:
        import_result = import_validate_python_artifacts(
            workdir=workdir,
            artifacts=artifacts,
            timeout_seconds=validation_timeout_seconds,
            python_executable=python_executable,
        )
        if import_result.status == "fail":
            return ValidationGateResult(
                status="fail",
                artifacts=artifacts,
                tier_1_static=static_result.tier_1_static,
                tier_2_import=import_result,
                failure_classification=classify_testgen_failure(
                    {"diagnostic": import_result.diagnostic},
                    style=style,
                ),
                diagnostics=dict(static_result.diagnostics),
            )
    collect_result: ValidationTierResult | None = None
    if run_collect_validation:
        collect_result = collect_validate_artifacts(
            workdir=workdir,
            artifacts=artifacts,
            style=style,
            timeout_seconds=validation_timeout_seconds,
            python_executable=python_executable,
        )
        if collect_result.status == "fail":
            return ValidationGateResult(
                status="fail",
                artifacts=artifacts,
                tier_1_static=static_result.tier_1_static,
                tier_2_import=import_result,
                tier_2_collect=collect_result,
                failure_classification=classify_testgen_failure(
                    {"diagnostic": collect_result.diagnostic},
                    style=style,
                ),
                diagnostics=dict(static_result.diagnostics),
            )
    return ValidationGateResult(
        status="pass",
        artifacts=artifacts,
        tier_1_static=static_result.tier_1_static,
        tier_2_import=import_result,
        tier_2_collect=collect_result,
        diagnostics=dict(static_result.diagnostics),
    )


def _validation_failure_payload(validation: ValidationGateResult) -> dict[str, Any]:
    tier = validation.tier_1_static
    if validation.tier_2_import and validation.tier_2_import.status == "fail":
        tier = validation.tier_2_import
    if validation.tier_2_collect and validation.tier_2_collect.status == "fail":
        tier = validation.tier_2_collect
    return {
        "validation_tier": tier.name,
        "diagnostic": tier.diagnostic,
        "failure_class": (
            validation.failure_classification.failure_class.value
            if validation.failure_classification
            else None
        ),
    }


def _validation_attempt_score(validation: ValidationGateResult) -> tuple[int, int, int]:
    """Lower is better; used for fallback selection after repair exhaustion."""

    if validation.passed:
        return (0, 0, 0)
    diagnostics = " ".join(
        str(item or "")
        for item in (
            validation.tier_1_static.diagnostic,
            validation.tier_2_import.diagnostic if validation.tier_2_import else "",
            validation.tier_2_collect.diagnostic if validation.tier_2_collect else "",
        )
    )
    failed_tiers = int(validation.tier_1_static.status == "fail")
    failed_tiers += int(
        bool(validation.tier_2_import and validation.tier_2_import.status == "fail")
    )
    failed_tiers += int(
        bool(validation.tier_2_collect and validation.tier_2_collect.status == "fail")
    )
    return (failed_tiers or 1, len(diagnostics), len(validation.artifacts))


def _artifacts_are_syntactically_valid(
    artifacts: list[dict[str, Any]],
    *,
    style: TestStyleProfile,
) -> bool:
    if not artifacts:
        return False
    if (style.language or "").lower() not in {"python", "py", "python3"}:
        return True
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            return False
        try:
            ast.parse(str(artifact.get("content") or ""))
        except SyntaxError:
            return False
    return True


def _tier_3_partially_passes(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    value = payload.get("unfiltered_pass_at_1", payload.get("all_pass_at_1"))
    try:
        unfiltered = float(value or 0.0)
    except (TypeError, ValueError):
        unfiltered = 0.0
    per_test = dict(payload.get("per_test_status") or {})
    statuses = {str(status or "").lower() for status in per_test.values()}
    return 0.0 < unfiltered < 1.0 or ("pass" in statuses and bool(statuses & {"fail", "error"}))


def _apply_tier_3_result(
    *,
    artifacts: list[dict[str, Any]],
    validation: ValidationGateResult,
    tier_3_payload: dict[str, Any],
    style: TestStyleProfile,
    extra_diagnostics: dict[str, Any] | None = None,
) -> ValidationGateResult:
    payload = dict(tier_3_payload or {})
    value = payload.get("unfiltered_pass_at_1", payload.get("all_pass_at_1"))
    try:
        unfiltered = float(value or 0.0)
    except (TypeError, ValueError):
        unfiltered = 0.0
    diagnostics = dict(validation.diagnostics)
    diagnostics["tier_3_run"] = payload
    if extra_diagnostics:
        diagnostics.update(extra_diagnostics)
    if unfiltered >= 1.0:
        return ValidationGateResult(
            status="pass",
            artifacts=artifacts,
            tier_1_static=validation.tier_1_static,
            tier_2_import=validation.tier_2_import,
            tier_2_collect=validation.tier_2_collect,
            tier_3_run=payload,
            repair_attempts=validation.repair_attempts,
            diagnostics=diagnostics,
        )
    classification = classify_testgen_failure(payload, style=style)
    return ValidationGateResult(
        status="fail",
        artifacts=artifacts,
        tier_1_static=validation.tier_1_static,
        tier_2_import=validation.tier_2_import,
        tier_2_collect=validation.tier_2_collect,
        tier_3_run=payload,
        failure_classification=classification,
        repair_attempts=validation.repair_attempts,
        diagnostics=diagnostics,
    )
