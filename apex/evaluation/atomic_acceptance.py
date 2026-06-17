"""Per-test atomic acceptance helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .code_emission import roundtrip
from .final_acceptance_gate import GeneratedArtifact, _coerce_run, strict_syntax_check


@dataclass(frozen=True)
class AtomicAppendResult:
    status: str
    artifact_text: str
    diagnostic: dict[str, Any] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def append_test_atomically(
    base_artifact: str,
    candidate_test_source: str,
    *,
    benchmark_adapter: Any,
    workdir: Path,
    path: str = "tests/test_generated.py",
    language: str = "python",
) -> AtomicAppendResult:
    """Append a candidate test only if the resulting artifact passes."""

    try:
        candidate = roundtrip(candidate_test_source, language=language)
    except SyntaxError as exc:
        return AtomicAppendResult(
            status="rejected",
            artifact_text=str(base_artifact or ""),
            diagnostic={
                "error_class": "SyntaxError",
                "failing_assertion": "",
                "stdout_tail": "",
                "stderr_tail": "",
                "message": str(exc),
            },
        )
    combined = _join_sources(str(base_artifact or ""), candidate)
    # W3 strict syntax gate on the combined artifact: roundtrip may accept
    # the candidate in isolation but joining it to the base artifact can
    # produce module-level structures (orphan ``return``, mismatched
    # decorators) that compile() rejects. Short-circuit before adapter run.
    syntax_ok, syntax_error = strict_syntax_check(combined, filename=path)
    if not syntax_ok:
        return AtomicAppendResult(
            status="rejected",
            artifact_text=str(base_artifact or ""),
            diagnostic={
                "error_class": "SyntaxError",
                "failing_assertion": "",
                "stdout_tail": "",
                "stderr_tail": "",
                "message": syntax_error or "strict_syntax_check failed",
            },
        )
    run = _coerce_run(
        benchmark_adapter.run_unfiltered(
            GeneratedArtifact(path=path, content=combined),
            Path(workdir),
        )
    )
    telemetry = run.to_dict()
    failing = run.failing_test_names | run.errored_test_names
    if run.status in {"pass", "ok", "passed"} and not failing:
        return AtomicAppendResult(
            status="accepted",
            artifact_text=combined,
            telemetry=telemetry,
        )
    return AtomicAppendResult(
        status="rejected",
        artifact_text=str(base_artifact or ""),
        diagnostic={
            "error_class": _diagnostic_error_class(run),
            "stdout_tail": run.stdout_tail,
            "stderr_tail": run.stderr_tail,
            "failing_assertion": _first_assertion_line(run.stdout_tail + "\n" + run.stderr_tail),
            "message": run.diagnostic,
        },
        telemetry=telemetry,
    )


def _join_sources(base: str, candidate: str) -> str:
    parts = [part.strip() for part in (base, candidate) if str(part or "").strip()]
    return "\n\n".join(parts) + ("\n" if parts else "")


def _diagnostic_error_class(run: Any) -> str:
    text = "\n".join(
        [str(run.diagnostic or ""), str(run.stdout_tail or ""), str(run.stderr_tail or "")]
    )
    for marker in ("AssertionError", "NameError", "AttributeError", "TypeError", "SyntaxError"):
        if marker in text:
            return marker
    if run.errored_test_names:
        return "Error"
    if run.failing_test_names:
        return "AssertionError"
    return "Unknown"


def _first_assertion_line(text: str) -> str:
    for line in str(text or "").splitlines():
        if "assert " in line or "AssertionError" in line:
            return line.strip()[:500]
    return ""
