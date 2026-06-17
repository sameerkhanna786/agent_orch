"""
Breakpoint-driven debugging helper used behind the ACI debugger tool.
"""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from apex.core.subprocess_utils import terminate_process_tree


@dataclass
class DebugSummary:
    """Structured summary returned by the debug helper."""

    breakpoints_hit: list[str] = field(default_factory=list)
    variable_snapshots: dict[str, str] = field(default_factory=dict)
    call_stack_at_failure: str = ""
    root_cause_hypothesis: str = ""
    suggested_fix_location: str = ""

    def to_concise_string(self) -> str:
        lines = []
        if self.breakpoints_hit:
            lines.append("Breakpoints:")
            lines.extend(f"- {item}" for item in self.breakpoints_hit)
        if self.variable_snapshots:
            lines.append("Variables:")
            lines.extend(f"- {key} = {value}" for key, value in self.variable_snapshots.items())
        if self.call_stack_at_failure:
            lines.append("Failure stack:")
            lines.append(self.call_stack_at_failure.strip())
        if self.root_cause_hypothesis:
            lines.append(f"Hypothesis: {self.root_cause_hypothesis}")
        if self.suggested_fix_location:
            lines.append(f"Suggested fix location: {self.suggested_fix_location}")
        return "\n".join(lines).strip() or "Debugger found no useful runtime signal."


class DebugSubagent:
    """Ephemeral debug helper that injects a one-shot breakpoint and inspects locals."""

    MAX_INSPECTION_ROUNDS = 5

    def __init__(self, workspace: str, timeout: int = 60):
        self.workspace = Path(workspace).resolve()
        self.timeout = timeout

    def run(
        self,
        test_command: str,
        suspect_file: str,
        suspect_lines: list[int],
        hypothesis: str,
    ) -> DebugSummary:
        summary = DebugSummary()
        suspect_path = (self.workspace / suspect_file).resolve()
        if not suspect_path.exists():
            raw_output = self._run_plain_command(test_command)
            summary.call_stack_at_failure = self._extract_failure_stack(raw_output)
            summary.root_cause_hypothesis = (
                hypothesis or "Debugger could not locate the suspect file."
            )
            summary.suggested_fix_location = suspect_file
            return summary

        original_content = suspect_path.read_text(errors="replace")
        candidate_lines = suspect_lines or self._default_suspect_lines(original_content)
        summary.suggested_fix_location = self._suggest_fix_location(suspect_file, candidate_lines)

        combined_outputs: list[str] = []
        try:
            for round_index, line_number in enumerate(
                candidate_lines[: self.MAX_INSPECTION_ROUNDS], start=1
            ):
                instrumented = self._inject_breakpoint(original_content, line_number, round_index)
                if instrumented is None:
                    continue
                suspect_path.write_text(instrumented)
                session_output = self._run_debug_session(test_command)
                combined_outputs.append(session_output)
                hits = self._count_breakpoint_hits(session_output)
                summary.breakpoints_hit.append(
                    f"{suspect_file}:{line_number} - hit {hits} time{'s' if hits != 1 else ''}"
                )
                for name, value in self._parse_locals_snapshots(session_output).items():
                    summary.variable_snapshots[f"{suspect_file}:{line_number}:{name}"] = value
        finally:
            suspect_path.write_text(original_content)

        aggregate_output = "\n".join(block for block in combined_outputs if block).strip()
        if not aggregate_output:
            aggregate_output = self._run_plain_command(test_command)
        summary.call_stack_at_failure = self._extract_failure_stack(aggregate_output)

        failure_hint = (
            summary.call_stack_at_failure.strip().splitlines()[-1]
            if summary.call_stack_at_failure
            else ""
        )
        if hypothesis and failure_hint:
            summary.root_cause_hypothesis = f"{hypothesis} Observed failure: {failure_hint}"
        elif hypothesis:
            summary.root_cause_hypothesis = hypothesis
        elif failure_hint:
            summary.root_cause_hypothesis = failure_hint
        else:
            summary.root_cause_hypothesis = (
                "Runtime summary did not surface a precise failure site."
            )

        if not summary.variable_snapshots:
            source_lines = original_content.splitlines()
            for line_number in candidate_lines[:2]:
                if 1 <= line_number <= len(source_lines):
                    summary.variable_snapshots[f"{suspect_file}:{line_number}:source"] = (
                        source_lines[line_number - 1].strip()
                    )

        return summary

    def _run_debug_session(self, test_command: str) -> str:
        commands = "\n".join(
            [
                "where",
                "p {k: repr(v) for k, v in locals().items() if k != 'pdb'}",
                "next",
                "p {k: repr(v) for k, v in locals().items() if k != 'pdb'}",
                "where",
                "continue",
                "quit",
                "",
            ]
        )
        process = subprocess.Popen(
            ["bash", "-lc", test_command],
            cwd=str(self.workspace),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            output, _ = process.communicate(commands, timeout=self.timeout)
            return output.strip()
        except subprocess.TimeoutExpired:
            output, _ = terminate_process_tree(process)
            return (output or "").strip() + f"\nCommand timed out after {self.timeout} seconds."

    def _run_plain_command(self, test_command: str) -> str:
        try:
            result = subprocess.run(
                ["bash", "-lc", test_command],
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return "\n".join(part for part in [result.stdout, result.stderr] if part).strip()
        except subprocess.TimeoutExpired:
            return f"Command timed out after {self.timeout} seconds."

    def _inject_breakpoint(self, content: str, line_number: int, marker_index: int) -> str | None:
        lines = content.splitlines()
        if not (1 <= line_number <= len(lines)):
            return None
        target = lines[line_number - 1]
        indent = target[: len(target) - len(target.lstrip())]
        marker = f"__apex_debug_hit_{marker_index}"
        injection = [
            f"{indent}if not globals().get('{marker}', False):",
            f"{indent}    globals()['{marker}'] = True",
            f"{indent}    import pdb; pdb.set_trace()",
        ]
        updated_lines = list(lines)
        updated_lines[line_number - 1 : line_number - 1] = injection
        return "\n".join(updated_lines) + "\n"

    def _parse_locals_snapshots(self, output: str) -> dict[str, str]:
        snapshots: dict[str, str] = {}
        for line in output.splitlines():
            stripped = line.strip()
            if not stripped.startswith("(Pdb) {") or not stripped.endswith("}"):
                continue
            payload = stripped[len("(Pdb) ") :]
            try:
                values = ast.literal_eval(payload)
            except Exception:
                continue
            if isinstance(values, dict):
                for key, value in values.items():
                    snapshots[str(key)] = str(value)
        return snapshots

    def _extract_failure_stack(self, output: str) -> str:
        if not output:
            return ""
        lines = output.splitlines()
        traceback_start = None
        for index, line in enumerate(lines):
            if (
                line.startswith("Traceback ")
                or line.strip() == "Traceback (most recent call last):"
            ):
                traceback_start = index
        if traceback_start is not None:
            return "\n".join(lines[traceback_start:])
        last_block = lines[-20:]
        return "\n".join(last_block)

    def _suggest_fix_location(self, suspect_file: str, suspect_lines: list[int]) -> str:
        if suspect_lines:
            start = min(suspect_lines)
            end = max(suspect_lines)
            return f"{suspect_file}:{start}-{end}"
        return suspect_file

    def _default_suspect_lines(self, content: str) -> list[int]:
        lines = content.splitlines()
        candidates = [
            index
            for index, line in enumerate(lines, start=1)
            if line.strip() and not line.strip().startswith("#")
        ]
        if not candidates:
            return [1]
        midpoint = candidates[min(len(candidates) // 2, len(candidates) - 1)]
        return [midpoint]

    def _count_breakpoint_hits(self, output: str) -> int:
        return sum(1 for line in output.splitlines() if line.startswith("> "))
